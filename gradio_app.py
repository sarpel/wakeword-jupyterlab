#!/usr/bin/env python3
"""
Enhanced Wakeword Training Gradio Application
Complete GUI for wakeword detection model training with comprehensive documentation
"""

# IMPORTANT: Disable Torch Dynamo/ONNX paths to avoid ml_dtypes/onnx import issues on Windows
import os as _early_os
_early_os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
_early_os.environ.setdefault("PYTORCH_JIT", "0")

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
try:
    from torch.cuda.amp import GradScaler, autocast
except Exception:
    GradScaler = None
    autocast = None
import numpy as np
import librosa
import soundfile as sf
import os
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from collections import OrderedDict
import gradio as gr
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import threading
import time
import json
from datetime import datetime
import markdown
import contextlib
warnings.filterwarnings('ignore')

# Enforce CUDA-only training (no CPU fallback allowed)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required. CPU training is disabled by policy.")
try:
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")  # enable TF32 if supported
except Exception:
    pass

# Configuration Classes (same as before)
class AudioConfig:
    SAMPLE_RATE = 16000
    DURATION = 1.5  # seconds
    N_MELS = 96
    N_FFT = 2048
    HOP_LENGTH = 512
    WIN_LENGTH = 2048
    FMIN = 0
    FMAX = 8000

class ModelConfig:
    HIDDEN_SIZE = 512
    NUM_LAYERS = 2
    DROPOUT = 0.3
    NUM_CLASSES = 2

class TrainingConfig:
    BATCH_SIZE = 64
    LEARNING_RATE = 0.00001
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1
    WEIGHT_DECAY = 0.00001

class AugmentationConfig:
    AUGMENTATION_PROB = 0.6
    NOISE_FACTOR = 0.15
    TIME_SHIFT_MAX = 0.3
    PITCH_SHIFT_MAX = 1.5
    SPEED_CHANGE_MIN = 0.9
    SPEED_CHANGE_MAX = 1.1

# Audio Processing Class (same as before)
class AudioProcessor:
    def __init__(self, config=AudioConfig):
        self.config = config
        # Small LRU cache for preprocessed (normalized + padded) audio to reduce disk IO and librosa cost across epochs
        self._preproc_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._cache_size = 512  # ~512 * 1.7s * 16k * 4B ~= 55MB
        # torchaudio transforms (CPU) — faster than librosa; keep CPU to avoid returning CUDA tensors in Dataset
        try:
            self._ta_mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.SAMPLE_RATE,
                n_fft=self.config.N_FFT,
                win_length=self.config.WIN_LENGTH,
                hop_length=self.config.HOP_LENGTH,
                f_min=self.config.FMIN,
                f_max=self.config.FMAX,
                n_mels=self.config.N_MELS,
                center=True,
                power=2.0,
                normalized=False,
            )
            self._ta_db = torchaudio.transforms.AmplitudeToDB(stype='power')
            self.has_torchaudio = True
        except Exception:
            self._ta_mel = None
            self._ta_db = None
            self.has_torchaudio = False

    def load_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.config.SAMPLE_RATE)
            return audio
        except Exception as e:
            return None

    def normalize_audio(self, audio):
        if len(audio) == 0:
            return audio
        max_abs = np.max(np.abs(audio))
        if not np.isfinite(max_abs) or max_abs < 1e-8:
            return np.zeros_like(audio)
        normalized = audio / max_abs
        if np.isnan(normalized).any() or np.isinf(normalized).any():
            normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
        return normalized

    def pad_or_truncate(self, audio, target_length):
        if len(audio) > target_length:
            start_idx = random.randint(0, len(audio) - target_length)
            return audio[start_idx:start_idx + target_length]
        else:
            return np.pad(audio, (0, target_length - len(audio)), mode='constant')

    def get_preprocessed_audio(self, file_path):
        """Load, normalize, and pad/truncate audio with a small LRU cache.

        Caches the base (pre-augmentation) audio to avoid repeated disk IO and heavy decode.
        """
        key = file_path
        if key in self._preproc_cache:
            # Move to end (recently used)
            arr = self._preproc_cache.pop(key)
            self._preproc_cache[key] = arr
            return arr.copy()

        audio = self.load_audio(file_path)
        if audio is None:
            return None
        audio = self.normalize_audio(audio)
        target_length = int(self.config.SAMPLE_RATE * self.config.DURATION)
        audio = self.pad_or_truncate(audio, target_length)
        # Insert into LRU
        self._preproc_cache[key] = audio.astype(np.float32, copy=False)
        if len(self._preproc_cache) > self._cache_size:
            # Evict oldest
            self._preproc_cache.popitem(last=False)
        return audio

    def audio_to_mel(self, audio):
        """Convert audio to log-mel with a consistent, deterministic frame width.

        With librosa's default center=True, the number of frames is:
            1 + floor(L / hop_length)
        where L is the target sample length after pad/trim.
        We compute this expected width and pad/truncate mel features to match.
        """
        target_len = int(self.config.SAMPLE_RATE * self.config.DURATION)
        expected_frames = 1 + int(np.floor(target_len / self.config.HOP_LENGTH))

        if len(audio) == 0:
            return np.zeros((self.config.N_MELS, expected_frames), dtype=np.float32)

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(np.abs(audio)) < 1e-8:
            return np.zeros((self.config.N_MELS, expected_frames), dtype=np.float32)

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.config.SAMPLE_RATE, n_mels=self.config.N_MELS,
            n_fft=self.config.N_FFT, hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH, fmin=self.config.FMIN, fmax=self.config.FMAX
        )

        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Ensure consistent time dimension
        t = log_mel.shape[1]
        if t < expected_frames:
            pad = expected_frames - t
            log_mel = np.pad(log_mel, ((0,0),(0,pad)), mode='constant')
        elif t > expected_frames:
            log_mel = log_mel[:, :expected_frames]

        return log_mel.astype(np.float32)

    def audio_to_mel_ta(self, audio_np: np.ndarray) -> torch.Tensor:
        """Fast path: torchaudio-based mel + dB on CPU, returns torch tensor (n_mels, frames)."""
        target_len = int(self.config.SAMPLE_RATE * self.config.DURATION)
        expected_frames = 1 + int(np.floor(target_len / self.config.HOP_LENGTH))
        if audio_np is None or len(audio_np) == 0:
            return torch.zeros(self.config.N_MELS, expected_frames, dtype=torch.float32)
        x = torch.from_numpy(np.asarray(audio_np, dtype=np.float32))  # (T,)
        x = x.unsqueeze(0)  # (1, T)
        mel = self._ta_mel(x)  # (1, n_mels, frames)
        mel = mel.squeeze(0)
        mel_db = self._ta_db(mel)
        # Pad/truncate frames
        t = mel_db.shape[1]
        if t < expected_frames:
            pad = expected_frames - t
            mel_db = torch.nn.functional.pad(mel_db, (0, pad))
        elif t > expected_frames:
            mel_db = mel_db[:, :expected_frames]
        return mel_db.to(dtype=torch.float32)

    def augment_audio(self, audio, config=AugmentationConfig):
        """Lightweight augmentation tuned for speed.

        To avoid very slow first epochs, heavy operations like pitch shift and
        time-stretch are applied with much lower probability. We prioritize
        cheap ops (time shift, additive noise). This significantly reduces per
        sample compute time on CPU/Windows.
        """
        augmented_audio = audio.copy()

        # Cheap: time shift
        if random.random() < min(0.6, config.AUGMENTATION_PROB):
            shift_amount = int(random.uniform(-config.TIME_SHIFT_MAX, config.TIME_SHIFT_MAX) * self.config.SAMPLE_RATE)
            augmented_audio = np.roll(augmented_audio, shift_amount)

        # Heavy: pitch shift (lower probability for speed)
        if random.random() < 0.15:  # was ~0.85
            try:
                n_steps = random.uniform(-config.PITCH_SHIFT_MAX, config.PITCH_SHIFT_MAX)
                augmented_audio = librosa.effects.pitch_shift(y=augmented_audio, sr=self.config.SAMPLE_RATE, n_steps=n_steps)
            except Exception:
                pass

        # Heavy: time stretch (lower probability for speed)
        if random.random() < 0.15:  # was ~0.85
            try:
                speed_factor = random.uniform(config.SPEED_CHANGE_MIN, config.SPEED_CHANGE_MAX)
                augmented_audio = librosa.effects.time_stretch(y=augmented_audio, rate=speed_factor)
                augmented_audio = self.pad_or_truncate(augmented_audio, len(audio))
            except Exception:
                pass

        # Cheap: additive noise
        if random.random() < min(0.7, config.AUGMENTATION_PROB):
            noise = np.random.normal(0, config.NOISE_FACTOR, len(augmented_audio))
            augmented_audio = augmented_audio + noise

        return augmented_audio

    def process_audio_file(self, file_path, augment=False):
        audio = self.load_audio(file_path)
        if audio is None:
            return None

        audio = self.normalize_audio(audio)
        target_length = int(self.config.SAMPLE_RATE * self.config.DURATION)
        audio = self.pad_or_truncate(audio, target_length)

        if augment:
            audio = self.augment_audio(audio)

        return self.audio_to_mel(audio)

# Neural Network Model (same as before)
class WakewordModel(nn.Module):
    def __init__(self, config=ModelConfig, audio_config=AudioConfig):
        super(WakewordModel, self).__init__()
        self.config = config
        self.audio_config = audio_config

        self.mel_height = audio_config.N_MELS
        self.mel_width = int(audio_config.SAMPLE_RATE * audio_config.DURATION / audio_config.HOP_LENGTH) + 1

        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cnn_output_size = 128

        self.lstm = nn.LSTM(
            input_size=self.cnn_output_size, hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS, batch_first=True,
            dropout=config.DROPOUT if config.NUM_LAYERS > 1 else 0
        )

        self.dropout = nn.Dropout(config.DROPOUT)
        self.fc = nn.Linear(config.HIDDEN_SIZE, config.NUM_CLASSES)

    def forward(self, x):
        batch_size = x.size(0)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.pool(x)

        x = x.view(batch_size, -1)
        x = x.unsqueeze(1)

        lstm_out, (h_n, c_n) = self.lstm(x)
        x = lstm_out[:, -1, :]

        x = self.dropout(x)
        x = self.fc(x)

        return x

# Dataset Class (same as before)
class EnhancedWakewordDataset(Dataset):
    def __init__(self, wakeword_files, hard_negative_files, random_negative_files,
                 background_files, processor, augment=False,
                 background_mix_prob=0.5, snr_range=(0, 20)):

        self.processor = processor
        self.augment = augment
        self.background_mix_prob = background_mix_prob
        self.snr_range = snr_range

        self.wakeword_files = wakeword_files
        self.hard_negative_files = hard_negative_files
        self.random_negative_files = random_negative_files
        self.background_files = background_files

        self.background_cache = self._cache_background_segments()
        self._create_balanced_dataset()

    def _cache_background_segments(self, max_cache_size=100):
        """Cache background segments trimmed to target length for fast mixing.

        Previously we cached full background files which could be very long,
        increasing memory/IO and slowing mixing. We now normalize and
        pad/trim to the target training length up-front.
        """
        cache = []
        target_length = int(self.processor.config.SAMPLE_RATE * self.processor.config.DURATION)
        for i, bg_file in enumerate(self.background_files[:max_cache_size]):
            try:
                audio = self.processor.load_audio(bg_file)
                if audio is not None and len(audio) > 0:
                    # Normalize and trim/pad to target length
                    audio = self.processor.normalize_audio(audio)
                    audio = self.processor.pad_or_truncate(audio, target_length)
                    # Final safety scaling
                    max_abs = np.max(np.abs(audio))
                    if np.isfinite(max_abs) and max_abs > 1e-8:
                        audio = (audio / max_abs) * 0.95
                    cache.append(audio.astype(np.float32, copy=False))
            except Exception:
                pass
        return cache

    def _create_balanced_dataset(self):
        n_wakeword = len(self.wakeword_files)
        n_hard_neg = min(len(self.hard_negative_files), int(n_wakeword * 4.5))
        n_random_neg = min(len(self.random_negative_files), int(n_wakeword * 8.75))
        n_background_pure = min(len(self.background_files), int(n_wakeword * 10))

        random.seed(42)
        sampled_hard_neg = random.sample(self.hard_negative_files, n_hard_neg) if n_hard_neg > 0 else []
        sampled_random_neg = random.sample(self.random_negative_files, n_random_neg) if n_random_neg > 0 else []
        sampled_background = random.sample(self.background_files, n_background_pure) if n_background_pure > 0 else []

        self.files = []
        self.labels = []
        self.categories = []

        for f in self.wakeword_files:
            self.files.append(f); self.labels.append(1); self.categories.append('wakeword')
        for f in sampled_hard_neg:
            self.files.append(f); self.labels.append(0); self.categories.append('hard_negative')
        for f in sampled_random_neg:
            self.files.append(f); self.labels.append(0); self.categories.append('random_negative')
        for f in sampled_background:
            self.files.append(f); self.labels.append(0); self.categories.append('background')

        indices = list(range(len(self.files)))
        random.shuffle(indices)
        self.files = [self.files[i] for i in indices]
        self.labels = [self.labels[i] for i in indices]
        self.categories = [self.categories[i] for i in indices]

    def _mix_with_background(self, audio, target_snr_db=None):
        if len(self.background_cache) == 0:
            return audio

        bg_audio = random.choice(self.background_cache)
        target_len = len(audio)

        if len(bg_audio) > target_len:
            start_idx = random.randint(0, len(bg_audio) - target_len)
            bg_segment = bg_audio[start_idx:start_idx + target_len]
        else:
            bg_segment = np.tile(bg_audio, (target_len // len(bg_audio) + 1))[:target_len]

        if target_snr_db is None:
            target_snr_db = random.uniform(self.snr_range[0], self.snr_range[1])

        signal_power = np.mean(audio ** 2)
        noise_power = np.mean(bg_segment ** 2)

        if noise_power > 0:
            noise_scale = np.sqrt(signal_power / (10 ** (target_snr_db / 10)) / noise_power)
            mixed = audio + noise_scale * bg_segment
            max_val = np.max(np.abs(mixed))
            if max_val > 0:
                mixed = mixed / max_val * 0.95
            return mixed

        return audio

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]
        category = self.categories[idx]

        audio = self.processor.get_preprocessed_audio(file_path)

        if audio is None:
            mel_spec = np.zeros((self.processor.config.N_MELS, 31), dtype=np.float32)
        else:
            if self.augment:
                audio = self.processor.augment_audio(audio)

            if category != 'background' and random.random() < self.background_mix_prob:
                audio = self._mix_with_background(audio)

            if getattr(self.processor, 'has_torchaudio', False) and self.processor._ta_mel is not None:
                mel_t = self.processor.audio_to_mel_ta(audio)
                mel_tensor = mel_t.unsqueeze(0)  # (1, n_mels, frames)
            else:
                mel_spec = self.processor.audio_to_mel(audio)
                mel_array = np.ascontiguousarray(mel_spec, dtype=np.float32)
                mel_tensor = torch.from_numpy(mel_array).unsqueeze(0)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return mel_tensor, label_tensor

# Training Class (same as before)
class WakewordTrainer:
    def __init__(self, model, device, config=TrainingConfig):
        self.model = model
        self.device = device
        self.config = config

        class_weights = torch.tensor([1.0, 2.5]).to(device)  # [negative, wakeword]
        self.criterion = nn.CrossEntropyLoss(weight=class_weights).to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor=0.5, patience=5)

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        self.patience = 10
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0

        self.current_epoch = 0
        self.is_training = False
        self.training_complete = False
        self._last_perf = ""

        # Pause/Resume support
        self.paused = False
        import threading as _thr
        self._pause_event = _thr.Event()
        self._pause_event.set()  # allow running by default

        # Gradient clipping configurable
        self.grad_clip_max_norm = 1.0
        # Mixed precision (CUDA only)
        self.use_amp = True
        self.scaler = GradScaler(enabled=self.use_amp) if GradScaler else None

    def pause(self):
        self.paused = True
        self._pause_event.clear()

    def resume(self):
        self.paused = False
        self._pause_event.set()

    def _save_last_checkpoint(self, epoch, train_loss, train_acc, val_loss, val_acc, path='last_checkpoint.pth'):
        try:
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                'val_acc': val_acc,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_acc': self.best_val_acc,
                'epochs_no_improve': self.epochs_no_improve,
                'config': {
                    'LEARNING_RATE': self.config.LEARNING_RATE,
                    'BATCH_SIZE': self.config.BATCH_SIZE,
                    'EPOCHS': self.config.EPOCHS,
                }
            }, path)
        except Exception:
            pass

    def load_last_checkpoint(self, path='last_checkpoint.pth'):
        if not os.path.exists(path):
            raise FileNotFoundError("Checkpoint bulunamadı: last_checkpoint.pth")
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        if ckpt.get('optimizer_state_dict') is not None:
            self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if ckpt.get('scheduler_state_dict') and self.scheduler:
            try:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            except Exception:
                pass
        self.best_val_acc = float(ckpt.get('best_val_acc', self.best_val_acc))
        self.epochs_no_improve = int(ckpt.get('epochs_no_improve', self.epochs_no_improve))
        # Resume from next epoch index
        return int(ckpt.get('epoch', -1)) + 1

    def train_epoch(self, train_loader, progress_callback=None):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        # Performance timing
        import time as _tm
        last_end = _tm.perf_counter()
        sum_data = 0.0
        sum_h2d = 0.0
        sum_compute = 0.0
        seen = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            # Handle pause/resume
            self._pause_event.wait()
            if not self.is_training:
                break

            # Data time (time to receive next batch)
            t_now = _tm.perf_counter()
            data_time = t_now - last_end

            # Host→Device transfer
            t_h2d0 = _tm.perf_counter()
            data = data.to(self.device, non_blocking=True)
            target = target.to(self.device, non_blocking=True).squeeze()
            t_h2d1 = _tm.perf_counter()

            self.optimizer.zero_grad(set_to_none=True)
            ctx = autocast(enabled=self.use_amp) if autocast else contextlib.nullcontext()
            with ctx:
                output = self.model(data)
                loss = self.criterion(output, target)

            # Compute + backward + step
            t_comp0 = _tm.perf_counter()
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                # Unscale before clipping to ensure correct gradient norms
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_max_norm)
                self.optimizer.step()
            t_comp1 = _tm.perf_counter()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Update perf accumulators
            sum_data += data_time
            sum_h2d += (t_h2d1 - t_h2d0)
            sum_compute += (t_comp1 - t_comp0)
            seen += 1

            if seen % 20 == 0:
                avg_data = sum_data / max(seen, 1)
                avg_h2d = sum_h2d / max(seen, 1)
                avg_comp = sum_compute / max(seen, 1)
                self._last_perf = (
                    f"Avg times (s) — data: {avg_data:.4f}, h2d: {avg_h2d:.4f}, compute: {avg_comp:.4f}; "
                    f"ratio compute/iter: {avg_comp / max((avg_data+avg_h2d+avg_comp), 1e-9):.2f}"
                )

            if progress_callback and batch_idx % 10 == 0:
                progress = (batch_idx + 1) / len(train_loader) * 100
                progress_callback(progress, batch_idx + 1, len(train_loader))

            last_end = _tm.perf_counter()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                # Allow pause during validation too
                self._pause_event.wait()
                if not self.is_training:
                    break
                data = data.to(self.device, non_blocking=True)
                target = target.to(self.device, non_blocking=True).squeeze()
                ctx = autocast(enabled=self.use_amp) if autocast else contextlib.nullcontext()
                with ctx:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, epochs, progress_callback=None,
              auto_extend=False, extend_step=5, max_extra_epochs=0,
              plateau_delta=0.001, worsen_patience=3):
        # Hard assert: model and tensors must be on CUDA
        if not torch.cuda.is_available() or next(self.model.parameters()).device.type != 'cuda':
            raise RuntimeError("CUDA device check failed: model must be on CUDA and CUDA must be available.")
        self.is_training = True
        self.training_complete = False
        self._pause_event.set()  # ensure running

        initial_epochs = int(epochs)
        target_epochs = int(epochs)
        extra_used = 0
        self.config.EPOCHS = target_epochs  # for UI

        print(f"Starting training for {target_epochs} epochs...")

        worsening_streak = 0
        last_val_loss = None

        epoch = 0
        while epoch < target_epochs and self.is_training:
            self.current_epoch = epoch + 1

            train_loss, train_acc = self.train_epoch(train_loader, progress_callback)
            val_loss, val_acc = self.validate(val_loader)

            # Save progress
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            self.scheduler.step(val_acc)

            # Console log for headless supervision
            try:
                print(f"[Epoch {self.current_epoch}/{target_epochs}] Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | Best Val Acc: {self.best_val_acc:.2f}%")
            except Exception:
                pass

            # Save checkpoints
            self._save_last_checkpoint(epoch, train_loss, train_acc, val_loss, val_acc)

            # Best model tracking
            if val_acc > self.best_val_acc + 1e-9:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_acc': val_acc,
                        'train_acc': train_acc,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                    }, 'best_wakeword_model.pth')
                except Exception:
                    pass
                # Auto-extend if enabled and nearing end
                if auto_extend and extra_used < max_extra_epochs and (epoch + 1) >= target_epochs:
                    add_epochs = min(extend_step, max_extra_epochs - extra_used)
                    target_epochs += add_epochs
                    self.config.EPOCHS = target_epochs
                    extra_used += add_epochs
                    print(f"Auto-extend: increasing target epochs to {target_epochs} (added {add_epochs})")
            else:
                self.epochs_no_improve += 1

            # Worsening detection
            if last_val_loss is not None and val_loss > last_val_loss + plateau_delta:
                worsening_streak += 1
            else:
                worsening_streak = 0
            last_val_loss = val_loss

            # Early stopping
            if self.epochs_no_improve >= self.patience:
                print("Early stopping: no improvement within patience.")
                break
            if worsen_patience and worsening_streak >= worsen_patience:
                print("Stopping: consecutive validation loss worsening detected.")
                break

            epoch += 1

        self.is_training = False
        self.training_complete = True
        return self.best_val_acc

# Enhanced Training Guide Content
COMPREHENSIVE_TRAINING_GUIDE = """
# 🎯 WAKEWORD DETECTION COMPLETE TRAINING GUIDE

## 📊 VERİ SETİ HAZIRLAMA - DETAYLI ANALİZ

### 🎤 POZİTİF VERİLER (WAKEWORD KAYITLARI)

#### Miktar ve Kalite Gereksinimleri
- **Minimum Miktar**: 100-500 temiz wakeword kaydı
- **İdeal Miktar**: 1000+ wakeword kaydı
- **SNR**: ≥ 20dB (sinyal/gürültü oranı)
- **Süre**: 1-2 saniye (optimum 1.7s)
- **Format**: WAV, 16-bit, 16kHz

#### Çeşitlilik Gereksinimleri
**🎤 Mikrofon Çeşitliliği:**
• Smartphone, USB, Bluetooth, laptop, profesyonel
• Farklı marka ve modeller
• Farklı sampling kaliteleri

**👥 Konuşan Çeşitliliği:**
• Erkek/Kadın/Çocuk sesleri
• Farklı yaş grupları (18-65 yaş)
• Farklı aksanlar ve lehçeler
• Farklı konuşma hızları (yavaş/hızlı)
• Farklı ses tonları (yüksek/alçak)

**🌍 Ortam Çeşitliliği:**
• Sessiz oda (SNR > 30dB)
• Ofis ortamı (SNR 20-25dB)
• Dış mekan (SNR 15-20dB)
• Araba içi (SNR 10-15dB)
• Kafe/restoran (SNR 5-10dB)

### 🔊 NEGATİF VERİLER

**A. Hard Negative Samples (Fonetik Benzer):**
• Miktar: Her wakeword için 4-5 sample
• Örnekler: "hey"→"hey computer", "ok"→"okay", "day"→"they"
• Phonetically benzer kelimeler seçilmeli
• Konuşma benzerliği yüksek olmalı

**B. Random Negative Samples:**
• Miktar: Her wakeword için 8-9 sample
• Türler: Günlük konuşmalar, telefon görüşmeleri, radyo/TV
• Çeşitlilik: Farklı diller, aksanlar, ortamlar
• Süre: 1-3 saniye arası

**C. Background Noise Samples:**
• Miktar: Minimum 66 saat, ideal 100+ saat
• Türler: Beyaz/pembe/kahverengi gürültü, fan sesi, trafik
• Çeşitlilik: Farklı SNR seviyeleri (0-30dB)
• Format: Yüksek kaliteli kayıtlar

### ⚖️ İDEAL VERİ DAĞILIMI
```
1 wakeword : 4.5 hard_negative : 8.75 random_negative : 10 background

Örnek (100 wakeword için):
• Wakeword: 100 samples (%4.2)
• Hard Negative: 450 samples (%18.8)
• Random Negative: 875 samples (%36.5)
• Background: 1000 samples (%41.7)
• TOPLAM: 2425 samples
```

## 🎵 SES KALİTESİ TEKNİK KRİTERLER

- **Sample Rate**: 16kHz (insan sesi için optimum)
- **Bit Depth**: 16-bit veya üzeri
- **Format**: WAV (kayıpsız), FLAC (sıkıştırılmış)
- **SNR**: Minimum 20dB, ideal 30dB+
- **Clipping**: -3dB'den fazla olmamalı
- **Phase**: Doğrusal faz响应ı
- **Dynamic Range**: En az 60dB

## 🔄 DETAYLI VERİ ARTIRMA TEKNİKLERİ

### 1. TIME SHIFTING (Zaman Kaydırma)
- **Aralık**: ±0.3 saniye (±4800 sample)
- **Amaç**: Farklı zamanlama senaryoları
- **Uygulama**: np.roll ile dairesel kaydırma
- **Limit**: Ses sınırları içinde kalmalı

### 2. PITCH SHIFTING (Perde Değiştirme)
- **Aralık**: ±1.5 semiton (%18 frekans değişimi)
- **Teknik**: PSOLA algoritması ile doğal değişim
- **Etki**: Sadece perdeyi değiştirir, süreyi korur
- **Doğallık**: İnsan kulağına doğal gelen aralık

### 3. SPEED CHANGING (Hız Değiştirme)
- **Aralık**: 0.9x - 1.1x (%10 hız değişimi)
- **Etkiler**: Hem perdeyi hem süreyi değiştirir
- **Kalite**: PSOLA ile doğallık korunur
- **Not**: Orijinal uzunluk korunur

### 4. BACKGROUND NOISE MIXING
- **SNR Range**: 0-20dB (training için)
- **SNR Range**: 5-15dB (validation için)
- **Olasılık**: %70 karıştırma oranı
- **Teknik**: Sinyal gücüne göre ölçeklendirme

### 5. ADDITIVE NOISE
- **Türler**: Beyaz, pembe, kahverengi gürültü
- **Seviye**: %15 sinyal gücü
- **Amaç**: Sensitivite artırma
- **Uygulama**: Gaussian gürültü ekleme

## 🧠 MODEL MİMARİSİ DETAYLI AÇIKLAMA

### CNN+LSTM Mimarisi
```
Input → Conv2D(1→32) → ReLU → Conv2D(32→64) → ReLU →
Conv2D(64→128) → ReLU → AdaptiveAvgPool2d →
Flatten → LSTM(128→256×2) → Dropout(0.6) → Linear(256→2)
```

### Katman Detayları
- **Input**: (batch, 1, 80, 31) - 80 mel bands, 31 time frames
- **Conv Layers**: 320 + 18,496 + 73,856 = 92,672 parameters
- **LSTM**: 788,992 parameters (2 layers, 256 hidden)
- **Total**: ~882K parameters

### 🎯 HIDDEN SIZE SEÇİMİ
- **128**: Küçük veri setleri, hızlı eğitim
- **256**: Dengeli performans (OPTIMUM)
- **512**: Büyük veri setleri, daha iyi accuracy
- **1024**: Çok büyük veri setleri, yavaş eğitim

## 📈 TRAINING SÜRECİ DETAYLI ANLATIM

### 1. VERİ YÜKLEME PIPELINE
Audio → Normalize → Pad/Truncate → Mel-Spec → Log → Tensor
- **Batch Size**: 32 (memory ve gradient dengesi)
- **Num Workers**: 2-4 (paralel loading)
- **Pin Memory**: True (GPU hızlandırma)

### 2. FORWARD PASS
Model forward → Loss calculation → Accuracy computation
- **CrossEntropyLoss**: Automatic softmax + NLL loss
- **Gradient calculation**: Autograd ile otomatik

### 3. BACKWARD PASS
Loss.backward() → Gradient clipping → Optimizer.step()
- **Gradient clipping**: max_norm=1.0 (patlama önleme)
- **Learning rate scheduling**: ReduceLROnPlateau

### 4. EARLY STOPPING
- **Patience**: 10 epoch (improvement yoksa dur)
- **Min Delta**: 0.001 (minimum improvement)
- **Mode**: max (validation accuracy'yi izle)

## ⚙️ KONFİGÜRASYON PARAMETRELERİ OPTİMİZASYONU

### AUDIO CONFIG
- **SAMPLE_RATE (16kHz)**: İnsan sesi için optimum
- **DURATION (1.7s)**: Wakeword'u tam kapsar
- **N_MELS (80)**: Dengeli frekans çözünürlüğü
- **N_FFT (2048)**: 7.8Hz frekans çözünürlüğü
- **HOP_LENGTH (512)**: 32ms zaman çözünürlüğü

### MODEL CONFIG
- **HIDDEN_SIZE (256)**: Dengeli model kapasitesi
- **NUM_LAYERS (2)**: Uygun derinlik
- **DROPOUT (0.6)**: Overfitting önleme
- **NUM_CLASSES (2)**: Binary classification

### TRAINING CONFIG
- **BATCH_SIZE (32)**: Memory ve gradient dengesi
- **LEARNING_RATE (0.0001)**: Stabil öğrenme hızı
- **EPOCHS (100)**: Maksimum eğitim süresi
- **VALIDATION_SPLIT (0.2)**: Dengeli validation

### AUGMENTATION CONFIG
- **AUGMENTATION_PROB (0.85)**: Dengeli artırma oranı
- **NOISE_FACTOR (0.15)**: Orta seviye gürültü
- **TIME_SHIFT_MAX (0.3)**: Uygun zaman kaydırma
- **PITCH_SHIFT_MAX (1.5)**: Doğal perde değişimi

## 📊 PERFORMANS METRİKLERİ VE YORUMLAMA

### ACCURACY METRİKLERİ
- **Accuracy**: Doğru tahmin oranı
- **Precision**: Pozitif tahminlerin doğruluğu
- **Recall**: Pozitiflerin ne kadarı yakalandı
- **F1-Score**: Precision ve recall harmonik ortalaması

### CONFUSION MATRIX YORUMLAMA
- **True Positive**: Doğru wakeword tespiti
- **False Positive**: Yanlış wakeword tespiti
- **True Negative**: Doğru negative tespiti
- **False Negative**: Kaçırılan wakeword

### PERFORMANS STANDARTLARI
- **Mükemmel**: F1-Score ≥ 0.90
- **Çok İyi**: F1-Score ≥ 0.85
- **İyi**: F1-Score ≥ 0.80
- **Orta**: F1-Score ≥ 0.70
- **Zayıf**: F1-Score < 0.70

## 🚨 YAYGIN SORUNLAR VE ÇÖZÜMLER

### OVERFITTING BELİRTİLERİ
- Train accuracy %95+, validation accuracy düşüyor
- Train loss azalıyor, validation loss artıyor
- **Çözümler**: Dropout artır, augmentation güçlendir, early stopping

### UNDERFITTING BELİRTİLERİ
- Train ve validation accuracy düşük (<%70)
- Loss yüksek ve azalmıyor
- **Çözümler**: Model kapasitesini artır, learning rate artır

### GRADIENT EXPLOSION
- Loss anında büyük değerler, NaN/Inf
- **Çözümler**: Gradient clipping, learning rate azalt, batch norm

### MEMORY ERROR
- CUDA out of memory hatası
- **Çözümler**: Batch size küçült, mixed precision, gradient accumulation

## 💡 EN İYİ UYGULAMALAR VE TAVSİYELER

### VERİ KALİTESİ
- Her wakeword'u 3-5 kez kaydet
- Farklı ortamlarda kayıt yap
- Mikrofon kalitesine dikkat et
- Clipping kontrolü yap

### MODEL GELİŞTİRME
- Cross-validation kullan (5-fold)
- Hyperparameter tuning yap
- A/B testing ile karşılaştır
- Model checkpoint'lerini kaydet

### TRAINING STRATEJİSİ
- Learning rate scheduling kullan
- Early stopping implemente et
- Gradient clipping uygula
- Mixed precision kullan (GPU için)

## 🎯 BAŞARILI MODEL İÇİN GEREKSİNİMLER

### Minimum
- 100+ wakeword kaydı
- %85+ validation accuracy
- %80+ test accuracy
- Dengeli veri seti

### İdeal
- 1000+ wakeword kaydı
- %90+ validation accuracy
- %85+ test accuracy
- Kapsamlı çeşitlilik

---

## 📚 DAHA FAZLA BİLGİ

Detaylı bilgi için `COMPREHENSIVE_TRAINING_GUIDE.md` dosyasına bakın.
Tüm teknik detaylar, optimizasyon stratejileri ve örnek kodlar mevcuttur.
"""

# Global variables
trainer = None
processor = None
model = None
device = None
training_thread = None

class WakewordTrainingApp:
    def __init__(self):
        self.device = torch.device('cuda')
        self.processor = AudioProcessor()
        self.model = WakewordModel().to(self.device)
        self.trainer = WakewordTrainer(self.model, self.device)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
        # Live batch progress (updated via update_progress callback during training)
        self._batch_progress = 0.0
        self._current_batch = 0
        self._total_batches = 0

        # Live metrics text
        self._last_metrics_text = ""

        # Keep last data load parameters to allow rebuilding loaders if needed
        self._last_data_params = None  # tuple: (positive_dir, negative_dir, background_dir, batch_size, val_split, test_split, bg_mix_prob, snr_min, snr_max)

    # --- Config application helpers ---
    def apply_audio_config(self, sample_rate: int, duration: float, n_mels: int):
        """Apply audio config to global config and rebuild processor.

        This keeps defaults unless overridden by UI, and refreshes internal
        torchaudio transforms and caches for new shapes.
        """
        # Update global audio configuration
        try:
            AudioConfig.SAMPLE_RATE = int(sample_rate)
            AudioConfig.DURATION = float(duration)
            AudioConfig.N_MELS = int(n_mels)
        except Exception:
            pass

        # Re-create processor to pick up new settings and clear caches
        self.processor = AudioProcessor(config=AudioConfig)

    def apply_augmentation_config(self, aug_prob: float, noise_factor: float, time_shift_s: float, pitch_shift_semitones: float):
        """Apply augmentation hyperparameters to the shared AugmentationConfig.

        EnhancedWakewordDataset uses AudioProcessor.augment_audio which reads
        AugmentationConfig defaults; updating these reflects on next dataset processing.
        """
        try:
            AugmentationConfig.AUGMENTATION_PROB = float(aug_prob)
            AugmentationConfig.NOISE_FACTOR = float(noise_factor)
            AugmentationConfig.TIME_SHIFT_MAX = float(time_shift_s)
            AugmentationConfig.PITCH_SHIFT_MAX = float(pitch_shift_semitones)
        except Exception:
            pass

    def rebuild_model(self, hidden_size: int, num_layers: int, dropout: float):
        """Rebuild the model and trainer with the selected architecture.

        Must be called before (re)starting training when model settings change.
        """
        try:
            ModelConfig.HIDDEN_SIZE = int(hidden_size)
            ModelConfig.NUM_LAYERS = int(num_layers)
            ModelConfig.DROPOUT = float(dropout)
        except Exception:
            pass

        # Build a fresh model that derives mel dims from current AudioConfig
        self.model = WakewordModel(config=ModelConfig, audio_config=AudioConfig).to(self.device)
        # Recreate trainer with same device
        self.trainer = WakewordTrainer(self.model, self.device)
        # Ensure dropout reflects UI
        try:
            self.model.dropout.p = float(dropout)
        except Exception:
            pass

    # Control methods
    def pause_training(self):
        if self.trainer and self.trainer.is_training:
            self.trainer.pause()
            return "Eğitim duraklatıldı"
        return "Eğitim zaten duraklatılmış veya başlamadı"

    def resume_training(self):
        if self.trainer and self.trainer.paused:
            self.trainer.resume()
            return "Eğitim devam ettirildi"
        return "Devam ettirilecek bir eğitim yok"

    def continue_from_checkpoint(self, epochs=None):
        try:
            start_next_epoch = self.trainer.load_last_checkpoint('last_checkpoint.pth')
            target_epochs = epochs if epochs is not None else self.trainer.config.EPOCHS

            def training_thread():
                self.trainer.train(self.train_loader, self.val_loader, target_epochs, self.update_progress)

            thread = threading.Thread(target=training_thread)
            thread.daemon = True
            thread.start()
            return f"Checkpoint'ten devam ediliyor (başlayacağı epoch: {start_next_epoch})."
        except Exception as e:
            return f"Devam edilemedi: {e}"

    def load_data(self, positive_dir, negative_dir, background_dir, batch_size, val_split, test_split,
                  background_mix_prob: float = 0.7, snr_min: float = 0.0, snr_max: float = 20.0):
        try:
            # Load audio files
            def load_audio_files(directory, extensions=['*.wav', '*.mp3', '*.flac']):
                files = []
                for ext in extensions:
                    files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
                return files

            wakeword_files = load_audio_files(positive_dir)
            negative_files = load_audio_files(negative_dir)
            background_files = load_audio_files(background_dir)

            if len(wakeword_files) == 0:
                return f"Hata: {positive_dir} klasöründe wakeword dosyası bulunamadı!", None, None

            # Split data
            wakeword_train, wakeword_test = train_test_split(wakeword_files, test_size=test_split, random_state=42)
            wakeword_train, wakeword_val = train_test_split(wakeword_train, test_size=val_split/(1-test_split), random_state=42)

            negative_train, negative_test = train_test_split(negative_files, test_size=test_split, random_state=42)
            negative_train, negative_val = train_test_split(negative_train, test_size=val_split/(1-test_split), random_state=42)

            # Optional small-caps for smoke tests (via env vars)
            # DATASET_CAP limits the number of samples per split for wakeword and negative sets
            # BACKGROUND_CAP limits background file count
            _cap = os.environ.get('DATASET_CAP')
            _bg_cap = os.environ.get('BACKGROUND_CAP')
            if _cap is not None:
                try:
                    _cap = int(_cap)
                    if _cap > 0:
                        wakeword_train = wakeword_train[:_cap]
                        wakeword_val = wakeword_val[:max(1, _cap // 5)]
                        wakeword_test = wakeword_test[:max(1, _cap // 5)]

                        negative_train = negative_train[:_cap * 2]
                        negative_val = negative_val[:max(1, (_cap * 2) // 5)]
                        negative_test = negative_test[:max(1, (_cap * 2) // 5)]
                except Exception:
                    pass

            if _bg_cap is not None:
                try:
                    _bg_cap = int(_bg_cap)
                    if _bg_cap > 0:
                        background_files = background_files[:_bg_cap]
                except Exception:
                    pass

            # İsim bazlı hard/random negative ayırma
            hard_negative_train = [f for f in negative_train if 'similar_' in os.path.basename(f)]
            random_negative_train = [f for f in negative_train if 'similar_' not in os.path.basename(f)]

            hard_negative_val = [f for f in negative_val if 'similar_' in os.path.basename(f)]
            random_negative_val = [f for f in negative_val if 'similar_' not in os.path.basename(f)]

            # Eğer hard negative bulunamazsa, eski yönteme dön
            if not hard_negative_train:
                print("⚠️ İsimde similar_ bulunamadı, dosyaları yarıya bölüyorum...")
                hard_negative_train = negative_train[:len(negative_train)//2]
                random_negative_train = negative_train[len(negative_train)//2:]

            if not hard_negative_val:
                hard_negative_val = negative_val[:len(negative_val)//2]
                random_negative_val = negative_val[len(negative_val)//2:]

            # Debug için sayıları yazdır
            print(f"📊 Hard Negatives: Train={len(hard_negative_train)}, Val={len(hard_negative_val)}")
            print(f"📊 Random Negatives: Train={len(random_negative_train)}, Val={len(random_negative_val)}")

            # Create datasets
            train_dataset = EnhancedWakewordDataset(
                wakeword_train, hard_negative_train,
                random_negative_train, background_files,
                self.processor, augment=True,
                background_mix_prob=float(background_mix_prob),
                snr_range=(float(snr_min), float(snr_max))
            )

            val_dataset = EnhancedWakewordDataset(
                wakeword_val, hard_negative_val,
                random_negative_val, background_files[:50],
                self.processor, augment=False,
                background_mix_prob=float(background_mix_prob) * 0.5,
                snr_range=(max(float(snr_min), 5.0), min(float(snr_max), 15.0))
            )

            # Create dataloaders
            # On Windows, multi-processing workers can cause storage resize errors with numpy/librosa.
            # Use single-process loading there for stability.
            import os as _os
            _is_windows = (_os.name == 'nt')
            # Try a modest number of workers on Windows; fallback to 0 on failure
            _suggested_workers = max(2, (os.cpu_count() or 4) // 4)
            _num_workers = _suggested_workers if _is_windows else 2
            def _make_loader(dataset, shuffle):
                if _num_workers > 0:
                    try:
                        return DataLoader(
                            dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=_num_workers,
                            pin_memory=True,
                            persistent_workers=True,
                            prefetch_factor=2,
                        )
                    except Exception:
                        # Fallback to single-process on any Windows/librosa issues
                        return DataLoader(
                            dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=0,
                            pin_memory=True,
                        )
                else:
                    return DataLoader(
                        dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=0,
                        pin_memory=True,
                    )

            self.train_loader = _make_loader(train_dataset, shuffle=True)
            self.val_loader = _make_loader(val_dataset, shuffle=False)

            # Remember last data params for potential rebuilds
            self._last_data_params = (
                positive_dir, negative_dir, background_dir, int(batch_size), float(val_split), float(test_split),
                float(background_mix_prob), float(snr_min), float(snr_max)
            )

            data_info = f"""
✅ Veri Yükleme Başarılı!

📊 Veri İstatistikleri:
• Wakeword dosyaları: {len(wakeword_files)}
• Negative dosyaları: {len(negative_files)}
• Background dosyaları: {len(background_files)}

📈 Train/Val/Test Dağılımı:
• Train: {len(wakeword_train)} wakeword + {len(negative_train)} negative
• Validation: {len(wakeword_val)} wakeword + {len(negative_val)} negative
• Test: {len(wakeword_test)} wakeword + {len(negative_test)} negative

⚙️ Model Parametreleri: {sum(p.numel() for p in self.model.parameters()):,}
🚀 Cihaz: {self.device}
            """

            return data_info, len(train_dataset), len(val_dataset)

        except Exception as e:
            return f"Veri yükleme hatası: {str(e)}", None, None

    def start_training(self, epochs, lr, batch_size, dropout, hidden_size=None, num_layers=None):
        try:
            # Update trainer config
            self.trainer.config.LEARNING_RATE = lr
            self.trainer.config.BATCH_SIZE = batch_size
            self.trainer.config.EPOCHS = epochs
            self.model.dropout.p = dropout

            # If model architecture sliders are provided, rebuild model/trainer
            if hidden_size is not None and num_layers is not None:
                self.rebuild_model(int(hidden_size), int(num_layers), float(dropout))

            def training_thread():
                # Enable auto-extend: add up to +20 epochs in steps of 5 if improving
                self.trainer.train(
                    self.train_loader, self.val_loader, epochs, self.update_progress,
                    auto_extend=True, extend_step=5, max_extra_epochs=20,
                    plateau_delta=0.001, worsen_patience=3
                )

            thread = threading.Thread(target=training_thread)
            thread.daemon = True
            thread.start()

            return "Eğitim başlatıldı! İlerlemeyi grafiklerden takip edebilirsiniz."

        except Exception as e:
            return f"Eğitim başlatma hatası: {str(e)}"

    def update_progress(self, progress, current_batch, total_batches):
        # Called from training thread: store live progress for polling UI
        self._batch_progress = float(progress)
        self._current_batch = int(current_batch)
        self._total_batches = int(total_batches)

    def get_training_status(self):
        if not self.trainer.is_training and not self.trainer.training_complete:
            fig = go.Figure()
            fig.add_annotation(text="Henüz eğitim başlatılmadı", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=600, title_text="Training Progress")
            return "Eğitim başlatılmadı", fig, ""

        if self.trainer.is_training:
            if self._total_batches > 0:
                status = (
                    f"Eğitim devam ediyor - Epoch {self.trainer.current_epoch}/{self.trainer.config.EPOCHS} "
                    f"| Batch {self._current_batch}/{self._total_batches} (~{self._batch_progress:.1f}%)"
                )
            else:
                status = f"Eğitim devam ediyor - Epoch {self.trainer.current_epoch}/{self.trainer.config.EPOCHS}"
        else:
            status = f"Eğitim tamamlandı - En iyi validation accuracy: {self.trainer.best_val_acc:.2f}%"

        # Create live plots
        if len(self.trainer.train_losses) > 0:
            fig = make_subplots(rows=2, cols=2, subplot_titles=('Training Loss', 'Validation Loss', 'Training Accuracy', 'Validation Accuracy'))

            epochs = list(range(1, len(self.trainer.train_losses) + 1))

            fig.add_trace(go.Scatter(x=epochs, y=self.trainer.train_losses, name='Train Loss', line=dict(color='blue')), row=1, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=self.trainer.val_losses, name='Val Loss', line=dict(color='red')), row=1, col=2)
            fig.add_trace(go.Scatter(x=epochs, y=self.trainer.train_accuracies, name='Train Acc', line=dict(color='green')), row=2, col=1)
            fig.add_trace(go.Scatter(x=epochs, y=self.trainer.val_accuracies, name='Val Acc', line=dict(color='orange')), row=2, col=2)

            fig.update_layout(height=600, showlegend=True, title_text="Training Progress")
        else:
            fig = go.Figure()
            fig.add_annotation(text="Henüz eğitim verisi yok", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

        # Build current metrics text for quick glance
        if len(self.trainer.train_losses) > 0:
            i = -1
            self._last_metrics_text = (
                f"Epoch {self.trainer.current_epoch}/{self.trainer.config.EPOCHS}\n"
                f"Train Loss: {self.trainer.train_losses[i]:.4f}\n"
                f"Val Loss: {self.trainer.val_losses[i]:.4f}\n"
                f"Train Acc: {self.trainer.train_accuracies[i]:.2f}%\n"
                f"Val Acc: {self.trainer.val_accuracies[i]:.2f}%\n"
                f"Best Val Acc: {self.trainer.best_val_acc:.2f}%\n"
                f"{self.trainer._last_perf}"
            )
        return status, fig, self._last_metrics_text

    def stop_training(self):
        if self.trainer.is_training:
            self.trainer.is_training = False
            return "Eğitim durduruldu"
        return "Eğitim zaten duruyor"

    def save_model(self):
        if os.path.exists('best_wakeword_model.pth'):
            # Create deployment package
            deployment_package = {
                'model_state_dict': self.model.state_dict(),
                'model_config': {
                    'HIDDEN_SIZE': ModelConfig.HIDDEN_SIZE,
                    'NUM_LAYERS': ModelConfig.NUM_LAYERS,
                    'DROPOUT': ModelConfig.DROPOUT,
                    'NUM_CLASSES': ModelConfig.NUM_CLASSES
                },
                'audio_config': {
                    'SAMPLE_RATE': AudioConfig.SAMPLE_RATE,
                    'DURATION': AudioConfig.DURATION,
                    'N_MELS': AudioConfig.N_MELS,
                    'N_FFT': AudioConfig.N_FFT,
                    'HOP_LENGTH': AudioConfig.HOP_LENGTH,
                    'FMIN': AudioConfig.FMIN,
                    'FMAX': AudioConfig.FMAX
                },
                'training_info': {
                    'best_val_accuracy': self.trainer.best_val_acc,
                    'epochs_trained': len(self.trainer.train_losses),
                    'device': str(self.device)
                },
                'classes': ['negative', 'wakeword']
            }

            torch.save(deployment_package, 'wakeword_deployment_model.pth')

            info = f"""
✅ Model Kaydedildi!

📁 Kaydedilen Dosyalar:
• best_wakeword_model.pth (en iyi model)
• wakeword_deployment_model.pth (deployment paketi)

📊 Model Bilgileri:
• En iyi validation accuracy: {self.trainer.best_val_acc:.2f}%
• Eğitilen epoch sayısı: {len(self.trainer.train_losses)}
• Model parametreleri: {sum(p.numel() for p in self.model.parameters()):,}
            """
            return info
        else:
            return "❌ Kaydedilecek model bulunamadı. Önce eğitim yapın."

    def test_model(self, threshold=0.35):
        if not os.path.exists('best_wakeword_model.pth'):
            return "❌ Test edilecek model bulunamadı", None

        try:
            # Load best model
            checkpoint = torch.load('best_wakeword_model.pth', map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            # Test on validation set
            all_preds = []
            all_labels = []
            all_probs = []  # positive class probabilities

            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device).squeeze()
                    logits = self.model(data)
                    probs = torch.softmax(logits, dim=1)[:, 1]
                    predicted = (probs >= 0.3).long()  # 0.3 threshold kullan

                    all_preds.extend(predicted.detach().cpu().numpy().tolist())
                    all_labels.extend(target.detach().cpu().numpy().tolist())
                    all_probs.extend(probs.detach().cpu().numpy().tolist())

            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')
            f1 = f1_score(all_labels, all_preds, average='weighted')

            # Create confusion matrix
            cm = confusion_matrix(all_labels, all_preds)

            # Advanced metrics
            from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, brier_score_loss
            try:
                fpr, tpr, roc_th = roc_curve(all_labels, all_probs)
                roc_auc = auc(fpr, tpr)
            except Exception:
                fpr, tpr, roc_auc = [0, 1], [0, 1], float('nan')

            try:
                pr_prec, pr_rec, pr_th = precision_recall_curve(all_labels, all_probs)
                ap = average_precision_score(all_labels, all_probs)
            except Exception:
                pr_prec, pr_rec, ap = [1, 0], [0, 1], float('nan')

            try:
                brier = brier_score_loss(all_labels, all_probs)
            except Exception:
                brier = float('nan')

            # Best threshold by F1
            best_thr, best_f1 = 0.5, f1
            try:
                ths = pr_th if 'pr_th' in locals() and len(pr_th) > 0 else [0.5]
                from sklearn.metrics import f1_score as _f1
                for thr in ths:
                    preds_thr = [1 if p >= thr else 0 for p in all_probs]
                    f1_thr = _f1(all_labels, preds_thr)
                    if f1_thr > best_f1:
                        best_f1, best_thr = f1_thr, float(thr)
            except Exception:
                pass

            fig = make_subplots(rows=2, cols=2, subplot_titles=('Confusion Matrix', 'Metrics', 'ROC Curve', 'Precision-Recall'))

            # Confusion Matrix
            fig.add_trace(go.Heatmap(z=cm, colorscale='Blues',
                                     x=['Negative', 'Wakeword'],
                                     y=['Negative', 'Wakeword'],
                                     showscale=True), row=1, col=1)

            # Metrics
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [accuracy, precision, recall, f1]
            fig.add_trace(go.Bar(x=metrics, y=values, name='Metrics', marker_color='lightblue'), row=1, col=2)

            # ROC Curve
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name=f'ROC (AUC={roc_auc:.3f})', line=dict(color='red')), row=2, col=1)

            # Precision-Recall Curve
            fig.add_trace(go.Scatter(x=pr_rec, y=pr_prec, name=f'PR (AP={ap:.3f})', line=dict(color='green')), row=2, col=2)

            fig.update_layout(height=800, showlegend=True, title_text="Model Evaluation Results")

            result_text = f"""
📊 MODEL TEST SONUÇLARI
=======================

🎯 Performans Metrikleri:
• Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
• Precision: {precision:.4f}
• Recall: {recall:.4f}
• F1-Score: {f1:.4f}
• ROC AUC: {roc_auc:.4f}
• Average Precision (AP): {ap:.4f}
• Brier Score: {brier:.4f}
• En İyi Eşik (F1): {best_thr:.3f} (F1={best_f1:.4f})

📈 Confusion Matrix:
• True Negative: {cm[0][0]}
• False Positive: {cm[0][1]}
• False Negative: {cm[1][0]}
• True Positive: {cm[1][1]}

💡 Model Kalitesi:
{self.evaluate_model_quality(accuracy, precision, recall, f1)}
            """

            return result_text, fig

        except Exception as e:
            return f"Test hatası: {str(e)}", None

    def evaluate_model_quality(self, accuracy, precision, recall, f1):
        if f1 >= 0.9:
            return "• Mükemmel model! Üretim için hazır ✅"
        elif f1 >= 0.8:
            return "• Çok iyi model. İyi generalize ediyor ✅"
        elif f1 >= 0.7:
            return "• İyi model. Daha fazla veri ile geliştirilebilir ⚠️"
        elif f1 >= 0.6:
            return "• Orta seviye model. Veri kalitesi kontrol edilmeli ⚠️"
        else:
            return "• Zayıf model. Veri ve parametreler gözden geçirilmeli ❌"

# Create app instance
app = WakewordTrainingApp()

# Create Enhanced Gradio Interface
def create_enhanced_interface():
    with gr.Blocks(title="Enhanced Wakeword Training System", theme=gr.themes.Soft(), css="""
        .scrollable-container {
            max-height: 600px;
            overflow-y: auto;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
        }
        .guide-section {
            margin-bottom: 20px;
        }
        .guide-section h3 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }
    """) as demo:

        gr.Markdown("# 🎯 Enhanced Wakeword Detection Training System")
        gr.Markdown("### 📚 Comprehensive Training with Detailed Documentation")

        with gr.Tabs():

            # Tab 1: Configuration
            with gr.TabItem("⚙️ Configuration"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### � Audio Configuration")
                        sample_rate = gr.Dropdown(label="Sample Rate ⚠️", choices=["8000", "16000", "22050", "44100"], value="16000", info="Ses örnekleme hızı. İnsan sesi için 16000Hz optimum, daha yüksek değerler gereksiz hesaplama yükü yaratır")
                        duration = gr.Slider(label="Audio Duration (s) ⚠️", minimum=0.5, maximum=3.0, value=1.7, step=0.1, info="Ses dosyasının işlenecek süresi. Wakeword'ü tam kapsamalı ama çok uzun olmamalı")
                        n_mels = gr.Slider(label="Mel Bands ⚠️", minimum=40, maximum=128, value=80, step=8, info="Mel frekans bandı sayısı. Daha fazla band daha iyi frekans çözünürlüğü sağlar")
                        n_fft = gr.Slider(label="FFT Window Size ⚠️", minimum=512, maximum=4096, value=2048, step=256, info="FFT pencere boyutu. Frekans çözünürlüğü ile zaman çözünürlüğü arasındaki denge")
                        hop_length = gr.Slider(label="Hop Length ⚠️", minimum=128, maximum=1024, value=512, step=64, info="Pencereler arası adım sayısı. Küçük değerler daha iyi zaman çözünürlüğü")
                        win_length = gr.Slider(label="Window Length ⚠️", minimum=512, maximum=4096, value=2048, step=256, info="Spektrogram pencere uzunluğu. FFT boyutu ile aynı olmalı")
                        fmin = gr.Slider(label="Min Frequency (Hz) ⚠️", minimum=0, maximum=500, value=0, step=20, info="Minimum frekans. Genellikle 0, insan dışı sesler için artırılabilir")
                        fmax = gr.Slider(label="Max Frequency (Hz) ⚠️", minimum=4000, maximum=22050, value=8000, step=500, info="Maksimum frekans. İnsan sesi için 8000Hz yeterli")

                        gr.Markdown("### 🧠 Model Configuration")
                        hidden_size = gr.Slider(label="Hidden Size ⚠️", minimum=128, maximum=1024, value=256, step=64, info="LSTM gizli katman boyutu. Daha büyük değerler daha karmaşık paternleri öğrenebilir")
                        num_layers = gr.Slider(label="LSTM Layers ⚠️", minimum=1, maximum=4, value=2, step=1, info="LSTM katman sayısı. 2 katman genellikle optimum, fazla katman overfitting riski")
                        dropout = gr.Slider(label="Dropout ⚠️", minimum=0.0, maximum=0.8, value=0.6, step=0.1, info="Overfitting önleme oranı. Eğitim sırasında rastgele nöronları devre dışı bırakır")

                        gr.Markdown("### 🔧 Advanced Model Settings")
                        grad_clip_max_norm = gr.Slider(label="Gradient Clip Norm ⚠️", minimum=0.1, maximum=5.0, value=1.0, step=0.1, info="Gradient clipping maksimum norm değeri. Exploding gradient'leri önler")
                        weight_decay = gr.Slider(label="Weight Decay ⚠️", minimum=0.0, maximum=0.001, value=0.00001, step=0.000001, info="L2 regularization ağırlığı. Overfitting'i önlemek için kullanılır")
                        use_amp = gr.Checkbox(label="Use Mixed Precision ⚠️", value=True, info="Mixed precision training kullan. GPU belleği tasarrufu ve hız artışı sağlar")

                        gr.Markdown("### 💾 Cache Settings")
                        feature_cache_size = gr.Slider(label="Feature Cache Size ⚠️", minimum=100, maximum=2000, value=512, step=50, info="Özellik önbellek boyutu. Daha büyük değerler daha az disk IO sağlar")
                        audio_cache_size = gr.Slider(label="Audio Cache Size ⚠️", minimum=50, maximum=1000, value=512, step=50, info="Ses önbellek boyutu. İşlenmiş ses dosyalarını bellekte tutar")

                        gr.Markdown("### 🛡️ Training Safety")
                        patience = gr.Slider(label="Early Stopping Patience ⚠️", minimum=3, maximum=30, value=10, step=1, info="Early stopping sabrı. Bu kadar epoch iyileşme olmazsa durur")

                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 Training Configuration")
                        epochs = gr.Slider(label="Epochs ⚠️", minimum=10, maximum=200, value=100, step=10, info="Maksimum eğitim dönemi sayısı. Early stopping ile erken bitebilir")
                        learning_rate = gr.Dropdown(label="Learning Rate ⚠️", choices=["0.001", "0.0005", "0.0001", "0.00005"], value="0.0001", info="Öğrenme hızı. Çok yüksek overfitting, çok düşük yavaş öğrenme")
                        lr = gr.Number(label="Custom Learning Rate ⚠️", value=0.0001, precision=5, info="Özel öğrenme hızı değeri. Dropdown ile senkronize olur")

                        gr.Markdown("### � Data Configuration")
                        val_split = gr.Slider(label="Validation Split ⚠️", minimum=0.1, maximum=0.3, value=0.2, step=0.05, info="Validation için ayrılan veri oranı. Model performansını değerlendirmek için kullanılır")
                        test_split = gr.Slider(label="Test Split ⚠️", minimum=0.05, maximum=0.3, value=0.1, step=0.05, info="Test için ayrılan veri oranı. Final model değerlendirmesi için kullanılır")
                        batch_size = gr.Slider(label="Batch Size ⚠️", minimum=8, maximum=64, value=32, step=8, info="Eğitim batch boyutu. GPU belleğine sığacak kadar büyük olmalı")

                        gr.Markdown("### 📈 Data Augmentation")
                        aug_prob = gr.Slider(label="Augmentation Probability ⚠️", minimum=0.0, maximum=1.0, value=0.85, step=0.05, info="Veri artırma uygulanma olasılığı. Eğitim çeşitliliği için yüksek olmalı")
                        noise_factor = gr.Slider(label="Noise Factor ⚠️", minimum=0.0, maximum=0.5, value=0.15, step=0.05, info="Eklenen gürültü miktarı. Çok yüksek değerler sesi bozar")
                        time_shift = gr.Slider(label="Time Shift (s) ⚠️", minimum=0.0, maximum=0.5, value=0.3, step=0.05, info="Zaman kaydırma maksimum değeri. Farklı zamanlamalar için")
                        pitch_shift = gr.Slider(label="Pitch Shift (semitones) ⚠️", minimum=0.0, maximum=3.0, value=1.5, step=0.5, info="Perde değiştirme aralığı. İnsan sesi varyasyonları için")
                        speed_change_min = gr.Slider(label="Speed Change Min ⚠️", minimum=0.5, maximum=1.0, value=0.9, step=0.05, info="Minimum hız değiştirme oranı. Konuşma hızı varyasyonları için")
                        speed_change_max = gr.Slider(label="Speed Change Max ⚠️", minimum=1.0, maximum=2.0, value=1.1, step=0.05, info="Maksimum hız değiştirme oranı. Konuşma hızı varyasyonları için")

                        gr.Markdown("### 🎵 Background Mixing")
                        bg_mix_prob = gr.Slider(label="Background Mix Probability ⚠️", minimum=0.0, maximum=1.0, value=0.7, step=0.05, info="Arka plan sesi karıştırma olasılığı. Gerçek dünya koşulları için")
                        snr_min = gr.Slider(label="SNR Min (dB) ⚠️", minimum=-10, maximum=20, value=0, step=5, info="Minimum sinyal-gürültü oranı. Çok düşük değerler zorlu koşullar")
                        snr_max = gr.Slider(label="SNR Max (dB) ⚠️", minimum=0, maximum=30, value=20, step=5, info="Maksimum sinyal-gürültü oranı. Temiz ses koşulları için")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎵 Feature Processing")
                        delta = gr.Checkbox(label="Include Delta Features ⚠️", value=True, info="Delta özelliklerini dahil et. Zaman değişimlerini yakalar")
                        delta_delta = gr.Checkbox(label="Include Delta-Delta Features ⚠️", value=False, info="Delta-delta özelliklerini dahil et. İkinci dereceden değişimler")
                        mean_norm = gr.Checkbox(label="Mean Normalization ⚠️", value=True, info="Ortalama normalizasyonu uygula. Özellikleri merkezler")
                        var_norm = gr.Checkbox(label="Variance Normalization ⚠️", value=False, info="Varyans normalizasyonu uygula. Özellikleri ölçeklendirir")

                        gr.Markdown("### 🔄 Advanced Augmentation")
                        time_shift_enabled = gr.Checkbox(label="Enable Time Shifting ⚠️", value=True, info="Zaman kaydırma artırmayı etkinleştir. Farklı zamanlamalar simüle eder")
                        pitch_shift_enabled = gr.Checkbox(label="Enable Pitch Shifting ⚠️", value=True, info="Perde değiştirme artırmayı etkinleştir. Farklı ses tonları için")
                        speed_change_enabled = gr.Checkbox(label="Enable Speed Changing ⚠️", value=True, info="Hız değiştirme artırmayı etkinleştir. Konuşma hızı varyasyonları için")
                        noise_addition_enabled = gr.Checkbox(label="Enable Noise Addition ⚠️", value=True, info="Gürültü ekleme artırmayı etkinleştir. Gerçek dünya koşulları için")
                        rirs_augmentation = gr.Checkbox(label="Enable RIRS Augmentation ⚠️", value=False, info="Oda impulse response artırmayı etkinleştir. RIRS dataset gerekli")

                    with gr.Column(scale=1):
                        gr.Markdown("### 🏠 RIRS Settings")
                        rirs_snr_min = gr.Slider(label="RIRS SNR Min (dB) ⚠️", minimum=0, maximum=20, value=5, step=1, info="RIRS için minimum SNR. Oda akustiği simülasyonu için")
                        rirs_snr_max = gr.Slider(label="RIRS SNR Max (dB) ⚠️", minimum=5, maximum=30, value=20, step=1, info="RIRS için maksimum SNR. Oda akustiği simülasyonu için")
                        rirs_probability = gr.Slider(label="RIRS Probability ⚠️", minimum=0.0, maximum=1.0, value=0.3, step=0.05, info="RIRS uygulama olasılığı. RIRS dataset mevcut olmalı")
                        max_rir_length = gr.Slider(label="Max RIR Length (s) ⚠️", minimum=1.0, maximum=10.0, value=3.0, step=0.5, info="Maksimum oda impulse response uzunluğu")

                # Dataset paths ve load button en alta
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📁 Dataset Paths")
                        positive_dir = gr.Textbox(label="Positive Dataset Path", value="./positive_dataset", placeholder="Wakeword recordings")
                        negative_dir = gr.Textbox(label="Negative Dataset Path", value="./negative_dataset", placeholder="Negative samples")
                        background_dir = gr.Textbox(label="Background Noise Path", value="./background_noise", placeholder="Background noise files")

                        load_data_btn = gr.Button("📥 Load Data", variant="primary")
                        data_status = gr.Textbox(label="Data Status", interactive=False, lines=8)

            # Tab 2: Training
            with gr.TabItem("🚀 Training"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Training Progress")

                        with gr.Row():
                            start_btn = gr.Button("▶️ Start Training", variant="primary")
                            pause_btn = gr.Button("⏸️ Pause", variant="secondary")
                            resume_btn = gr.Button("▶️ Resume", variant="secondary")
                            cont_btn = gr.Button("🔁 Continue from Checkpoint", variant="secondary")
                            stop_btn = gr.Button("⏹️ Stop Training", variant="secondary")
                            save_btn = gr.Button("💾 Save Model", variant="secondary")

                        training_status = gr.Textbox(label="Training Status", interactive=False)

                        # Live plots
                        training_plots = gr.Plot(label="Training Progress")

                        # Auto-refresh for live updates
                        timer = gr.Timer(value=2.0)

                    with gr.Column(scale=1):
                        gr.Markdown("### 📋 System Info")
                        system_info = gr.Textbox(label="System Information", value=f"""
🖥️ Device: {app.device}
🧮 CPU Cores: {os.cpu_count()}
💾 GPU Available: {torch.cuda.is_available()}
                        """, interactive=False, lines=5)

                        gr.Markdown("### 📊 Current Metrics")
                        current_metrics = gr.Textbox(label="Current Metrics", interactive=False, lines=8)

                        gr.Markdown("### 💾 Model Info")
                        model_info = gr.Textbox(label="Model Information", value=f"""
📐 Parameters: {sum(p.numel() for p in app.model.parameters()):,}
🏗️ Architecture: CNN + LSTM
🔧 Dropout: {ModelConfig.DROPOUT}
📊 Hidden Size: {ModelConfig.HIDDEN_SIZE}
                        """, interactive=False, lines=5)

            # Tab 3: Evaluation
            with gr.TabItem("📈 Evaluation"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🧪 Model Testing")

                        # YENİ: Threshold slider ekle
                        test_threshold = gr.Slider(
                            label="Detection Threshold",
                            minimum=0.1,
                            maximum=0.9,
                            value=0.35,  # False negative azaltmak için düşük başla
                            step=0.05,
                            info="Lower = fewer missed detections (but more false alarms)"
                        )
                        test_btn = gr.Button("🔬 Run Model Test", variant="primary")
                        test_results = gr.Textbox(label="Test Results", interactive=False, lines=10)

                        gr.Markdown("### 📁 Model Files")
                        refresh_files_btn = gr.Button("🔄 Refresh Files")
                        model_files = gr.Textbox(label="Available Models", interactive=False, lines=3)

                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Evaluation Plots")
                        evaluation_plots = gr.Plot(label="Model Evaluation")

            # Tab 4: Enhanced Training Guide (SCROLLABLE)
            with gr.TabItem("📚 Enhanced Training Guide"):
                gr.Markdown("### 🎯 Comprehensive Wakeword Training Documentation")

                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎯 Quick Navigation")
                        quick_start = gr.Markdown("""
**📖 GUIDE SECTIONS:**

1. **Veri Seti Hazırlama**
   • Miktar ve kalite gereksinimleri
   • Çeşitlilik stratejileri
   • Optimum dağılım oranları

2. **Model Mimarisi**
   • CNN+LSTM detayları
   • Parametre optimizasyonu
   • Katman analizi

3. **Training Süreci**
   • Pipeline detayları
   • Hyperparameter tuning
   • Early stopping

4. **Performans**
   • Metrikler ve yorumlama
   • Sorun giderme
   • Best practices

5. **Deployment**
   • Model export
   • Production monitoring
   • A/B testing
                        """)

                        gr.Markdown("### ⚠️ Common Issues")
                        common_issues = gr.Markdown("""
**🔧 FREQUENT PROBLEMS:**

**GPU Memory Error:**
• Batch size'ı azaltın
• Mixed precision kullanın
• Gradient accumulation uygulayın

**Overfitting:**
• Dropout'u artırın
• Augmentation'u güçlendirin
• Early stopping kullanın

**Low Accuracy:**
• Veri kalitesini kontrol edin
• Model kapasitesini artırın
• Hyperparameter'ları ayarlayın

**Slow Training:**
• GPU kullanımını kontrol edin
• Data loading'i optimize edin
                        """)

                    with gr.Column(scale=2):
                        gr.Markdown("### 📖 Complete Training Guide")

                        # Use HTML component for scrolling
                        guide_html = f"""
                        <div class="scrollable-container">
                            <div class="guide-section">
                                {markdown.markdown(COMPREHENSIVE_TRAINING_GUIDE)}
                            </div>
                        </div>
                        """

                        guide_content = gr.HTML(guide_html)

        # Event handlers
        def load_data_handler(positive_dir, negative_dir, background_dir, batch_size, val_split, test_split,
                              sample_rate, duration, n_mels,
                              aug_prob, noise_factor, time_shift, pitch_shift,
                              bg_mix_prob, snr_min, snr_max,
                              hidden_size, num_layers, dropout,
                              grad_clip_max_norm, weight_decay, use_amp,
                              feature_cache_size, audio_cache_size,
                              patience,
                              n_fft, hop_length, win_length, fmin, fmax,
                              speed_change_min, speed_change_max,
                              delta, delta_delta, mean_norm, var_norm,
                              time_shift_enabled, pitch_shift_enabled, speed_change_enabled, noise_addition_enabled, rirs_augmentation,
                              rirs_snr_min, rirs_snr_max, rirs_probability, max_rir_length):
            # Apply audio/model/augmentation configs immediately
            app.apply_audio_config(int(sample_rate), float(duration), int(n_mels))
            # Update additional audio config
            AudioConfig.N_FFT = int(n_fft)
            AudioConfig.HOP_LENGTH = int(hop_length)
            AudioConfig.WIN_LENGTH = int(win_length)
            AudioConfig.FMIN = int(fmin)
            AudioConfig.FMAX = int(fmax)
            app.apply_augmentation_config(float(aug_prob), float(noise_factor), float(time_shift), float(pitch_shift))
            # Update speed change config
            AugmentationConfig.SPEED_CHANGE_MIN = float(speed_change_min)
            AugmentationConfig.SPEED_CHANGE_MAX = float(speed_change_max)
            app.rebuild_model(int(hidden_size), int(num_layers), float(dropout))
            # Update trainer safety knobs and advanced settings
            app.trainer.patience = int(patience)
            app.trainer.grad_clip_max_norm = float(grad_clip_max_norm)
            app.trainer.use_amp = bool(use_amp)
            # Update training config
            TrainingConfig.WEIGHT_DECAY = float(weight_decay)
            # Update cache settings
            app.processor._cache_size = int(audio_cache_size)
            # Note: feature_cache_size would need to be implemented in the feature extraction pipeline
            # Load data using background mixing and SNR range
            return app.load_data(
                positive_dir, negative_dir, background_dir,
                batch_size, val_split, test_split,
                background_mix_prob=bg_mix_prob, snr_min=snr_min, snr_max=snr_max
            )

        def start_training_handler(epochs, lr, batch_size, dropout,
                                   sample_rate, duration, n_mels,
                                   hidden_size, num_layers,
                                   grad_clip_max_norm, weight_decay, use_amp,
                                   patience,
                                   n_fft, hop_length, win_length, fmin, fmax,
                                   speed_change_min, speed_change_max):
            # Ensure latest audio/model and safety settings are applied
            app.apply_audio_config(int(sample_rate), float(duration), int(n_mels))
            AudioConfig.N_FFT = int(n_fft)
            AudioConfig.HOP_LENGTH = int(hop_length)
            AudioConfig.WIN_LENGTH = int(win_length)
            AudioConfig.FMIN = int(fmin)
            AudioConfig.FMAX = int(fmax)
            AugmentationConfig.SPEED_CHANGE_MIN = float(speed_change_min)
            AugmentationConfig.SPEED_CHANGE_MAX = float(speed_change_max)
            app.rebuild_model(int(hidden_size), int(num_layers), float(dropout))
            app.trainer.patience = int(patience)
            app.trainer.grad_clip_max_norm = float(grad_clip_max_norm)
            app.trainer.use_amp = bool(use_amp)
            TrainingConfig.WEIGHT_DECAY = float(weight_decay)
            return app.start_training(int(epochs), float(lr), int(batch_size), float(dropout), hidden_size=int(hidden_size), num_layers=int(num_layers))

        def update_training_plots():
            return app.get_training_status()

        def stop_training_handler():
            return app.stop_training()

        def save_model_handler():
            return app.save_model()

        def test_model_handler(test_threshold):
            return app.test_model(threshold=test_threshold)

        def refresh_files_handler():
            files = []
            if os.path.exists('best_wakeword_model.pth'):
                files.append("best_wakeword_model.pth")
            if os.path.exists('wakeword_deployment_model.pth'):
                files.append("wakeword_deployment_model.pth")
            return "Available model files:\n" + "\n".join(files) if files else "No model files found"

        # Connect events
        load_data_btn.click(
            load_data_handler,
            inputs=[
                positive_dir, negative_dir, background_dir, batch_size, val_split, test_split,
                sample_rate, duration, n_mels,
                aug_prob, noise_factor, time_shift, pitch_shift,
                bg_mix_prob, snr_min, snr_max,
                hidden_size, num_layers, dropout,
                grad_clip_max_norm, weight_decay, use_amp,
                feature_cache_size, audio_cache_size,
                patience,
                n_fft, hop_length, win_length, fmin, fmax,
                speed_change_min, speed_change_max,
                delta, delta_delta, mean_norm, var_norm,
                time_shift_enabled, pitch_shift_enabled, speed_change_enabled, noise_addition_enabled, rirs_augmentation,
                rirs_snr_min, rirs_snr_max, rirs_probability, max_rir_length
            ],
            outputs=[data_status]
        )

        start_btn.click(
            start_training_handler,
            inputs=[epochs, lr, batch_size, dropout, sample_rate, duration, n_mels, hidden_size, num_layers, grad_clip_max_norm, weight_decay, use_amp, patience, n_fft, hop_length, win_length, fmin, fmax, speed_change_min, speed_change_max],
            outputs=[training_status]
        )

        pause_btn.click(
            lambda: app.pause_training(),
            outputs=[training_status]
        )

        resume_btn.click(
            lambda: app.resume_training(),
            outputs=[training_status]
        )

        cont_btn.click(
            lambda: app.continue_from_checkpoint(),
            outputs=[training_status]
        )

        stop_btn.click(
            stop_training_handler,
            outputs=[training_status]
        )

        save_btn.click(
            save_model_handler,
            outputs=[training_status]
        )

        test_btn.click(
            test_model_handler,
            inputs=[test_threshold],
            outputs=[test_results, evaluation_plots]
        )

        refresh_files_btn.click(
            refresh_files_handler,
            outputs=[model_files]
        )

        # Auto-refresh training plots
        timer.tick(
            update_training_plots,
            outputs=[training_status, training_plots, current_metrics]
        )

        # Update learning rate when dropdown changes
        def _sync_lr(x):
            try:
                return float(x)
            except Exception:
                return 0.0001
        learning_rate.change(_sync_lr, inputs=learning_rate, outputs=lr)

    return demo

if __name__ == "__main__":
    demo = create_enhanced_interface()
    demo.launch(share=True, debug=True)
