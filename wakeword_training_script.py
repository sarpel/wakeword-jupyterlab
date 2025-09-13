#!/usr/bin/env python3
"""
Wakeword Training Script for Ubuntu WSL
Optimized for GPU training with CUDA support
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa
import soundfile as sf
import os
import glob
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import subprocess
import sys
warnings.filterwarnings('ignore')

# Configuration Classes
class AudioConfig:
    SAMPLE_RATE = 16000
    DURATION = 1.0
    N_MELS = 80
    N_FFT = 2048
    HOP_LENGTH = 512
    WIN_LENGTH = 2048
    FMIN = 0
    FMAX = 8000

class ModelConfig:
    HIDDEN_SIZE = 256
    NUM_LAYERS = 2
    DROPOUT = 0.6
    NUM_CLASSES = 2

class TrainingConfig:
    BATCH_SIZE = 16
    LEARNING_RATE = 0.0001
    EPOCHS = 10  # At least 10 epochs as requested
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1

class AugmentationConfig:
    AUGMENTATION_PROB = 0.8
    NOISE_FACTOR = 0.15
    TIME_SHIFT_MAX = 0.3
    PITCH_SHIFT_MAX = 3
    SPEED_CHANGE_MIN = 0.7
    SPEED_CHANGE_MAX = 1.3

# Audio Processing Class
class AudioProcessor:
    def __init__(self, config=AudioConfig):
        self.config = config

    def load_audio(self, file_path):
        try:
            audio, sr = librosa.load(file_path, sr=self.config.SAMPLE_RATE)
            return audio
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

    def normalize_audio(self, audio):
        if len(audio) == 0:
            return audio
        return audio / np.max(np.abs(audio))

    def pad_or_truncate(self, audio, target_length):
        if len(audio) > target_length:
            start_idx = random.randint(0, len(audio) - target_length)
            return audio[start_idx:start_idx + target_length]
        else:
            return np.pad(audio, (0, target_length - len(audio)), mode='constant')

    def audio_to_mel(self, audio):
        if len(audio) == 0:
            return np.zeros((self.config.N_MELS, int(self.config.SAMPLE_RATE * self.config.DURATION / self.config.HOP_LENGTH) + 1))

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.config.SAMPLE_RATE,
            n_mels=self.config.N_MELS,
            n_fft=self.config.N_FFT,
            hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH,
            fmin=self.config.FMIN,
            fmax=self.config.FMAX
        )

        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec

    def augment_audio(self, audio, config=AugmentationConfig):
        augmented_audio = audio.copy()

        if random.random() < config.AUGMENTATION_PROB:
            shift_amount = int(random.uniform(-config.TIME_SHIFT_MAX, config.TIME_SHIFT_MAX) * self.config.SAMPLE_RATE)
            augmented_audio = np.roll(augmented_audio, shift_amount)

        if random.random() < config.AUGMENTATION_PROB:
            n_steps = random.uniform(-config.PITCH_SHIFT_MAX, config.PITCH_SHIFT_MAX)
            augmented_audio = librosa.effects.pitch_shift(y=augmented_audio, sr=self.config.SAMPLE_RATE, n_steps=n_steps)

        if random.random() < config.AUGMENTATION_PROB:
            speed_factor = random.uniform(config.SPEED_CHANGE_MIN, config.SPEED_CHANGE_MAX)
            augmented_audio = librosa.effects.time_stretch(y=augmented_audio, rate=speed_factor)
            augmented_audio = self.pad_or_truncate(augmented_audio, len(audio))

        if random.random() < config.AUGMENTATION_PROB:
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

        mel_spec = self.audio_to_mel(audio)
        return mel_spec

# Neural Network Model
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
            input_size=self.cnn_output_size,
            hidden_size=config.HIDDEN_SIZE,
            num_layers=config.NUM_LAYERS,
            batch_first=True,
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

# Dataset Class
class WakewordDataset(Dataset):
    def __init__(self, wakeword_files, negative_files, processor, augment=False):
        self.wakeword_files = wakeword_files
        self.negative_files = negative_files
        self.processor = processor
        self.augment = augment

        self.files = wakeword_files + negative_files
        self.labels = [1] * len(wakeword_files) + [0] * len(negative_files)

        print(f"Dataset created with {len(self.files)} samples")
        print(f"Wakeword samples: {len(wakeword_files)}")
        print(f"Negative samples: {len(negative_files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        label = self.labels[idx]

        mel_spec = self.processor.process_audio_file(file_path, augment=self.augment)

        if mel_spec is None:
            mel_spec = np.zeros((self.processor.config.N_MELS, 31))

        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)
        label_tensor = torch.LongTensor([label])

        return mel_tensor, label_tensor

# Training Class
class WakewordTrainer:
    def __init__(self, model, device, config=TrainingConfig):
        self.model = model
        self.device = device
        self.config = config

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)

        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, patience=5
        )

        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []

        self.patience = 10
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device).squeeze()

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device).squeeze()
                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def train(self, train_loader, val_loader, epochs):
        print(f"Starting training for {epochs} epochs...")
        print(f"Using device: {self.device}")
        print(f"Learning rate: {self.config.LEARNING_RATE}")
        print(f"Batch size: {self.config.BATCH_SIZE}")

        self.best_val_acc = 0.0
        self.epochs_no_improve = 0

        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")

            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e6
                gpu_reserved = torch.cuda.memory_reserved() / 1e6
                print(f"GPU Memory: {gpu_memory:.1f}MB allocated, {gpu_reserved:.1f}MB reserved")
            else:
                print("GPU Memory: Not available (using CPU)")

            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            self.scheduler.step(val_acc)

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.epochs_no_improve = 0

                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, 'best_wakeword_model.pth')
                print(f"🎉 New best model saved! Validation accuracy: {val_acc:.2f}%")
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                print(f"\n⏹️  Early stopping triggered! No improvement for {self.patience} epochs.")
                print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
                break

        print(f"\n🎉 Training completed!")
        print(f"Best validation accuracy: {self.best_val_acc:.2f}%")
        print(f"Total epochs trained: {epoch + 1}")
        return self.best_val_acc

def create_sample_data():
    """Create sample audio data for training"""
    print("Creating sample data for training...")

    # Create directories
    os.makedirs('wakeword_data', exist_ok=True)
    os.makedirs('negative_data', exist_ok=True)
    os.makedirs('background_noise', exist_ok=True)

    # Generate sample wakeword data (simulated)
    for i in range(50):  # 50 wakeword samples
        duration = 1.0
        sr = 16000
        # Generate random audio with some structure
        audio = np.random.randn(int(duration * sr)) * 0.1
        # Add some periodic components to simulate speech
        t = np.linspace(0, duration, int(duration * sr))
        audio += np.sin(2 * np.pi * 200 * t) * 0.3  # Fundamental frequency
        audio += np.sin(2 * np.pi * 400 * t) * 0.2  # Harmonic

        sf.write(f'wakeword_data/wakeword_{i:03d}.wav', audio, sr)

    # Generate sample negative data
    for i in range(100):  # 100 negative samples
        duration = 1.0
        sr = 16000
        # Generate more random audio
        audio = np.random.randn(int(duration * sr)) * 0.2

        sf.write(f'negative_data/negative_{i:03d}.wav', audio, sr)

    # Generate background noise
    for i in range(20):  # 20 noise samples
        duration = 5.0
        sr = 16000
        # Generate noise
        audio = np.random.randn(int(duration * sr)) * 0.1

        sf.write(f'background_noise/noise_{i:03d}.wav', audio, sr)

    print("✅ Sample data created successfully!")
    print(f"   Wakeword samples: 50")
    print(f"   Negative samples: 100")
    print(f"   Background noise samples: 20")

def main():
    print("🎯 Wakeword Training System for Ubuntu WSL")
    print("=" * 50)

    # GPU Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔍 System Diagnostic:")
    print(f"   PyTorch version: {torch.__version__}")
    print(f"   Using device: {device}")
    print(f"   GPU Available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")
    else:
        print("   ⚠️  GPU not detected - running on CPU")

    # Create sample data if no data exists
    if not os.path.exists('wakeword_data') or len(os.listdir('wakeword_data')) == 0:
        create_sample_data()

    # Load audio files
    def load_audio_files(directory, extensions=['*.wav', '*.mp3', '*.flac']):
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, ext)))
        return files

    wakeword_files = load_audio_files('wakeword_data')
    negative_files = load_audio_files('negative_data')

    print(f"\n📊 Data Summary:")
    print(f"   Wakeword files: {len(wakeword_files)}")
    print(f"   Negative files: {len(negative_files)}")

    if len(wakeword_files) == 0 or len(negative_files) == 0:
        print("❌ No data files found!")
        return

    # Split data
    wakeword_train, wakeword_test = train_test_split(wakeword_files, test_size=TrainingConfig.TEST_SPLIT, random_state=42)
    wakeword_train, wakeword_val = train_test_split(wakeword_train, test_size=TrainingConfig.VALIDATION_SPLIT, random_state=42)

    negative_train, negative_test = train_test_split(negative_files, test_size=TrainingConfig.TEST_SPLIT, random_state=42)
    negative_train, negative_val = train_test_split(negative_train, test_size=TrainingConfig.VALIDATION_SPLIT, random_state=42)

    print(f"\n📈 Data Split:")
    print(f"   Training: {len(wakeword_train)} wakeword + {len(negative_train)} negative = {len(wakeword_train) + len(negative_train)} total")
    print(f"   Validation: {len(wakeword_val)} wakeword + {len(negative_val)} negative = {len(wakeword_val) + len(negative_val)} total")
    print(f"   Test: {len(wakeword_test)} wakeword + {len(negative_test)} negative = {len(wakeword_test) + len(negative_test)} total")

    # Create processor and model
    processor = AudioProcessor()
    model = WakewordModel().to(device)

    print(f"\n🧠 Model Created:")
    print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   Model on device: {next(model.parameters()).device}")

    # Create datasets
    train_dataset = WakewordDataset(wakeword_train, negative_train, processor, augment=True)
    val_dataset = WakewordDataset(wakeword_val, negative_val, processor, augment=False)
    test_dataset = WakewordDataset(wakeword_test, negative_test, processor, augment=False)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=TrainingConfig.BATCH_SIZE, shuffle=False, num_workers=2)

    # Create trainer and start training
    trainer = WakewordTrainer(model, device)

    print(f"\n🚀 Starting Training...")
    print(f"   Target epochs: {TrainingConfig.EPOCHS}")

    try:
        best_val_acc = trainer.train(train_loader, val_loader, TrainingConfig.EPOCHS)

        print(f"\n🎉 Training completed successfully!")
        print(f"   Best validation accuracy: {best_val_acc:.2f}%")
        print(f"   Target epochs reached: {TrainingConfig.EPOCHS}")

        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': {
                'model': ModelConfig.__dict__,
                'audio': AudioConfig.__dict__,
                'training': TrainingConfig.__dict__,
            },
            'best_val_acc': best_val_acc,
            'device': str(device)
        }, 'final_wakeword_model.pth')

        print("✅ Final model saved as 'final_wakeword_model.pth'")

    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()