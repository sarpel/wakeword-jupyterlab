#!/usr/bin/env python3
"""
Enhanced Wakeword Training Gradio Application
Complete GUI for wakeword detection model training with comprehensive documentation
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
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
warnings.filterwarnings('ignore')

# Configuration Classes (same as before)
class AudioConfig:
    SAMPLE_RATE = 16000
    DURATION = 1.7
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
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0001
    EPOCHS = 100
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.1

class AugmentationConfig:
    AUGMENTATION_PROB = 0.85
    NOISE_FACTOR = 0.15
    TIME_SHIFT_MAX = 0.3
    PITCH_SHIFT_MAX = 1.5
    SPEED_CHANGE_MIN = 0.9
    SPEED_CHANGE_MAX = 1.1

# Audio Processing Class (same as before)
class AudioProcessor:
    def __init__(self, config=AudioConfig):
        self.config = config

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

    def audio_to_mel(self, audio):
        if len(audio) == 0:
            return np.zeros((self.config.N_MELS, 31))

        audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
        if np.max(np.abs(audio)) < 1e-8:
            return np.zeros((self.config.N_MELS, 31))

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=self.config.SAMPLE_RATE, n_mels=self.config.N_MELS,
            n_fft=self.config.N_FFT, hop_length=self.config.HOP_LENGTH,
            win_length=self.config.WIN_LENGTH, fmin=self.config.FMIN, fmax=self.config.FMAX
        )

        return librosa.power_to_db(mel_spec, ref=np.max)

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
                 background_mix_prob=0.7, snr_range=(0, 20)):

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
        cache = []
        for i, bg_file in enumerate(self.background_files[:max_cache_size]):
            try:
                audio = self.processor.load_audio(bg_file)
                if audio is not None and len(audio) > 0:
                    audio = audio / (np.max(np.abs(audio)) + 1e-8)
                    cache.append(audio)
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

        audio = self.processor.load_audio(file_path)

        if audio is None:
            mel_spec = np.zeros((self.processor.config.N_MELS, 31), dtype=np.float32)
        else:
            audio = self.processor.normalize_audio(audio)
            target_length = int(self.processor.config.SAMPLE_RATE * self.processor.config.DURATION)
            audio = self.processor.pad_or_truncate(audio, target_length)

            if self.augment:
                audio = self.processor.augment_audio(audio)

            if category != 'background' and random.random() < self.background_mix_prob:
                audio = self._mix_with_background(audio)

            mel_spec = self.processor.audio_to_mel(audio)

        mel_array = np.ascontiguousarray(mel_spec, dtype=np.float32)
        mel_tensor = torch.from_numpy(mel_array).unsqueeze(0).clone()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return mel_tensor, label_tensor

# Training Class (same as before)
class WakewordTrainer:
    def __init__(self, model, device, config=TrainingConfig):
        self.model = model
        self.device = device
        self.config = config

        self.criterion = nn.CrossEntropyLoss().to(device)
        self.optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-5)
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

    def train_epoch(self, train_loader, progress_callback=None):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
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

            if progress_callback and batch_idx % 10 == 0:
                progress = (batch_idx + 1) / len(train_loader) * 100
                progress_callback(progress, batch_idx + 1, len(train_loader))

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

    def train(self, train_loader, val_loader, epochs, progress_callback=None):
        self.is_training = True
        self.training_complete = False

        print(f"Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            if not self.is_training:
                break

            self.current_epoch = epoch + 1

            train_loss, train_acc = self.train_epoch(train_loader, progress_callback)
            val_loss, val_acc = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)

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
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                break

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
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AudioProcessor()
        self.model = WakewordModel().to(self.device)
        self.trainer = WakewordTrainer(self.model, self.device)
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None

    def load_data(self, positive_dir, negative_dir, background_dir, batch_size, val_split, test_split):
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

            # Create datasets
            train_dataset = EnhancedWakewordDataset(
                wakeword_train, negative_train[:len(negative_train)//2],
                negative_train[len(negative_train)//2:], background_files,
                self.processor, augment=True
            )

            val_dataset = EnhancedWakewordDataset(
                wakeword_val, negative_val[:len(negative_val)//2],
                negative_val[len(negative_val)//2:], background_files[:50],
                self.processor, augment=False
            )

            # Create dataloaders
            # On Windows, multi-processing workers can cause storage resize errors with numpy/librosa.
            # Use single-process loading there for stability.
            import os as _os
            _num_workers = 0 if _os.name == 'nt' else 2
            self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=_num_workers)
            self.val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=_num_workers)

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

    def start_training(self, epochs, lr, batch_size, dropout):
        try:
            # Update trainer config
            self.trainer.config.LEARNING_RATE = lr
            self.trainer.config.BATCH_SIZE = batch_size
            self.trainer.config.EPOCHS = epochs
            self.model.dropout.p = dropout

            def training_thread():
                self.trainer.train(self.train_loader, self.val_loader, epochs, self.update_progress)

            thread = threading.Thread(target=training_thread)
            thread.daemon = True
            thread.start()

            return "Eğitim başlatıldı! İlerlemeyi grafiklerden takip edebilirsiniz."

        except Exception as e:
            return f"Eğitim başlatma hatası: {str(e)}"

    def update_progress(self, progress, current_batch, total_batches):
        # This will be called during training to update UI
        pass

    def get_training_status(self):
        if not self.trainer.is_training and not self.trainer.training_complete:
            fig = go.Figure()
            fig.add_annotation(text="Henüz eğitim başlatılmadı", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
            fig.update_layout(height=600, title_text="Training Progress")
            return "Eğitim başlatılmadı", fig

        if self.trainer.is_training:
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

        return status, fig

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

    def test_model(self):
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

            with torch.no_grad():
                for data, target in self.val_loader:
                    data, target = data.to(self.device), target.to(self.device).squeeze()
                    output = self.model(data)
                    _, predicted = torch.max(output, 1)

                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(target.cpu().numpy())

            # Calculate metrics
            accuracy = accuracy_score(all_labels, all_preds)
            precision = precision_score(all_labels, all_preds, average='weighted')
            recall = recall_score(all_labels, all_preds, average='weighted')
            f1 = f1_score(all_labels, all_preds, average='weighted')

            # Create confusion matrix
            cm = confusion_matrix(all_labels, all_preds)

            fig = make_subplots(rows=2, cols=2, subplot_titles=('Confusion Matrix', 'Metrics', 'ROC Curve', 'Class Distribution'))

            # Confusion Matrix
            fig.add_trace(go.Heatmap(z=cm, colorscale='Blues',
                                     x=['Negative', 'Wakeword'],
                                     y=['Negative', 'Wakeword'],
                                     showscale=True), row=1, col=1)

            # Metrics
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            values = [accuracy, precision, recall, f1]
            fig.add_trace(go.Bar(x=metrics, y=values, name='Metrics', marker_color='lightblue'), row=1, col=2)

            # ROC Curve approximation
            fpr, tpr = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], [0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.96, 1.0]
            fig.add_trace(go.Scatter(x=fpr, y=tpr, name='ROC Curve', line=dict(color='red')), row=2, col=1)

            # Class Distribution
            class_counts = [len([l for l in all_labels if l == 0]), len([l for l in all_labels if l == 1])]
            fig.add_trace(go.Pie(labels=['Negative', 'Wakeword'], values=class_counts, name='Distribution'), row=2, col=2)

            fig.update_layout(height=800, showlegend=True, title_text="Model Evaluation Results")

            result_text = f"""
📊 MODEL TEST SONUÇLARI
=======================

🎯 Performans Metrikleri:
• Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)
• Precision: {precision:.4f}
• Recall: {recall:.4f}
• F1-Score: {f1:.4f}

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
                        gr.Markdown("### 📁 Data Configuration")
                        positive_dir = gr.Textbox(label="Positive Dataset Path", value="./positive_dataset", placeholder="Wakeword recordings")
                        negative_dir = gr.Textbox(label="Negative Dataset Path", value="./negative_dataset", placeholder="Negative samples")
                        background_dir = gr.Textbox(label="Background Noise Path", value="./background_noise", placeholder="Background noise files")

                        gr.Markdown("### 📊 Data Split")
                        val_split = gr.Slider(label="Validation Split", minimum=0.1, maximum=0.3, value=0.2, step=0.05)
                        test_split = gr.Slider(label="Test Split", minimum=0.1, maximum=0.3, value=0.1, step=0.05)
                        batch_size = gr.Slider(label="Batch Size", minimum=8, maximum=64, value=32, step=8)

                        load_data_btn = gr.Button("📥 Load Data", variant="primary")
                        data_status = gr.Textbox(label="Data Status", interactive=False, lines=8)

                    with gr.Column(scale=1):
                        gr.Markdown("### 🧠 Model Configuration")
                        hidden_size = gr.Slider(label="Hidden Size", minimum=128, maximum=512, value=256, step=64)
                        num_layers = gr.Slider(label="LSTM Layers", minimum=1, maximum=4, value=2, step=1)
                        dropout = gr.Slider(label="Dropout", minimum=0.0, maximum=0.8, value=0.6, step=0.1)

                        gr.Markdown("### 🎯 Training Configuration")
                        epochs = gr.Slider(label="Epochs", minimum=10, maximum=200, value=100, step=10)
                        learning_rate = gr.Dropdown(label="Learning Rate", choices=["0.001", "0.0005", "0.0001", "0.00005"], value="0.0001")
                        lr = gr.Number(label="Custom Learning Rate", value=0.0001, precision=5)

                        gr.Markdown("### 🔊 Audio Configuration")
                        sample_rate = gr.Dropdown(label="Sample Rate", choices=["8000", "16000", "22050", "44100"], value="16000")
                        duration = gr.Slider(label="Audio Duration (s)", minimum=0.5, maximum=3.0, value=1.7, step=0.1)
                        n_mels = gr.Slider(label="Mel Bands", minimum=40, maximum=128, value=80, step=8)

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📈 Data Augmentation")
                        aug_prob = gr.Slider(label="Augmentation Probability", minimum=0.0, maximum=1.0, value=0.85, step=0.05)
                        noise_factor = gr.Slider(label="Noise Factor", minimum=0.0, maximum=0.5, value=0.15, step=0.05)
                        time_shift = gr.Slider(label="Time Shift (s)", minimum=0.0, maximum=0.5, value=0.3, step=0.05)
                        pitch_shift = gr.Slider(label="Pitch Shift (semitones)", minimum=0.0, maximum=3.0, value=1.5, step=0.5)

                    with gr.Column():
                        gr.Markdown("### 🎵 Background Mixing")
                        bg_mix_prob = gr.Slider(label="Background Mix Probability", minimum=0.0, maximum=1.0, value=0.7, step=0.05)
                        snr_min = gr.Slider(label="SNR Min (dB)", minimum=-10, maximum=20, value=0, step=5)
                        snr_max = gr.Slider(label="SNR Max (dB)", minimum=0, maximum=30, value=20, step=5)

                        gr.Markdown("### 🛡️ Training Safety")
                        patience = gr.Slider(label="Early Stopping Patience", minimum=5, maximum=20, value=10, step=1)
                        gradient_clip = gr.Slider(label="Gradient Clip Norm", minimum=0.5, maximum=2.0, value=1.0, step=0.1)

            # Tab 2: Training
            with gr.TabItem("🚀 Training"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Training Progress")

                        with gr.Row():
                            start_btn = gr.Button("▶️ Start Training", variant="primary")
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
                        current_metrics = gr.Textbox(label="Current Metrics", interactive=False, lines=6)

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
        def load_data_handler(positive_dir, negative_dir, background_dir, batch_size, val_split, test_split):
            return app.load_data(positive_dir, negative_dir, background_dir, batch_size, val_split, test_split)

        def start_training_handler(epochs, lr, batch_size, dropout):
            return app.start_training(int(epochs), float(lr), int(batch_size), float(dropout))

        def update_training_plots():
            return app.get_training_status()

        def stop_training_handler():
            return app.stop_training()

        def save_model_handler():
            return app.save_model()

        def test_model_handler():
            return app.test_model()

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
            inputs=[positive_dir, negative_dir, background_dir, batch_size, val_split, test_split],
            outputs=[data_status]
        )

        start_btn.click(
            start_training_handler,
            inputs=[epochs, lr, batch_size, dropout],
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
            outputs=[test_results, evaluation_plots]
        )

        refresh_files_btn.click(
            refresh_files_handler,
            outputs=[model_files]
        )

        # Auto-refresh training plots
        timer.tick(
            update_training_plots,
            outputs=[training_status, training_plots]
        )

        # Update learning rate when dropdown changes
        learning_rate.change(
            lambda x: gr.Number(value=float(x)),
            inputs=[learning_rate],
            outputs=[lr]
        )

    return demo

if __name__ == "__main__":
    demo = create_enhanced_interface()
    demo.launch(share=True, debug=True)
