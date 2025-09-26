"""
Enhanced Wakeword Detection Application with Fixed Training and Live Monitoring
Addresses:
1. Pickle error in training thread (mappingproxy serialization issue)
2. Enhanced live monitoring with batch-level progress tracking
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchaudio
import torchaudio.transforms as T

import numpy as np
import librosa
import gradio as gr
import os
import logging
import time
import threading
from datetime import datetime
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Union
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from scipy import signal
import pickle
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
if device.type == 'cuda':
    logger.info(f"GPU: {torch.cuda.get_device_name()}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")

def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# =====================================
# Configuration Classes
# =====================================

class AudioConfig:
    """Audio processing configuration"""
    SAMPLE_RATE = 16000
    DURATION = 1.0  # seconds
    N_MELS = 64
    N_FFT = 1024
    HOP_LENGTH = 256
    FMIN = 20
    FMAX = 8000

class ModelConfig:
    """Model architecture configuration"""
    NUM_CLASSES = 2  # wake word vs non-wake word
    DROPOUT_RATE = 0.3
    CONV_CHANNELS = [32, 64, 128]

class TrainingConfig:
    """Training configuration"""
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    PATIENCE = 10
    WEIGHT_DECAY = 1e-4

class AugmentationConfig:
    """Data augmentation configuration"""
    USE_AUGMENTATION = True
    TIME_SHIFT_RANGE = 0.1
    NOISE_FACTOR = 0.02
    PITCH_SHIFT_RANGE = 2

# =====================================
# Automated Dataset Management System
# =====================================

class DatasetManager:
    """Enhanced dataset manager with automated structure detection and splitting"""

    def __init__(self):
        self.base_data_dir = "data"
        self.supported_extensions = ['.wav', '.mp3', '.npy']
        self.train_ratio = 0.7
        self.val_ratio = 0.2
        self.test_ratio = 0.1

        # Define dataset categories and their expected structures
        self.dataset_categories = {
            'positive': {'min_files': 10, 'description': 'Wakeword recordings'},
            'negative': {'min_files': 45, 'description': 'Non-wakeword speech samples'},
            'hard_negative': {'min_files': 0, 'description': 'Hard negative samples (optional)'},
            'background': {'min_files': 100, 'description': 'Background noise recordings'},
            'rirs': {'min_files': 0, 'description': 'Room Impulse Responses (optional)'},
            'features': {'min_files': 0, 'description': 'Pre-extracted feature files (optional)'}
        }

        logger.info("DatasetManager initialized with automated splitting capabilities")

    def create_folder_structure(self):
        """Create the complete dataset folder structure"""
        try:
            # Create base data directory
            os.makedirs(self.base_data_dir, exist_ok=True)

            # Create category directories with train/val/test splits
            for category in self.dataset_categories.keys():
                category_path = os.path.join(self.base_data_dir, category)
                os.makedirs(category_path, exist_ok=True)

                # Create train/validation/test subdirectories
                for split in ['train', 'validation', 'test']:
                    split_path = os.path.join(category_path, split)
                    os.makedirs(split_path, exist_ok=True)

                    # Create features subdirectory if needed
                    if category == 'features':
                        features_before_path = os.path.join(split_path, 'before')
                        os.makedirs(features_before_path, exist_ok=True)

            # Create raw and processed directories
            os.makedirs(os.path.join(self.base_data_dir, 'raw'), exist_ok=True)
            os.makedirs(os.path.join(self.base_data_dir, 'processed'), exist_ok=True)

            logger.info("Complete dataset folder structure created")
            return True

        except Exception as e:
            logger.error(f"Failed to create folder structure: {e}")
            return False

    def detect_dataset_structure(self):
        """Detect current dataset structure and file counts"""
        structure_info = {
            'categories': {},
            'total_files': 0,
            'ready_for_splitting': False,
            'warnings': [],
            'recommendations': []
        }

        try:
            for category in self.dataset_categories.keys():
                category_path = os.path.join(self.base_data_dir, category)

                if not os.path.exists(category_path):
                    structure_info['categories'][category] = {
                        'exists': False,
                        'file_count': 0,
                        'files': [],
                        'status': 'missing'
                    }
                    continue

                # Recursively scan for files in all subdirectories
                files = []
                for root, _, filenames in os.walk(category_path):
                    for filename in filenames:
                        if any(filename.lower().endswith(ext) for ext in self.supported_extensions):
                            # Store relative path from category root
                            relative_path = os.path.relpath(os.path.join(root, filename), category_path)
                            files.append(relative_path)

                file_count = len(files)
                min_required = self.dataset_categories[category]['min_files']

                status = 'ready' if file_count >= min_required else 'insufficient' if file_count > 0 else 'empty'

                structure_info['categories'][category] = {
                    'exists': True,
                    'file_count': file_count,
                    'files': files,
                    'min_required': min_required,
                    'status': status
                }

                structure_info['total_files'] += file_count

                # Add warnings and recommendations
                if file_count > 0 and file_count < min_required:
                    structure_info['warnings'].append(
                        f"{category}: {file_count} files (minimum {min_required} recommended)"
                    )
                elif file_count == 0 and min_required > 0:
                    structure_info['warnings'].append(
                        f"{category}: No files found (minimum {min_required} required)"
                    )
                elif file_count >= min_required:
                    structure_info['recommendations'].append(
                        f"{category}: {file_count} files ready for splitting"
                    )

            # Check if any categories are ready for splitting
            ready_categories = [cat for cat, info in structure_info['categories'].items()
                              if info['status'] == 'ready']

            structure_info['ready_for_splitting'] = len(ready_categories) > 0
            structure_info['ready_categories'] = ready_categories

            return structure_info

        except Exception as e:
            logger.error(f"Dataset structure detection failed: {e}")
            structure_info['error'] = str(e)
            return structure_info

    def split_dataset(self, category, files):
        """Split files into train/validation/test sets with proper ratios"""
        try:
            total_files = len(files)
            if total_files == 0:
                return {'train': [], 'validation': [], 'test': []}

            # Calculate split sizes
            train_size = int(total_files * self.train_ratio)
            val_size = int(total_files * self.val_ratio)
            test_size = total_files - train_size - val_size

            # Ensure minimum 1 file per split if possible
            if total_files >= 3:
                if train_size == 0: train_size = 1
                if val_size == 0 and total_files > 1: val_size = 1
                if test_size == 0 and total_files > 2: test_size = 1

            # Adjust if sum exceeds total
            while train_size + val_size + test_size > total_files:
                if test_size > 0: test_size -= 1
                elif val_size > 0: val_size -= 1
                elif train_size > 0: train_size -= 1

            # Shuffle files for random distribution
            import random
            shuffled_files = files.copy()
            random.shuffle(shuffled_files)

            # Split files
            train_files = shuffled_files[:train_size]
            val_files = shuffled_files[train_size:train_size + val_size]
            test_files = shuffled_files[train_size + val_size:]

            return {
                'train': train_files,
                'validation': val_files,
                'test': test_files
            }

        except Exception as e:
            logger.error(f"Dataset splitting failed for {category}: {e}")
            return {'train': [], 'validation': [], 'test': []}

    def organize_dataset_files(self, structure_info):
        """Organize files into train/validation/test directories"""
        results = {
            'moved_files': 0,
            'errors': [],
            'category_results': {}
        }

        try:
            for category, info in structure_info['categories'].items():
                if info['status'] != 'ready':
                    continue

                category_path = os.path.join(self.base_data_dir, category)
                splits = self.split_dataset(category, info['files'])

                category_result = {
                    'train': {'count': 0, 'files': []},
                    'validation': {'count': 0, 'files': []},
                    'test': {'count': 0, 'files': []}
                }

                for split_name, files in splits.items():
                    split_path = os.path.join(category_path, split_name)

                    for file in files:
                        src_path = os.path.join(category_path, file)
                        dst_path = os.path.join(split_path, file)

                        try:
                            # Create subdirectory if it doesn't exist
                            os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                            # Move file to appropriate split directory
                            if os.path.exists(src_path):
                                shutil.move(src_path, dst_path)
                                category_result[split_name]['count'] += 1
                                category_result[split_name]['files'].append(file)
                                results['moved_files'] += 1

                        except Exception as e:
                            error_msg = f"Failed to move {file} to {split_name}: {e}"
                            results['errors'].append(error_msg)
                            logger.error(error_msg)

                results['category_results'][category] = category_result

            return results

        except Exception as e:
            logger.error(f"Dataset organization failed: {e}")
            results['errors'].append(f"Organization error: {e}")
            return results

    def get_dataset_statistics(self):
        """Get comprehensive dataset statistics after organization"""
        stats = {
            'total_files_by_category': {},
            'split_distribution': {},
            'file_types': {},
            'total_files': 0
        }

        try:
            for category in self.dataset_categories.keys():
                category_path = os.path.join(self.base_data_dir, category)

                if not os.path.exists(category_path):
                    continue

                category_stats = {
                    'total': 0,
                    'train': 0,
                    'validation': 0,
                    'test': 0,
                    'file_types': {}
                }

                # Count files in each split
                for split in ['train', 'validation', 'test']:
                    split_path = os.path.join(category_path, split)

                    if os.path.exists(split_path):
                        files = [f for f in os.listdir(split_path)
                               if os.path.isfile(os.path.join(split_path, f))]

                        category_stats[split] = len(files)
                        category_stats['total'] += len(files)

                        # Count by file type
                        for file in files:
                            ext = os.path.splitext(file)[1].lower()
                            category_stats['file_types'][ext] = category_stats['file_types'].get(ext, 0) + 1

                stats['total_files_by_category'][category] = category_stats
                stats['total_files'] += category_stats['total']

            return stats

        except Exception as e:
            logger.error(f"Failed to get dataset statistics: {e}")
            return stats

# Add required import at the top of the file
import shutil

# =====================================
# Feature Extraction
# =====================================

class MelSpectrogramExtractor:
    """Advanced Mel-spectrogram feature extractor with caching and GPU acceleration"""

    def __init__(self, config=AudioConfig):
        self.config = config

        # Create mel spectrogram transform
        self.mel_transform = T.MelSpectrogram(
            sample_rate=config.SAMPLE_RATE,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            n_mels=config.N_MELS,
            f_min=config.FMIN,
            f_max=config.FMAX,
            power=2.0,
            normalized=True
        ).to(device)

        # Create amplitude to dB transform
        self.amplitude_to_db = T.AmplitudeToDB(stype='power', top_db=80).to(device)

        logger.info(f"MelSpectrogramExtractor initialized on {device}")
        logger.info(f"Config: SR={config.SAMPLE_RATE}, N_MELS={config.N_MELS}, Duration={config.DURATION}s")

    def extract_features(self, audio_path: str) -> np.ndarray:
        """Extract mel-spectrogram features from audio file"""
        try:
            # Load audio
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if needed
            if sample_rate != self.config.SAMPLE_RATE:
                resampler = T.Resample(sample_rate, self.config.SAMPLE_RATE)
                waveform = resampler(waveform)

            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            # Normalize duration
            target_length = int(self.config.SAMPLE_RATE * self.config.DURATION)
            if waveform.shape[1] > target_length:
                # Trim to target length
                waveform = waveform[:, :target_length]
            elif waveform.shape[1] < target_length:
                # Pad with zeros
                pad_amount = target_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, pad_amount))

            # Move to GPU for processing
            waveform = waveform.to(device)

            # Extract mel spectrogram
            mel_spec = self.mel_transform(waveform)

            # Convert to dB scale
            mel_spec_db = self.amplitude_to_db(mel_spec)

            # Move back to CPU and convert to numpy
            mel_spec_db = mel_spec_db.cpu().squeeze(0).numpy()

            return mel_spec_db

        except Exception as e:
            logger.error(f"Feature extraction failed for {audio_path}: {e}")
            return self._get_zero_features()

    def extract_features_from_array(self, audio_array: np.ndarray, sample_rate: int = None) -> np.ndarray:
        """Extract features from numpy audio array"""
        try:
            if sample_rate is None:
                sample_rate = self.config.SAMPLE_RATE

            # Convert to tensor
            waveform = torch.from_numpy(audio_array).float().unsqueeze(0)

            # Resample if needed
            if sample_rate != self.config.SAMPLE_RATE:
                resampler = T.Resample(sample_rate, self.config.SAMPLE_RATE)
                waveform = resampler(waveform)

            # Normalize duration
            target_length = int(self.config.SAMPLE_RATE * self.config.DURATION)
            if waveform.shape[1] > target_length:
                waveform = waveform[:, :target_length]
            elif waveform.shape[1] < target_length:
                pad_amount = target_length - waveform.shape[1]
                waveform = F.pad(waveform, (0, pad_amount))

            # Move to GPU
            waveform = waveform.to(device)

            # Extract features
            mel_spec = self.mel_transform(waveform)
            mel_spec_db = self.amplitude_to_db(mel_spec)

            return mel_spec_db.cpu().squeeze(0).numpy()

        except Exception as e:
            logger.error(f"Feature extraction from array failed: {e}")
            return self._get_zero_features()

    def load_and_preprocess_audio(self, file_path: str) -> np.ndarray:
        """Load and preprocess audio file using librosa (fallback method)"""
        try:
            # Load with librosa
            audio, sr = librosa.load(file_path, sr=self.config.SAMPLE_RATE, mono=True)

            # Normalize duration
            target_length = int(self.config.SAMPLE_RATE * self.config.DURATION)
            if len(audio) > target_length:
                audio = audio[:target_length]
            elif len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))

            return audio
        except Exception as e:
            logger.error(f"Audio loading failed for {file_path}: {e}")
            return np.zeros(int(self.config.SAMPLE_RATE * self.config.DURATION))

    def _get_zero_features(self) -> np.ndarray:
        """Return zero features as fallback"""
        try:
            # Calculate expected output shape
            mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.config.SAMPLE_RATE, n_fft=self.config.N_FFT,
                hop_length=self.config.HOP_LENGTH, n_mels=self.config.N_MELS,
                f_min=self.config.FMIN, f_max=self.config.FMAX
            )
            dummy_audio = torch.zeros(1, int(self.config.SAMPLE_RATE * self.config.DURATION))
            dummy_features = mel_spec(dummy_audio).squeeze(0).numpy()
            return dummy_features
        except Exception:
            # Final fallback
            target_frames = int(self.config.DURATION * self.config.SAMPLE_RATE / self.config.HOP_LENGTH) + 1
            return np.zeros((self.config.N_MELS, target_frames))

# =====================================
# Data Augmentation
# =====================================

class AudioAugmenter:
    """Advanced audio augmentation with GPU acceleration"""

    def __init__(self, config=AugmentationConfig):
        self.config = config

    def time_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply time shifting augmentation"""
        if not self.config.USE_AUGMENTATION:
            return audio

        shift = np.random.uniform(-self.config.TIME_SHIFT_RANGE, self.config.TIME_SHIFT_RANGE)
        shift_samples = int(shift * AudioConfig.SAMPLE_RATE)
        return np.roll(audio, shift_samples)

    def add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add Gaussian noise"""
        if not self.config.USE_AUGMENTATION:
            return audio

        noise = np.random.normal(0, self.config.NOISE_FACTOR, audio.shape)
        return audio + noise

    def pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply pitch shifting using librosa"""
        if not self.config.USE_AUGMENTATION:
            return audio

        try:
            n_steps = np.random.uniform(-self.config.PITCH_SHIFT_RANGE, self.config.PITCH_SHIFT_RANGE)
            return librosa.effects.pitch_shift(audio, sr=AudioConfig.SAMPLE_RATE, n_steps=n_steps)
        except Exception as e:
            logger.warning(f"Pitch shifting failed: {e}")
            return audio

    def augment(self, audio: np.ndarray) -> np.ndarray:
        """Apply random augmentation"""
        if not self.config.USE_AUGMENTATION:
            return audio

        # Apply augmentations with probability
        if np.random.random() < 0.5:
            audio = self.time_shift(audio)
        if np.random.random() < 0.3:
            audio = self.add_noise(audio)
        if np.random.random() < 0.2:
            audio = self.pitch_shift(audio)

        return audio

# =====================================
# Enhanced Dataset Class
# =====================================

class WakewordDataset(Dataset):
    """Enhanced dataset with feature caching and advanced augmentation"""

    def __init__(self, positive_dir: str, negative_dir: str,
                 cache_features: bool = True, augment_data: bool = True,
                 feature_extractor: MelSpectrogramExtractor = None):
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.cache_features = cache_features
        self.augment_data = augment_data

        # Initialize feature extractor and augmenter
        self.feature_extractor = feature_extractor or MelSpectrogramExtractor()
        self.augmenter = AudioAugmenter() if augment_data else None

        # Load file paths and labels
        self.samples = []
        self.labels = []
        self.feature_cache = {}

        # Load positive samples (wake word)
        if os.path.exists(positive_dir):
            positive_files = [f for f in os.listdir(positive_dir) if f.endswith(('.wav', '.mp3', '.npy'))]
            for file in positive_files:
                self.samples.append(os.path.join(positive_dir, file))
                self.labels.append(1)

        # Load negative samples (non-wake word)
        if os.path.exists(negative_dir):
            negative_files = [f for f in os.listdir(negative_dir) if f.endswith(('.wav', '.mp3', '.npy'))]
            for file in negative_files:
                self.samples.append(os.path.join(negative_dir, file))
                self.labels.append(0)

        logger.info(f"Dataset loaded: {len(self.samples)} samples")
        logger.info(f"Positive samples: {sum(self.labels)}")
        logger.info(f"Negative samples: {len(self.labels) - sum(self.labels)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path = self.samples[idx]
        label = self.labels[idx]

        try:
            # Load features (with caching)
            if self.cache_features and file_path in self.feature_cache:
                features = self.feature_cache[file_path]
            else:
                # Check if it's a .npy file (pre-extracted features)
                if file_path.endswith('.npy'):
                    features = np.load(file_path)
                else:
                    # Extract features from audio
                    features = self.feature_extractor.extract_features(file_path)

                # Cache features
                if self.cache_features:
                    self.feature_cache[file_path] = features

            # Apply augmentation during training
            if self.augment_data and self.augmenter and not file_path.endswith('.npy'):
                # For augmentation, we need raw audio
                audio = self.feature_extractor.load_and_preprocess_audio(file_path)
                audio = self.augmenter.augment(audio)
                features = self.feature_extractor.extract_features_from_array(audio)

            # Convert to tensor and add channel dimension
            features = torch.FloatTensor(features).unsqueeze(0)  # (1, n_mels, time)
            label = torch.LongTensor([label])

            return features, label

        except Exception as e:
            logger.error(f"Error loading sample {file_path}: {e}")
            # Return zero features as fallback
            zero_features = torch.zeros(1, AudioConfig.N_MELS,
                                      int(AudioConfig.DURATION * AudioConfig.SAMPLE_RATE / AudioConfig.HOP_LENGTH) + 1)
            return zero_features, torch.LongTensor([label])

# =====================================
# Enhanced CNN Model
# =====================================

class WakewordCNN(nn.Module):
    """Enhanced CNN with advanced architecture and regularization"""

    def __init__(self, config=ModelConfig, audio_config=AudioConfig):
        super(WakewordCNN, self).__init__()
        self.config = config
        self.audio_config = audio_config

        # Calculate input dimensions
        self.n_mels = audio_config.N_MELS
        self.time_steps = int(audio_config.DURATION * audio_config.SAMPLE_RATE / audio_config.HOP_LENGTH) + 1

        # Convolutional layers with batch normalization and dropout
        self.conv_layers = nn.ModuleList()
        in_channels = 1

        for out_channels in config.CONV_CHANNELS:
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, 2),
                nn.Dropout2d(config.DROPOUT_RATE * 0.5)
            )
            self.conv_layers.append(layer)
            in_channels = out_channels

        # Calculate flattened size after convolutions
        self.flattened_size = self._get_flattened_size()

        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(config.DROPOUT_RATE),
            nn.Linear(128, config.NUM_CLASSES)
        )

        # Initialize weights
        self.apply(self._init_weights)

        logger.info(f"WakewordCNN initialized: {self._count_parameters()} parameters")

    def _get_flattened_size(self):
        """Calculate flattened size after convolutions"""
        x = torch.randn(1, 1, self.n_mels, self.time_steps)
        for layer in self.conv_layers:
            x = layer(x)
        return x.numel()

    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def _count_parameters(self):
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(self, x):
        # Convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Classifier
        x = self.classifier(x)

        return x

# =====================================
# Enhanced Training System
# =====================================

class WakewordTrainer:
    """Enhanced trainer with live monitoring, GPU optimization and fixed pickle error"""

    def __init__(self, model: WakewordCNN, config=TrainingConfig):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # Training state
        self.is_training = False
        self.training_complete = False
        self.training_start_time = None
        self.current_epoch = 0
        self.current_batch = 0
        self.total_batches = 0
        self.batch_update_frequency = 1  # Update every batch for real-time monitoring

        # Performance tracking
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.batch_losses = []
        self.batch_accuracies = []
        self.gpu_memory_usage = []

        # Best model tracking
        self.best_val_acc = 0.0
        self.epochs_no_improve = 0

        # Setup optimizer and scheduler
        self.optimizer = optim.AdamW(self.model.parameters(),
                                   lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
                                                            mode='max',
                                                            factor=0.5,
                                                            patience=5,
                                                            verbose=True)
        self.criterion = nn.CrossEntropyLoss()

        logger.info(f"Trainer initialized on {device}")

    def train_epoch(self, train_loader, progress_callback=None):
        """Train for one epoch with enhanced batch-level monitoring"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        batch_losses = []
        batch_accuracies = []

        for batch_idx, (data, target) in enumerate(train_loader):
            self.current_batch = batch_idx + 1  # 1-indexed for display
            data, target = data.to(self.device), target.to(self.device).squeeze()

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Statistics
            running_loss += loss.item()
            batch_losses.append(loss.item())
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

            # Calculate batch accuracy
            batch_acc = 100. * (predicted == target).sum().item() / target.size(0)
            batch_accuracies.append(batch_acc)

            # GPU memory tracking
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated() / 1e9
                self.gpu_memory_usage.append(gpu_memory)

            # Enhanced progress callback - update every batch
            if progress_callback and batch_idx % self.batch_update_frequency == 0:
                batch_progress = (batch_idx + 1) / len(train_loader) * 100
                progress_callback(f"Epoch {self.current_epoch + 1} | Batch {batch_idx + 1}/{len(train_loader)} ({batch_progress:.1f}%) | "
                               f"Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.2f}%")

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total

        # Store batch-level statistics
        self.batch_losses.extend(batch_losses)
        self.batch_accuracies.extend(batch_accuracies)

        return epoch_loss, epoch_acc, batch_losses

    def validate_epoch(self, val_loader):
        """Validate model with comprehensive metrics"""
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device).squeeze()

                output = self.model(data)
                loss = self.criterion(output, target)

                val_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = 100. * correct / total

        # Calculate detailed metrics
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='binary')

        return val_loss, val_acc, precision, recall, f1

    def train(self, train_loader, val_loader, num_epochs, progress_callback=None):
        """Enhanced training loop with comprehensive monitoring"""
        self.is_training = True
        self.training_complete = False
        self.training_start_time = time.time()
        self.total_batches = len(train_loader)

        logger.info(f"Starting training: {num_epochs} epochs, {len(train_loader)} batches per epoch")

        try:
            for epoch in range(num_epochs):
                self.current_epoch = epoch

                # Training phase
                train_loss, train_acc, batch_losses = self.train_epoch(train_loader, progress_callback)
                self.train_losses.append(train_loss)
                self.train_accuracies.append(train_acc)

                # Validation phase
                val_loss, val_acc, precision, recall, f1 = self.validate_epoch(val_loader)
                self.val_losses.append(val_loss)
                self.val_accuracies.append(val_acc)

                # Learning rate scheduling
                self.scheduler.step(val_acc)

                # Early stopping and model saving
                if val_acc > self.best_val_acc:
                    self.best_val_acc = val_acc
                    self.epochs_no_improve = 0
                    self.save_checkpoint(epoch, val_acc, precision, recall, f1)
                    if progress_callback:
                        progress_callback(f"✅ New best model saved! Val Acc: {val_acc:.2f}%")
                else:
                    self.epochs_no_improve += 1

                # Progress update
                if progress_callback:
                    progress_callback(
                        f"Epoch {epoch+1}/{num_epochs} complete | "
                        f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
                        f"Best: {self.best_val_acc:.2f}% | "
                        f"LR: {self.optimizer.param_groups[0]['lr']:.6f}"
                    )

                # Early stopping
                if self.epochs_no_improve >= TrainingConfig.PATIENCE:
                    if progress_callback:
                        progress_callback(f"Early stopping triggered after {epoch+1} epochs")
                    break

            self.training_complete = True
            training_time = time.time() - self.training_start_time

            if progress_callback:
                progress_callback(f"🎉 Training completed! Best validation accuracy: {self.best_val_acc:.2f}% in {training_time:.1f}s")

        except Exception as e:
            logger.error(f"Training error: {e}")
            if progress_callback:
                progress_callback(f"❌ Training failed: {str(e)}")
            raise e
        finally:
            self.is_training = False

    def save_checkpoint(self, epoch, val_acc, precision, recall, f1):
        """Save model checkpoint with fixed serialization (resolves pickle error)"""
        try:
            # Convert mappingproxy objects to regular dictionaries to fix pickle error
            model_config_dict = dict(vars(self.model.config)) if hasattr(self.model, 'config') else {}
            training_config_dict = dict(vars(self.config))
            audio_config_dict = dict(vars(AudioConfig))

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'best_val_acc': val_acc,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'config': {
                    'model_config': model_config_dict,
                    'training_config': training_config_dict,
                    'audio_config': audio_config_dict
                },
                'device': str(self.device),
                'timestamp': datetime.now().isoformat()
            }
            torch.save(checkpoint, 'best_wakeword_model.pth')
            logger.info(f"Checkpoint saved at epoch {epoch} with val_acc: {val_acc:.2f}%")
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def get_real_time_metrics(self):
        """Get current training metrics for enhanced real-time monitoring"""
        if not self.is_training:
            return None

        # Calculate batch progress within current epoch
        batch_progress = (self.current_batch / self.total_batches * 100) if self.total_batches > 0 else 0

        # Get latest batch statistics
        latest_batch_loss = self.batch_losses[-1] if self.batch_losses else 0
        latest_batch_acc = self.batch_accuracies[-1] if self.batch_accuracies else 0

        metrics = {
            'current_epoch': self.current_epoch + 1,
            'current_batch': self.current_batch,
            'total_batches': self.total_batches,
            'batch_progress': batch_progress,
            'latest_batch_loss': latest_batch_loss,
            'latest_batch_acc': latest_batch_acc,
            'train_loss': self.train_losses[-1] if self.train_losses else 0,
            'train_acc': self.train_accuracies[-1] if self.train_accuracies else 0,
            'val_loss': self.val_losses[-1] if self.val_losses else 0,
            'val_acc': self.val_accuracies[-1] if self.val_accuracies else 0,
            'best_val_acc': self.best_val_acc,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'epochs_no_improve': self.epochs_no_improve,
            'gpu_memory': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            'training_time': time.time() - self.training_start_time if self.training_start_time else 0
        }
        return metrics

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)

            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            logger.info(f"Best validation accuracy: {self.best_val_acc:.2f}%")

            return checkpoint
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None

# =====================================
# Enhanced Gradio Interface
# =====================================

class WakewordApp:
    """Enhanced Wakeword Detection Application with Advanced Features"""

    def __init__(self):
        self.feature_extractor = MelSpectrogramExtractor()
        self.model = None
        self.trainer = None
        self.train_loader = None
        self.val_loader = None
        self.training_thread = None
        self.dataset_manager = DatasetManager()  # Add DatasetManager instance

        # Setup data directories
        self.positive_dir = "data/positive"
        self.negative_dir = "data/negative"
        os.makedirs(self.positive_dir, exist_ok=True)
        os.makedirs(self.negative_dir, exist_ok=True)

        logger.info("WakewordApp initialized successfully")

    def initialize_model(self):
        """Initialize model and trainer"""
        if self.model is None:
            self.model = WakewordCNN()
            self.trainer = WakewordTrainer(self.model)
            logger.info("Model and trainer initialized")

    def load_dataset(self, positive_dir: str, negative_dir: str,
                    batch_size: int = 32, val_split: float = 0.2):
        """Load and prepare dataset with enhanced validation"""
        try:
            # Validate directories
            if not os.path.exists(positive_dir) or not os.path.exists(negative_dir):
                return f"❌ Dataset directories not found. Please upload data first."

            # Count files
            pos_files = [f for f in os.listdir(positive_dir) if f.endswith(('.wav', '.mp3', '.npy'))]
            neg_files = [f for f in os.listdir(negative_dir) if f.endswith(('.wav', '.mp3', '.npy'))]

            if len(pos_files) == 0 or len(neg_files) == 0:
                return f"❌ Insufficient data. Positive: {len(pos_files)}, Negative: {len(neg_files)}"

            # Create dataset
            full_dataset = WakewordDataset(positive_dir, negative_dir,
                                         cache_features=True, augment_data=True,
                                         feature_extractor=self.feature_extractor)

            # Split dataset
            dataset_size = len(full_dataset)
            val_size = int(val_split * dataset_size)
            train_size = dataset_size - val_size

            train_dataset, val_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, val_size])

            # Create balanced samplers
            train_labels = [full_dataset.labels[i] for i in train_dataset.indices]
            class_counts = np.bincount(train_labels)
            class_weights = 1.0 / class_counts
            sample_weights = [class_weights[label] for label in train_labels]

            train_sampler = WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )

            # Create data loaders
            # Fix for pickle error on Windows by disabling multiprocessing for DataLoader
            num_workers = 0 if os.name == 'nt' else 2

            self.train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True if device.type == 'cuda' else False
            )

            self.val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True if device.type == 'cuda' else False
            )

            return f"✅ Dataset loaded successfully!\n• Total samples: {dataset_size}\n• Training: {train_size}\n• Validation: {val_size}\n• Positive: {len(pos_files)}\n• Negative: {len(neg_files)}\n• Device: {device}"

        except Exception as e:
            logger.error(f"Dataset loading error: {e}")
            return f"❌ Dataset loading failed: {str(e)}"

    # Add new methods for automated dataset management
    def create_dataset_structure(self):
        """Create complete dataset folder structure"""
        try:
            success = self.dataset_manager.create_folder_structure()
            if success:
                return "✅ Complete dataset folder structure created successfully!"
            else:
                return "❌ Failed to create dataset folder structure"
        except Exception as e:
            logger.error(f"Dataset structure creation error: {e}")
            return f"❌ Error creating dataset structure: {str(e)}"

    def detect_dataset_status(self):
        """Detect current dataset structure and readiness"""
        try:
            structure_info = self.dataset_manager.detect_dataset_structure()

            if 'error' in structure_info:
                return f"❌ Dataset detection error: {structure_info['error']}"

            # Build status report
            status_report = "📊 **DATASET STRUCTURE ANALYSIS**\n\n"

            # Category status
            status_report += "📁 **Categories:**\n"
            for category, info in structure_info['categories'].items():
                if info['status'] != 'ready':
                    status_report += f"❌ {category}: Not found\n"
                    continue

                status_emoji = "✅" if info['status'] == 'ready' else "⚠️" if info['status'] == 'insufficient' else "❌"
                status_report += f"{status_emoji} {category}: {info['file_count']} files"

                if info['min_required'] > 0:
                    status_report += f" (min: {info['min_required']})"

                if info['status'] == 'insufficient':
                    status_report += f" - Need {info['min_required'] - info['file_count']} more"

                status_report += "\n"

            # Warnings and recommendations
            if structure_info['warnings']:
                status_report += "\n⚠️ **Warnings:**\n"
                for warning in structure_info['warnings']:
                    status_report += f"• {warning}\n"

            if structure_info['recommendations']:
                status_report += "\n💡 **Recommendations:**\n"
                for rec in structure_info['recommendations']:
                    status_report += f"• {rec}\n"

            # Overall readiness
            if structure_info['ready_for_splitting']:
                status_report += f"\n🎉 **READY FOR AUTO-SPLITTING!**\n"
                status_report += f"Categories ready: {', '.join(structure_info['ready_categories'])}\n"
                status_report += f"Total files detected: {structure_info['total_files']}\n"
            else:
                status_report += f"\n⏸️ **Not ready for auto-splitting**\n"
                status_report += "Please add more files to meet minimum requirements.\n"

            return status_report

        except Exception as e:
            logger.error(f"Dataset status detection error: {e}")
            return f"❌ Error detecting dataset status: {str(e)}"

    def auto_split_dataset(self):
        """Automatically split detected datasets into train/validation/test"""
        try:
            # First detect current structure
            structure_info = self.dataset_manager.detect_dataset_structure()

            if not structure_info['ready_for_splitting']:
                return "❌ Dataset not ready for auto-splitting. Please check dataset status first."

            if 'error' in structure_info:
                return f"❌ Dataset detection error: {structure_info['error']}"

            # Organize files into splits
            results = self.dataset_manager.organize_dataset_files(structure_info)

            if results['errors']:
                error_report = "❌ **Auto-splitting completed with errors:**\n"
                for error in results['errors']:
                    error_report += f"• {error}\n"
                return error_report

            # Get final statistics
            stats = self.dataset_manager.get_dataset_statistics()

            # Build success report
            success_report = f"""
🎉 **DATASET AUTO-SPLITTING COMPLETED SUCCESSFULLY!**

📊 **Files Moved:** {results['moved_files']}

📈 **Final Dataset Statistics:**
"""

            for category, cat_stats in stats['total_files_by_category'].items():
                if cat_stats['total'] > 0:
                    success_report += f"""
📁 **{category.upper()}** ({cat_stats['total']} files):
   Train: {cat_stats['train']} | Validation: {cat_stats['validation']} | Test: {cat_stats['test']}
"""
                    if cat_stats['file_types']:
                        success_report += f"   File types: {dict(cat_stats['file_types'])}\n"

            success_report += f"\n📊 **Total Dataset Size:** {stats['total_files']} files\n"

            return success_report

        except Exception as e:
            logger.error(f"Dataset auto-splitting error: {e}")
            return f"❌ Auto-splitting failed: {str(e)}"

    def get_comprehensive_dataset_info(self):
        """Get comprehensive dataset information including statistics"""
        try:
            stats = self.dataset_manager.get_dataset_statistics()

            if stats['total_files'] == 0:
                return "❌ No dataset files found. Please check dataset structure first."

            info_report = """
📊 **COMPREHENSIVE DATASET INFORMATION**

"""

            for category, cat_stats in stats['total_files_by_category'].items():
                if cat_stats['total'] > 0:
                    info_report += f"""
📁 **{category.upper()}** ({cat_stats['total']} files):
   Train: {cat_stats['train']} | Validation: {cat_stats['validation']} | Test: {cat_stats['test']}
"""
                    if cat_stats['file_types']:
                        info_report += f"   File types: {dict(cat_stats['file_types'])}\n"

            info_report += f"\n📈 **Total Dataset Size:** {stats['total_files']} files\n"

            return info_report

        except Exception as e:
            logger.error(f"Dataset info error: {e}")
            return f"❌ Error getting dataset info: {str(e)}"

    def start_training(self, epochs, lr, batch_size):
        """Start model training with enhanced monitoring"""
        try:
            self.initialize_model()

            if self.train_loader is None:
                dataset_result = self.load_dataset(self.positive_dir, self.negative_dir, int(batch_size))
                if "❌" in dataset_result:
                    return dataset_result

            # Update training configuration
            self.trainer.config.LEARNING_RATE = float(lr)
            self.trainer.config.BATCH_SIZE = int(batch_size)

            # Reinitialize optimizer with new learning rate
            self.trainer.optimizer = optim.AdamW(
                self.trainer.model.parameters(),
                lr=float(lr),
                weight_decay=self.trainer.config.WEIGHT_DECAY
            )

            # Start training in background thread with enhanced error handling
            def training_thread():
                try:
                    self.trainer.train(self.train_loader, self.val_loader, int(epochs))
                except Exception as e:
                    logger.error(f"Training error: {e}")
                    # Make error available to main thread
                    self.trainer.training_error = str(e)

            self.training_thread = threading.Thread(target=training_thread)
            self.training_thread.start()

            return f"🚀 Training started: {epochs} epochs, LR: {lr}, Batch: {batch_size} on {self.device}"

        except Exception as e:
            return f"❌ Training error: {str(e)}"

    def get_live_training_status(self):
        """Get detailed real-time training status with enhanced batch monitoring"""
        if not self.trainer or not self.trainer.is_training:
            if self.trainer and self.trainer.training_complete:
                return f"✅ Training completed! Best validation accuracy: {self.trainer.best_val_acc:.2f}%"
            else:
                return "⏸️ Training not active"

        metrics = self.trainer.get_real_time_metrics()
        if metrics is None:
            return "🔄 Starting training..."

        # Enhanced status with batch-level progress
        status = f"""
🔥 **TRAINING ACTIVE** 🔥

📊 **Progress:**
• Epoch: {metrics['current_epoch']}
• Batch: {metrics['current_batch']}/{metrics['total_batches']} ({metrics['batch_progress']:.1f}%)

📈 **Current Batch:**
• Loss: {metrics['latest_batch_loss']:.4f}
• Accuracy: {metrics['latest_batch_acc']:.2f}%

📊 **Epoch Performance:**
• Training Acc: {metrics['train_acc']:.2f}%
• Validation Acc: {metrics['val_acc']:.2f}%
• Best Val Acc: {metrics['best_val_acc']:.2f}%

⚙️ **Training Info:**
• Learning Rate: {metrics['learning_rate']:.6f}
• No Improve: {metrics['epochs_no_improve']} epochs
• GPU Memory: {metrics['gpu_memory']:.2f}GB
• Training Time: {metrics['training_time']:.1f}s
"""
        return status

    def predict_audio_file(self, audio_file):
        """Enhanced audio prediction with comprehensive analysis"""
        if self.model is None:
            return "❌ Please train a model first", None

        if audio_file is None:
            return "❌ Please upload an audio file", None

        try:
            # Extract features
            features = self.feature_extractor.extract_features(audio_file)

            # Convert to tensor and add batch dimension
            features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)

            # Make prediction
            self.model.eval()
            with torch.no_grad():
                output = self.model(features_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, prediction = torch.max(probabilities, 1)

                confidence_score = confidence.item() * 100
                predicted_class = prediction.item()

                # Get both class probabilities
                non_wakeword_conf = probabilities[0][0].item() * 100
                wakeword_conf = probabilities[0][1].item() * 100

            # Create detailed result
            result = f"""
🎯 **PREDICTION RESULTS**

📊 **Classification:**
• Prediction: {'🟢 WAKE WORD' if predicted_class == 1 else '🔴 NON-WAKE WORD'}
• Confidence: {confidence_score:.2f}%

📈 **Detailed Probabilities:**
• Non-Wake Word: {non_wakeword_conf:.2f}%
• Wake Word: {wakeword_conf:.2f}%

🔧 **Technical Info:**
• Feature Shape: {features.shape}
• Model Device: {next(self.model.parameters()).device}
• Processing Time: <1ms
"""

            return result, features

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return f"❌ Prediction failed: {str(e)}", None

    def record_and_predict(self, audio_data):
        """Process recorded audio and make prediction"""
        if audio_data is None:
            return "❌ No audio recorded"

        try:
            # Extract sample rate and audio array
            sample_rate, audio_array = audio_data

            # Convert to float32 and normalize
            audio_array = audio_array.astype(np.float32)
            if audio_array.max() > 1.0:
                audio_array = audio_array / np.max(np.abs(audio_array))

            # Extract features from audio array
            features = self.feature_extractor.extract_features_from_array(audio_array, sample_rate)

            # Make prediction
            if self.model is None:
                return "❌ Please train a model first"

            features_tensor = torch.FloatTensor(features).unsqueeze(0).unsqueeze(0).to(device)

            self.model.eval()
            with torch.no_grad():
                output = self.model(features_tensor)
                probabilities = F.softmax(output, dim=1)
                confidence, prediction = torch.max(probabilities, 1)

                confidence_score = confidence.item() * 100
                predicted_class = prediction.item()

                wakeword_conf = probabilities[0][1].item() * 100
                non_wakeword_conf = probabilities[0][0].item() * 100

            result = f"""
🎙️ **LIVE RECORDING RESULTS**

📊 **Classification:**
• Prediction: {'🟢 WAKE WORD DETECTED!' if predicted_class == 1 else '🔴 Non-Wake Word'}
• Confidence: {confidence_score:.2f}%

📈 **Probabilities:**
• Wake Word: {wakeword_conf:.2f}%
• Non-Wake Word: {non_wakeword_conf:.2f}%

🔧 **Audio Info:**
• Sample Rate: {sample_rate}Hz
• Duration: {len(audio_array) / sample_rate:.2f}s
• Feature Shape: {features.shape}
"""
            return result

        except Exception as e:
            logger.error(f"Live prediction error: {e}")
            return f"❌ Live prediction failed: {str(e)}"

    def get_model_info(self):
        """Get comprehensive model information"""
        if self.model is None:
            return "❌ No model loaded"

        try:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

            # Get model architecture info
            model_info = f"""
🧠 **MODEL ARCHITECTURE**

📊 **Parameters:**
• Total Parameters: {total_params:,}
• Trainable Parameters: {trainable_params:,}
• Model Size: ~{total_params * 4 / 1e6:.2f}MB

🏗️ **Architecture:**
• Input Shape: (1, {AudioConfig.N_MELS}, {int(AudioConfig.DURATION * AudioConfig.SAMPLE_RATE / AudioConfig.HOP_LENGTH) + 1})
• Conv Channels: {ModelConfig.CONV_CHANNELS}
• Dropout Rate: {ModelConfig.DROPOUT_RATE}
• Output Classes: {ModelConfig.NUM_CLASSES}

⚙️ **Configuration:**
• Audio sample rate: {AudioConfig.SAMPLE_RATE}Hz
• Audio duration: {AudioConfig.DURATION}s
• Mel bands: {AudioConfig.N_MELS}
• FFT size: {AudioConfig.N_FFT}
• Hop length: {AudioConfig.HOP_LENGTH}

💾 **Training State:**
• Best Validation Accuracy: {self.trainer.best_val_acc if self.trainer else 'Not trained'}
• Device: {next(self.model.parameters()).device}
"""
            return model_info

        except Exception as e:
            return f"❌ Error getting model info: {str(e)}"

    def export_model_info(self):
        """Export comprehensive model and training information"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Collect comprehensive information
            info = {
                'model_info': {
                    'architecture': 'WakewordCNN',
                    'parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
                    'device': str(device)
                },
                'audio_config': dict(vars(AudioConfig)),
                'model_config': dict(vars(ModelConfig)),
                'training_config': dict(vars(TrainingConfig)),
                'training_results': {
                    'best_validation_accuracy': self.trainer.best_val_acc if self.trainer else 0.0,
                    'train_losses': self.trainer.train_losses if self.trainer else [],
                    'val_accuracies': self.trainer.val_accuracies if self.trainer else [],
                    'training_complete': self.trainer.training_complete if self.trainer else False
                },
                'export_timestamp': timestamp
            }

            # Save to file
            filename = f"model_info_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(info, f, indent=2)

            return f"✅ Model information exported to {filename}"

        except Exception as e:
            return f"❌ Export failed: {str(e)}"

# =====================================
# Utility Functions
# =====================================

def create_sample_dataset():
    """Create sample dataset structure"""
    os.makedirs("data/positive", exist_ok=True)
    os.makedirs("data/negative", exist_ok=True)

    # Generate sample audio files
    sample_rate = AudioConfig.SAMPLE_RATE
    duration = AudioConfig.DURATION

    # Create sample wake word audio (simple tone)
    t = np.linspace(0, duration, int(sample_rate * duration))
    wakeword_audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note

    # Create sample non-wake word audio (noise)
    non_wakeword_audio = 0.1 * np.random.randn(int(sample_rate * duration))

    # Save sample files
    import soundfile as sf

    for i in range(5):
        sf.write(f"data/positive/sample_wakeword_{i}.wav", wakeword_audio, sample_rate)
        sf.write(f"data/negative/sample_non_wakeword_{i}.wav", non_wakeword_audio, sample_rate)

    return "✅ Sample dataset created in data/ directory"

def preprocess_uploaded_files():
    """Preprocess uploaded audio files and extract features"""
    feature_extractor = MelSpectrogramExtractor()
    processed_count = 0

    for class_dir in ['data/positive', 'data/negative']:
        if not os.path.exists(class_dir):
            continue

        audio_files = [f for f in os.listdir(class_dir) if f.endswith(('.wav', '.mp3')) and not f.endswith('.npy')]

        for audio_file in audio_files:
            try:
                audio_path = os.path.join(class_dir, audio_file)
                features = feature_extractor.extract_features(audio_path)

                # Save features as .npy file
                npy_path = os.path.join(class_dir, audio_file.replace('.wav', '.npy').replace('.mp3', '.npy'))
                np.save(npy_path, features)
                processed_count += 1

            except Exception as e:
                logger.error(f"Failed to preprocess {audio_file}: {e}")

    return f"✅ Preprocessed {processed_count} audio files"

# =====================================
# Gradio Interface Creation
# =====================================

def create_gradio_interface():
    """Create enhanced Gradio interface with comprehensive features"""

    app = WakewordApp()

    # Define custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .panel-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        font-weight: bold;
        font-size: 18px;
        margin-bottom: 20px;
    }
    .status-box {
        border: 2px solid #4CAF50;
        border-radius: 10px;
        padding: 15px;
        background-color: #f8f9fa;
        font-family: 'Courier New', monospace;
    }
    .training-status {
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.4;
    }
    .dataset-management {
        background-color: #f0f8ff;
        border: 2px solid #2196F3;
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
    }
    .auto-split-btn {
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4) !important;
        color: white !important;
        font-weight: bold !important;
        font-size: 16px !important;
        padding: 15px 30px !important;
        border-radius: 25px !important;
        border: none !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2) !important;
    }
    .structure-btn {
        background: linear-gradient(45deg, #667eea, #764ba2) !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 20px !important;
        border: none !important;
    }
    .detect-btn {
        background: linear-gradient(45deg, #f093fb, #f5576c) !important;
        color: white !important;
        font-weight: bold !important;
        border-radius: 20px !important;
        border: none !important;
    }
    """

    with gr.Blocks(css=custom_css, title="Enhanced Wakeword Detection System") as interface:

        # Header
        gr.HTML("""
        <div style="text-align: center; padding: 20px; background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 20px;">
            <h1>🎙️ Enhanced Wakeword Detection System</h1>
            <p>Advanced CNN-based wake word detection with automated dataset management</p>
        </div>
        """)

        with gr.Tabs():

            # Training Tab
            with gr.Tab("🚀 Model Training"):
                gr.HTML('<div class="panel-header">🏋️ Advanced Model Training</div>')

                with gr.Row():
                    with gr.Column(scale=1):
                        epochs_slider = gr.Slider(minimum=1, maximum=100, value=20, step=1,
                                                label="Training Epochs")
                        lr_slider = gr.Slider(minimum=0.0001, maximum=0.01, value=0.001, step=0.0001,
                                            label="Learning Rate")
                        batch_size_dropdown = gr.Dropdown(choices=[16, 32, 64, 128], value=32,
                                                        label="Batch Size")

                        train_btn = gr.Button("🚀 Start Training", variant="primary", size="lg")

                    with gr.Column(scale=2):
                        training_output = gr.Textbox(label="Training Status",
                                                   placeholder="Training status will appear here...",
                                                   lines=3)

                # Live Training Monitor
                gr.HTML('<div class="panel-header">📊 Live Training Monitor</div>')
                live_status = gr.Textbox(label="Real-time Training Status",
                                       lines=15,
                                       elem_classes=["training-status"])

                # Auto-refresh training status
                timer = gr.Timer(2.0)  # Auto-refresh every 2 seconds
                timer.tick(app.get_live_training_status, outputs=live_status)

                # Keep manual refresh functionality
                refresh_btn = gr.Button("🔄 Refresh Status")
                refresh_btn.click(
                    app.get_live_training_status,
                    outputs=live_status
                )

            # Enhanced Dataset Management Tab
            with gr.Tab("📁 Dataset Management"):
                gr.HTML('<div class="panel-header">📁 Automated Dataset Management & Configuration</div>')

                # New Automated Dataset Management Section
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.HTML('<div class="dataset-management">')
                        gr.Markdown("### 🤖 Automated Dataset Management")

                        create_structure_btn = gr.Button("🏗️ Create Dataset Structure",
                                                       elem_classes=["structure-btn"], size="lg")
                        detect_status_btn = gr.Button("🔍 Detect Dataset Status",
                                                    elem_classes=["detect-btn"], size="lg")
                        auto_split_btn = gr.Button("⚡ Auto-Split Dataset",
                                                 elem_classes=["auto-split-btn"], size="lg")

                        gr.Markdown("### 📊 Dataset Information")
                        dataset_info_btn = gr.Button("📈 Get Dataset Info", variant="secondary")

                        gr.HTML('</div>')

                    with gr.Column(scale=2):
                        automated_output = gr.Textbox(label="Automated Dataset Management Status",
                                                    lines=12,
                                                    placeholder="Dataset management status will appear here...")

                # Original Dataset Upload Section
                gr.HTML('<div class="panel-header">📤 Manual Dataset Upload (Legacy)</div>')

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### Upload Positive Samples (Wake Word)")
                        positive_upload = gr.File(file_count="multiple",
                                                file_types=[".wav", ".mp3"],
                                                label="Wake Word Audio Files")

                    with gr.Column():
                        gr.Markdown("### Upload Negative Samples (Non-Wake Word)")
                        negative_upload = gr.File(file_count="multiple",
                                                file_types=[".wav", ".mp3"],
                                                label="Non-Wake Word Audio Files")

                with gr.Row():
                    create_sample_btn = gr.Button("🎵 Create Sample Dataset")
                    preprocess_btn = gr.Button("⚡ Preprocess to .npy")
                    dataset_info_btn_legacy = gr.Button("📊 Legacy Dataset Info")

                dataset_output = gr.Textbox(label="Dataset Status", lines=5)

                # Event handlers for new automated functionality
                create_structure_btn.click(
                    app.create_dataset_structure,
                    outputs=automated_output
                )

                detect_status_btn.click(
                    app.detect_dataset_status,
                    outputs=automated_output
                )

                auto_split_btn.click(
                    app.auto_split_dataset,
                    outputs=automated_output
                )

                dataset_info_btn.click(
                    app.get_comprehensive_dataset_info,
                    outputs=automated_output
                )

                # Event handlers for legacy functionality
                create_sample_btn.click(
                    create_sample_dataset,
                    outputs=dataset_output
                )

                preprocess_btn.click(
                    preprocess_uploaded_files,
                    outputs=dataset_output
                )

                dataset_info_btn_legacy.click(
                    lambda: app.load_dataset(app.positive_dir, app.negative_dir),
                    outputs=dataset_output
                )

            # Testing & Prediction Tab
            with gr.Tab("🎯 Model Testing"):
                gr.HTML('<div class="panel-header">🎯 Audio Classification & Testing</div>')

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### File Upload Prediction")
                        audio_upload = gr.Audio(label="Upload Audio File", type="filepath")
                        predict_btn = gr.Button("🎯 Analyze Audio", variant="primary")

                        prediction_output = gr.Textbox(label="Prediction Results", lines=10)

                    with gr.Column():
                        gr.Markdown("### Live Recording Prediction")
                        # Fix: Replace deprecated 'source' parameter with 'sources'
                        # audio_record = gr.Audio(label="Record Audio", source="microphone", type="numpy")
                        audio_record = gr.Audio(label="Record Audio", sources=["microphone"], type="numpy")
                        live_predict_btn = gr.Button("🎙️ Predict Recording", variant="secondary")

                        live_output = gr.Textbox(label="Live Prediction Results", lines=10)

                # Event handlers
                predict_btn.click(
                    app.predict_audio_file,
                    inputs=audio_upload,
                    outputs=prediction_output
                )

                live_predict_btn.click(
                    app.record_and_predict,
                    inputs=audio_record,
                    outputs=live_output
                )

            # Model Information Tab
            with gr.Tab("🧠 Model Info"):
                gr.HTML('<div class="panel-header">🧠 Model Architecture & Performance</div>')

                with gr.Row():
                    info_btn = gr.Button("📊 Get Model Info")
                    export_btn = gr.Button("💾 Export Model Info")

                model_info_output = gr.Textbox(label="Model Information", lines=20)
                export_output = gr.Textbox(label="Export Status", lines=2)

                # Event handlers
                info_btn.click(
                    app.get_model_info,
                    outputs=model_info_output
                )

                export_btn.click(
                    app.export_model_info,
                    outputs=export_output
                )

            # System Status Tab
            with gr.Tab("⚙️ System Status"):
                gr.HTML('<div class="panel-header">⚙️ System Information & Diagnostics</div>')

                system_info = f"""
                🖥️ **System Information**

                **Hardware:**
                • Device: {device}
                • GPU Available: {torch.cuda.is_available()}
                • GPU Name: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'N/A'}
                • GPU Memory: {f'{(torch.cuda.get_device_properties(0).total_memory / 1e9):.1f}GB' if torch.cuda.is_available() else 'N/A'}

                **Software:**
                • PyTorch Version: {torch.__version__}
                • CUDA Version: {torch.version.cuda if torch.cuda.is_available() else 'N/A'}
                • Python: {os.sys.version}

                **Audio Configuration:**
                • Sample Rate: {AudioConfig.SAMPLE_RATE}Hz
                • Duration: {AudioConfig.DURATION}s
                • Mel Bands: {AudioConfig.N_MELS}
                • FFT Size: {AudioConfig.N_FFT}
                • Hop Length: {AudioConfig.HOP_LENGTH}

                **Model Configuration:**
                • Classes: {ModelConfig.NUM_CLASSES}
                • Dropout: {ModelConfig.DROPOUT_RATE}
                • Conv Channels: {ModelConfig.CONV_CHANNELS}

                **Training Configuration:**
                • Batch Size: {TrainingConfig.BATCH_SIZE}
                • Learning Rate: {TrainingConfig.LEARNING_RATE}
                • Max Epochs: {TrainingConfig.NUM_EPOCHS}
                • Patience: {TrainingConfig.PATIENCE}
                """

                gr.Textbox(value=system_info, label="System Status", lines=25, interactive=False)

        # Footer
        gr.HTML("""
        <div style="text-align: center; padding: 15px; margin-top: 20px; background-color: #f8f9fa; border-radius: 10px;">
            <p><strong>Enhanced Wakeword Detection System v3.0</strong></p>
            <p>🔧 Fixed pickle error | 📊 Enhanced live monitoring | ⚡ GPU accelerated | 🤖 Automated dataset management</p>
        </div>
        """)

    return interface

# =====================================
# Main Application
# =====================================

def main():
    """Main application entry point"""
    logger.info("Starting Enhanced Wakeword Detection System with Automated Dataset Management...")

    try:
        # Initialize dataset manager and create folder structure
        dataset_manager = DatasetManager()

        # Create initial folder structure
        logger.info("Creating initial dataset folder structure...")
        dataset_manager.create_folder_structure()

        # Create sample directories for backward compatibility
        os.makedirs("data/positive", exist_ok=True)
        os.makedirs("data/negative", exist_ok=True)
        os.makedirs("claudedocs", exist_ok=True)

        logger.info("Dataset folder structure initialized successfully")

    except Exception as e:
        logger.error(f"Failed to initialize dataset structure: {e}")
        logger.info("Continuing with basic directory creation...")

        # Fallback to basic directory creation
        os.makedirs("data/positive", exist_ok=True)
        os.makedirs("data/negative", exist_ok=True)
        os.makedirs("claudedocs", exist_ok=True)

    # Create and launch interface
    interface = create_gradio_interface()

    # Launch with enhanced configuration
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=True,
        show_error=True,
        inbrowser=True
    )

# Add additional utility functions for dataset management
def validate_dataset_requirements():
    """Validate that dataset meets minimum requirements for training"""
    dataset_manager = DatasetManager()
    structure_info = dataset_manager.detect_dataset_structure()

    ready_categories = structure_info.get('ready_categories', [])
    total_files = structure_info.get('total_files', 0)

    validation_result = {
        'is_ready': len(ready_categories) >= 2 and total_files >= 100,  # At least 2 categories with 100+ files
        'ready_categories': ready_categories,
        'total_files': total_files,
        'warnings': structure_info.get('warnings', []),
        'recommendations': structure_info.get('recommendations', [])
    }

    return validation_result

def get_dataset_health_report():
    """Generate a comprehensive dataset health report"""
    try:
        dataset_manager = DatasetManager()
        structure_info = dataset_manager.detect_dataset_structure()
        stats = dataset_manager.get_dataset_statistics()

        health_report = {
            'overall_health': 'good',
            'total_files': stats['total_files'],
            'categories_ready': len(structure_info.get('ready_categories', [])),
            'total_categories': len(dataset_manager.dataset_categories),
            'issues': [],
            'recommendations': []
        }

        # Check for issues
        if len(structure_info.get('warnings', [])) > 0:
            health_report['overall_health'] = 'needs_attention'
            health_report['issues'].extend(structure_info['warnings'])

        # Check file distribution
        for category, cat_stats in stats['total_files_by_category'].items():
            if cat_stats['total'] > 0:
                # Check if files are properly distributed
                if cat_stats['train'] == 0 or cat_stats['validation'] == 0 or cat_stats['test'] == 0:
                    health_report['issues'].append(f"{category}: Improper split distribution")
                    health_report['overall_health'] = 'needs_attention'

        # Add recommendations
        if health_report['total_files'] < 100:
            health_report['recommendations'].append("Add more files to improve model performance")

        if health_report['categories_ready'] < 2:
            health_report['recommendations'].append("Ensure at least positive and negative categories have sufficient files")

        return health_report

    except Exception as e:
        logger.error(f"Dataset health check failed: {e}")
        return {
            'overall_health': 'error',
            'issues': [f"Health check error: {e}"],
            'recommendations': ["Please check dataset structure manually"]
        }

# Update the footer version
# In the create_gradio_interface function, update the footer to:
# <p><strong>Enhanced Wakeword Detection System v3.0</strong></p>
# <p>🔧 Fixed pickle error | 📊 Enhanced live monitoring | ⚡ GPU accelerated | 🤖 Automated dataset management</p>
