#!/usr/bin/env python3
"""
Enhanced Wakeword Dataset with GPU-accelerated feature extraction and optimized processing
"""

import os
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
import json
from dataclasses import dataclass
import yaml
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing as mp
from datetime import datetime
import time

# Configure logging before any other imports that might use logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Use the new GPU-accelerated feature extractor
try:
    from feature_extractor import GPUFeatureExtractor as FeatureExtractor, RIRAugmentation
    logger.info("Using GPU-accelerated FeatureExtractor")
except ImportError as e:
    logger.warning(f"Failed to import GPUFeatureExtractor: {e}")
    from feature_extractor import FeatureExtractor, RIRAugmentation
    logger.info("Using standard FeatureExtractor")


@dataclass
class EnhancedAudioConfig:
    """Enhanced audio configuration with GPU acceleration options"""
    sample_rate: int = 16000
    duration: float = 2.0
    n_mels: int = 40
    n_fft: int = 1024
    hop_length: int = 160
    win_length: int = 400
    fmin: int = 20
    fmax: int = 8000

    # Feature extraction settings
    use_precomputed_features: bool = True
    features_dir: str = "features/"
    feature_cache_enabled: bool = True
    feature_config_path: str = "config/feature_config.yaml"
    use_gpu_acceleration: bool = True  # New GPU acceleration flag
    gpu_device: str = None  # Allow specific GPU device selection

    # RIRS augmentation settings
    use_rirs_augmentation: bool = False
    rirs_dataset_path: str = "datasets/mit_rirs/rir_data"
    rirs_snr_range: Tuple[float, float] = (5, 20)
    rirs_probability: float = 0.3

    # Traditional augmentation settings
    time_shift_amount: float = 0.1
    pitch_shift_range: Tuple[float, float] = (-2.0, 2.0)
    speed_change_range: Tuple[float, float] = (0.8, 1.2)
    noise_snr_range: Tuple[float, float] = (10, 30)
    augmentation_probability: float = 0.5

    # Performance settings
    num_workers: int = 4  # For parallel processing
    batch_size: int = 32  # For batch feature extraction
    prefetch_factor: int = 2  # For DataLoader optimization


class EnhancedWakewordDataset(Dataset):
    """Enhanced dataset with GPU-accelerated feature extraction and optimized processing"""

    def __init__(self,
                 positive_dir: str,
                 negative_dir: str,
                 features_dir: str = None,
                 rirs_dir: str = None,
                 config: EnhancedAudioConfig = None,
                 mode: str = "train"):
        """
        Initialize enhanced wakeword dataset with GPU acceleration

        Args:
            positive_dir: Directory containing positive audio samples
            negative_dir: Directory containing negative audio samples
            features_dir: Directory containing pre-computed .npy features
            rirs_dir: Directory containing RIRS dataset
            config: Audio configuration object
            mode: Dataset mode ('train', 'validation', 'test')
        """
        self.positive_dir = positive_dir
        self.negative_dir = negative_dir
        self.features_dir = features_dir
        self.rirs_dir = rirs_dir
        self.mode = mode
        self.config = config or EnhancedAudioConfig()

        # Initialize components with GPU acceleration
        self.feature_extractor = None
        self.rir_augmentation = None

        if self.config.use_precomputed_features:
            # Use GPU-accelerated feature extractor
            self.feature_extractor = FeatureExtractor(
                self.config.feature_config_path,
                device=self.config.gpu_device
            )

        if self.config.use_rirs_augmentation:
            self.rir_augmentation = RIRAugmentation(self.config.rirs_dataset_path)

        # Load data with performance tracking
        logger.info(f"Loading dataset for mode: {mode}")
        load_start_time = time.time()

        self.positive_files = self._load_audio_files(positive_dir)
        self.negative_files = self._load_audio_files(negative_dir)
        self.feature_files = self._load_feature_files() if features_dir else []

        load_time = time.time() - load_start_time

        # Dataset statistics
        self.positive_count = len(self.positive_files)
        self.negative_count = len(self.negative_files)
        self.feature_count = len(self.feature_files)

        # Balance dataset
        self._balance_dataset()

        # Create labels
        self._create_labels()

        logger.info(f"Dataset initialized in {load_time:.3f}s:")
        logger.info(f"  - Mode: {self.mode}")
        logger.info(f"  - Positive audio files: {self.positive_count}")
        logger.info(f"  - Negative audio files: {self.negative_count}")
        logger.info(f"  - Pre-computed features: {self.feature_count}")
        logger.info(f"  - Total samples: {len(self)}")
        logger.info(f"  - GPU acceleration: {'Enabled' if self.config.use_gpu_acceleration and torch.cuda.is_available() else 'Disabled'}")
        logger.info(f"  - RIRS augmentation: {'Enabled' if self.rir_augmentation else 'Disabled'}")
        logger.info(f"  - Feature caching: {'Enabled' if self.feature_extractor else 'Disabled'}")

    def _load_audio_files(self, directory: str) -> List[str]:
        """Load all audio files from directory with performance tracking"""
        audio_files = []
        if os.path.exists(directory):
            logger.info(f"Loading audio files from {directory}")
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.wav', '.flac', '.mp3', '.ogg', '.m4a')):
                        audio_files.append(os.path.join(root, file))
            logger.info(f"Found {len(audio_files)} audio files in {directory}")
        return audio_files

    def _load_feature_files(self) -> List[str]:
        """Load all .npy feature files with performance tracking"""
        feature_files = []
        if os.path.exists(self.features_dir):
            logger.info(f"Loading feature files from {self.features_dir}")
            for root, dirs, files in os.walk(self.features_dir):
                for file in files:
                    if file.endswith('.npy'):
                        feature_files.append(os.path.join(root, file))
            logger.info(f"Found {len(feature_files)} feature files")
        return feature_files

    def _balance_dataset(self):
        """Balance positive and negative samples with logging"""
        max_samples = max(self.positive_count, self.negative_count)
        logger.info(f"Balancing dataset to {max_samples} samples per class")

        # Duplicate positive samples if needed
        if self.positive_count < max_samples:
            extra_needed = max_samples - self.positive_count
            logger.info(f"Duplicating {extra_needed} positive samples")
            self.positive_files.extend(self.positive_files[:extra_needed])

        # Duplicate negative samples if needed
        if self.negative_count < max_samples:
            extra_needed = max_samples - self.negative_count
            logger.info(f"Duplicating {extra_needed} negative samples")
            self.negative_files.extend(self.negative_files[:extra_needed])

        # Update counts
        self.positive_count = len(self.positive_files)
        self.negative_count = len(self.negative_files)

        logger.info(f"Dataset balanced: {self.positive_count} positive, {self.negative_count} negative")

    def _create_labels(self):
        """Create labels for all samples with validation"""
        self.labels = []
        self.sources = []

        # Add feature file labels
        for feature_path in self.feature_files:
            if 'positive' in feature_path.lower():
                self.labels.append(1)
            else:
                self.labels.append(0)
            self.sources.append('feature')

        # Add audio file labels
        self.labels.extend([1] * self.positive_count)
        self.sources.extend(['audio'] * self.positive_count)

        self.labels.extend([0] * self.negative_count)
        self.sources.extend(['audio'] * self.negative_count)

        logger.info(f"Created labels for {len(self.labels)} samples")

    def __len__(self) -> int:
        """Get dataset size"""
        return self.feature_count + self.positive_count + self.negative_count

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item with GPU-accelerated processing"""
        start_time = time.time()

        label = self.labels[idx]
        source = self.sources[idx]

        try:
            if source == 'feature':
                # Load pre-computed features
                feature_path = self.feature_files[idx]
                features = self._load_features(feature_path)
                result = {
                    'features': torch.FloatTensor(features),
                    'label': torch.LongTensor([label]),
                    'source': 'feature',
                    'path': feature_path
                }
            else:
                # Load and process audio with GPU acceleration
                if idx < self.feature_count + self.positive_count:
                    audio_idx = idx - self.feature_count
                    audio_path = self.positive_files[audio_idx]
                else:
                    audio_idx = idx - self.feature_count - self.positive_count
                    audio_path = self.negative_files[audio_idx]

                result = self._load_and_process_audio(audio_path, label)

            # Log performance for slow operations
            load_time = time.time() - start_time
            if load_time > 0.1:  # Log slow operations
                logger.debug(f"Slow item load: idx={idx}, time={load_time:.3f}s, source={source}")

            return result

        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}")
            return self._get_default_item(label)

    def _load_features(self, feature_path: str) -> np.ndarray:
        """Load pre-computed .npy features with performance tracking"""
        try:
            start_time = time.time()
            features = np.load(feature_path)
            load_time = time.time() - start_time

            if load_time > 0.05:  # Log slow loads
                logger.debug(f"Slow feature load: {feature_path} took {load_time:.3f}s")

            # Apply RIRS augmentation to features if enabled
            if self.rir_augmentation and np.random.rand() < self.config.rirs_probability:
                # Note: RIRS augmentation typically requires audio domain
                # For feature domain, we'll apply alternative augmentation
                features = self._apply_feature_augmentation(features)

            return features

        except Exception as e:
            logger.error(f"Error loading features from {feature_path}: {e}")
            return self._get_default_features()

    def _load_and_process_audio(self, audio_path: str, label: int) -> Dict[str, torch.Tensor]:
        """Load audio file and process it with GPU acceleration"""
        try:
            # Use GPU-accelerated feature extraction if available
            if self.feature_extractor:
                features = self.feature_extractor.extract_features(audio_path)
                return {
                    'features': torch.FloatTensor(features),
                    'label': torch.LongTensor([label]),
                    'source': 'audio',
                    'path': audio_path
                }
            else:
                # Fallback to CPU processing
                return self._load_and_process_audio_cpu(audio_path, label)

        except Exception as e:
            logger.error(f"Error processing audio {audio_path}: {e}")
            return self._get_default_item(label)

    def _load_and_process_audio_cpu(self, audio_path: str, label: int) -> Dict[str, torch.Tensor]:
        """Fallback CPU-based audio processing"""
        try:
            # Load audio
            audio, sr = librosa.load(
                audio_path,
                sr=self.config.sample_rate,
                duration=self.config.duration
            )

            # Pad or trim to exact duration
            target_length = int(self.config.sample_rate * self.config.duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]

            # Apply augmentations
            if self.mode == 'train' and np.random.rand() < self.config.augmentation_probability:
                audio = self._apply_augmentations(audio)

            # Extract features using CPU
            features = self._extract_mel_spectrogram(audio)

            return {
                'features': torch.FloatTensor(features),
                'label': torch.LongTensor([label]),
                'source': 'audio',
                'path': audio_path
            }

        except Exception as e:
            logger.error(f"Error processing audio {audio_path} (CPU fallback): {e}")
            return self._get_default_item(label)

    def _apply_augmentations(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio augmentations with logging"""
        augmented = audio.copy()
        augmentation_applied = []

        # Time shift
        if np.random.rand() < 0.5:
            augmented = self._time_shift(augmented)
            augmentation_applied.append('time_shift')

        # RIRS augmentation
        if self.rir_augmentation and np.random.rand() < self.config.rirs_probability:
            augmented = self.rir_augmentation.apply_rir(
                augmented,
                sr=self.config.sample_rate,
                snr_range=self.config.rirs_snr_range
            )
            augmentation_applied.append('rirs')

        # Pitch shift
        if np.random.rand() < 0.3:
            augmented = self._pitch_shift(augmented)
            augmentation_applied.append('pitch_shift')

        # Speed change
        if np.random.rand() < 0.3:
            augmented = self._speed_change(augmented)
            augmentation_applied.append('speed_change')

        # Noise addition
        if np.random.rand() < 0.5:
            augmented = self._add_noise(augmented)
            augmentation_applied.append('noise')

        if augmentation_applied:
            logger.debug(f"Applied augmentations: {augmentation_applied}")

        return augmented

    def _apply_feature_augmentation(self, features: np.ndarray) -> np.ndarray:
        """Apply augmentations to pre-computed features with logging"""
        augmented = features.copy()
        augmentation_applied = []

        # Time shift in feature domain
        if np.random.rand() < 0.5:
            shift = int(np.random.uniform(-10, 10))
            augmented = np.roll(augmented, shift, axis=1)
            augmentation_applied.append('time_shift')

        # Frequency masking
        if np.random.rand() < 0.3:
            fmask_param = int(features.shape[0] * 0.1)
            f0 = np.random.randint(0, features.shape[0] - fmask_param)
            augmented[f0:f0 + fmask_param, :] = 0
            augmentation_applied.append('freq_mask')

        # Time masking
        if np.random.rand() < 0.3:
            tmask_param = int(features.shape[1] * 0.1)
            t0 = np.random.randint(0, features.shape[1] - tmask_param)
            augmented[:, t0:t0 + tmask_param] = 0
            augmentation_applied.append('time_mask')

        # Add noise
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.1, features.shape)
            augmented = augmented + noise
            augmentation_applied.append('noise')

        if augmentation_applied:
            logger.debug(f"Applied feature augmentations: {augmentation_applied}")

        return augmented

    def _time_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply time shift augmentation"""
        shift = int(np.random.uniform(-self.config.time_shift_amount, self.config.time_shift_amount) * self.config.sample_rate)
        return np.roll(audio, shift)

    def _pitch_shift(self, audio: np.ndarray) -> np.ndarray:
        """Apply pitch shift augmentation"""
        n_steps = np.random.uniform(*self.config.pitch_shift_range)
        return librosa.effects.pitch_shift(audio, sr=self.config.sample_rate, n_steps=n_steps)

    def _speed_change(self, audio: np.ndarray) -> np.ndarray:
        """Apply speed change augmentation"""
        speed = np.random.uniform(*self.config.speed_change_range)
        return librosa.effects.time_stretch(audio, rate=speed)

    def _add_noise(self, audio: np.ndarray) -> np.ndarray:
        """Add background noise"""
        noise_snr = np.random.uniform(*self.config.noise_snr_range)
        noise_power = np.mean(audio ** 2) / (10 ** (noise_snr / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise

    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel-spectrogram from audio with GPU fallback"""
        try:
            # Try GPU acceleration first
            if torch.cuda.is_available() and self.config.use_gpu_acceleration:
                audio_tensor = torch.from_numpy(audio).float().cuda()
                mel_spec = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.config.sample_rate,
                    n_mels=self.config.n_mels,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    win_length=self.config.win_length,
                    f_min=self.config.fmin,
                    f_max=self.config.fmax,
                    power=2.0
                ).cuda()(audio_tensor)

                log_mel = torchaudio.transforms.AmplitudeToDB(stype='power', top_db=80.0).cuda()(mel_spec)
                return log_mel.cpu().numpy()
            else:
                # Fallback to librosa
                mel_spec = librosa.feature.melspectrogram(
                    y=audio,
                    sr=self.config.sample_rate,
                    n_mels=self.config.n_mels,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    win_length=self.config.win_length,
                    fmin=self.config.fmin,
                    fmax=self.config.fmax,
                    power=2.0
                )

                log_mel = librosa.power_to_db(mel_spec, ref=np.max)
                return log_mel

        except Exception as e:
            logger.error(f"Mel-spectrogram extraction failed: {e}")
            # Return normalized zeros on error
            n_frames = int(self.config.sample_rate * self.config.duration / self.config.hop_length)
            return np.zeros((self.config.n_mels, n_frames), dtype=np.float32)

    def _get_default_item(self, label: int) -> Dict[str, torch.Tensor]:
        """Get default item for error cases"""
        features = self._get_default_features()
        return {
            'features': torch.FloatTensor(features),
            'label': torch.LongTensor([label]),
            'source': 'default',
            'path': 'default'
        }

    def _get_default_features(self) -> np.ndarray:
        """Get default features"""
        n_frames = int(self.config.sample_rate * self.config.duration / self.config.hop_length)
        return np.zeros((self.config.n_mels, n_frames), dtype=np.float32)

    def get_dataset_stats(self) -> Dict:
        """Get dataset statistics with GPU info"""
        stats = {
            'total_samples': len(self),
            'positive_samples': self.positive_count,
            'negative_samples': self.negative_count,
            'feature_samples': self.feature_count,
            'audio_samples': self.positive_count + self.negative_count,
            'balance_ratio': self.positive_count / (self.negative_count + 1e-8),
            'rirs_available': self.rir_augmentation is not None and self.rir_augmentation.is_available(),
            'feature_cache_available': self.feature_extractor is not None,
            'gpu_acceleration_enabled': torch.cuda.is_available() and self.config.use_gpu_acceleration,
            'num_workers_configured': self.config.num_workers,
            'batch_size_configured': self.config.batch_size
        }

        if self.feature_extractor:
            stats['cache_stats'] = self.feature_extractor.get_cache_stats()

        return stats

    def save_dataset_info(self, save_path: str):
        """Save dataset information to JSON file with performance metrics"""
        stats = self.get_dataset_stats()
        stats['config'] = {
            'sample_rate': self.config.sample_rate,
            'duration': self.config.duration,
            'n_mels': self.config.n_mels,
            'use_precomputed_features': self.config.use_precomputed_features,
            'use_rirs_augmentation': self.config.use_rirs_augmentation,
            'use_gpu_acceleration': self.config.use_gpu_acceleration,
            'mode': self.mode,
            'num_workers': self.config.num_workers,
            'batch_size': self.config.batch_size
        }
        stats['timestamp'] = datetime.now().isoformat()

        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Dataset info saved to {save_path}")


def create_dataloaders(positive_dir: str,
                      negative_dir: str,
                      val_positive_dir: str = None,
                      val_negative_dir: str = None,
                      features_dir: str = None,
                      rirs_dir: str = None,
                      batch_size: int = 32,
                      config: EnhancedAudioConfig = None) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create optimized train and validation dataloaders with GPU acceleration"""

    # Use separate directories for validation if provided
    val_positive_dir = val_positive_dir or positive_dir.replace('train', 'val')
    val_negative_dir = val_negative_dir or negative_dir.replace('train', 'val')

    logger.info("Creating optimized dataloaders with GPU acceleration...")

    # Create datasets
    train_dataset = EnhancedWakewordDataset(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        features_dir=features_dir,
        rirs_dir=rirs_dir,
        config=config,
        mode='train'
    )

    val_dataset = EnhancedWakewordDataset(
        positive_dir=val_positive_dir,
        negative_dir=val_negative_dir,
        features_dir=features_dir,
        rirs_dir=rirs_dir,
        config=config,
        mode='validation'
    )

    # Optimized DataLoader configuration
    train_config = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': config.num_workers if config else 4,
        'pin_memory': True,
        'drop_last': False,
        'prefetch_factor': config.prefetch_factor if config else 2,
        'persistent_workers': True  # Keep workers alive between epochs
    }

    val_config = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': max(1, (config.num_workers if config else 4) // 2),  # Fewer workers for validation
        'pin_memory': True,
        'drop_last': False,
        'prefetch_factor': max(1, (config.prefetch_factor if config else 2) // 2),
        'persistent_workers': True
    }

    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_config)
    val_loader = torch.utils.data.DataLoader(val_dataset, **val_config)

    logger.info(f"Created dataloaders:")
    logger.info(f"  - Train batches: {len(train_loader)} (batch_size={batch_size})")
    logger.info(f"  - Validation batches: {len(val_loader)} (batch_size={batch_size})")
    logger.info(f"  - GPU acceleration: {'Enabled' if torch.cuda.is_available() and (config.use_gpu_acceleration if config else True) else 'Disabled'}")

    return train_loader, val_loader


def main():
    """Test enhanced dataset functionality with GPU acceleration"""
    logger.info("Testing enhanced dataset with GPU acceleration...")

    # Create configuration with GPU acceleration
    config = EnhancedAudioConfig(
        use_precomputed_features=False,  # Force audio processing to test GPU
        use_rirs_augmentation=False,
        use_gpu_acceleration=True,
        num_workers=4,
        batch_size=16
    )

    # Create dataset
    dataset = EnhancedWakewordDataset(
        positive_dir="positive_dataset",
        negative_dir="negative_dataset",
        features_dir="features/train",
        rirs_dir="datasets/mit_rirs/rir_data",
        config=config,
        mode="train"
    )

    # Test dataset
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Dataset stats: {dataset.get_dataset_stats()}")

    # Test data loading with timing
    if len(dataset) > 0:
        start_time = time.time()
        sample = dataset[0]
        load_time = time.time() - start_time

        logger.info(f"Sample loaded in {load_time:.3f}s")
        logger.info(f"Sample features shape: {sample['features'].shape}")
        logger.info(f"Sample label: {sample['label']}")
        logger.info(f"Sample source: {sample['source']}")

    # Test batch processing
    if len(dataset) > 10:
        logger.info("Testing batch processing...")
        batch_start = time.time()

        batch_samples = []
        for i in range(min(10, len(dataset))):
            batch_samples.append(dataset[i])

        batch_time = time.time() - batch_start
        logger.info(f"Batch of 10 samples loaded in {batch_time:.3f}s")
        logger.info(f"Average load time per sample: {batch_time/10:.3f}s")

    # Save dataset info
    dataset.save_dataset_info("dataset_info_enhanced.json")

    logger.info("Enhanced dataset test completed")


if __name__ == "__main__":
    main()
