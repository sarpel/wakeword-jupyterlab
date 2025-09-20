#!/usr/bin/env python3
"""
Enhanced Wakeword Dataset with .npy Feature File and MIT RIRS Support
"""

import os
import numpy as np
import torch
import librosa
import soundfile as sf
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import random
import json
from dataclasses import dataclass
import yaml

from feature_extractor import FeatureExtractor, RIRAugmentation


@dataclass
class EnhancedAudioConfig:
    """Enhanced audio configuration with feature extraction options"""
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


class EnhancedWakewordDataset(Dataset):
    """Enhanced dataset with .npy feature file and MIT RIRS support"""

    def __init__(self,
                 positive_dir: str,
                 negative_dir: str,
                 features_dir: str = None,
                 rirs_dir: str = None,
                 config: EnhancedAudioConfig = None,
                 mode: str = "train"):
        """
        Initialize enhanced wakeword dataset

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

        # Initialize components
        self.feature_extractor = None
        self.rir_augmentation = None

        if self.config.use_precomputed_features:
            self.feature_extractor = FeatureExtractor(self.config.feature_config_path)

        if self.config.use_rirs_augmentation:
            self.rir_augmentation = RIRAugmentation(self.config.rirs_dataset_path)

        # Load data
        self.positive_files = self._load_audio_files(positive_dir)
        self.negative_files = self._load_audio_files(negative_dir)
        self.feature_files = self._load_feature_files() if features_dir else []

        # Dataset statistics
        self.positive_count = len(self.positive_files)
        self.negative_count = len(self.negative_files)
        self.feature_count = len(self.feature_files)

        # Balance dataset
        self._balance_dataset()

        # Create labels
        self._create_labels()

        print(f"Dataset initialized:")
        print(f"  - Mode: {self.mode}")
        print(f"  - Positive audio files: {self.positive_count}")
        print(f"  - Negative audio files: {self.negative_count}")
        print(f"  - Pre-computed features: {self.feature_count}")
        print(f"  - Total samples: {len(self)}")
        print(f"  - RIRS augmentation: {'Enabled' if self.rir_augmentation else 'Disabled'}")
        print(f"  - Feature caching: {'Enabled' if self.feature_extractor else 'Disabled'}")

    def _load_audio_files(self, directory: str) -> List[str]:
        """Load all audio files from directory"""
        audio_files = []
        if os.path.exists(directory):
            for root, dirs, files in os.walk(directory):
                for file in files:
                    if file.lower().endswith(('.wav', '.flac', '.mp3', '.ogg', '.m4a')):
                        audio_files.append(os.path.join(root, file))
        return audio_files

    def _load_feature_files(self) -> List[str]:
        """Load all .npy feature files"""
        feature_files = []
        if os.path.exists(self.features_dir):
            for root, dirs, files in os.walk(self.features_dir):
                for file in files:
                    if file.endswith('.npy'):
                        feature_files.append(os.path.join(root, file))
        return feature_files

    def _balance_dataset(self):
        """Balance positive and negative samples"""
        max_samples = max(self.positive_count, self.negative_count)

        # Duplicate positive samples if needed
        if self.positive_count < max_samples:
            extra_needed = max_samples - self.positive_count
            self.positive_files.extend(self.positive_files[:extra_needed])

        # Duplicate negative samples if needed
        if self.negative_count < max_samples:
            extra_needed = max_samples - self.negative_count
            self.negative_files.extend(self.negative_files[:extra_needed])

        # Update counts
        self.positive_count = len(self.positive_files)
        self.negative_count = len(self.negative_files)

    def _create_labels(self):
        """Create labels for all samples"""
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

    def __len__(self) -> int:
        """Get dataset size"""
        return self.feature_count + self.positive_count + self.negative_count

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get dataset item"""
        label = self.labels[idx]
        source = self.sources[idx]

        try:
            if source == 'feature':
                # Load pre-computed features
                feature_path = self.feature_files[idx]
                features = self._load_features(feature_path)
                return {
                    'features': torch.FloatTensor(features),
                    'label': torch.LongTensor([label]),
                    'source': 'feature',
                    'path': feature_path
                }
            else:
                # Load and process audio
                if idx < self.feature_count + self.positive_count:
                    audio_idx = idx - self.feature_count
                    audio_path = self.positive_files[audio_idx]
                else:
                    audio_idx = idx - self.feature_count - self.positive_count
                    audio_path = self.negative_files[audio_idx]

                return self._load_and_process_audio(audio_path, label)

        except Exception as e:
            print(f"Error loading item {idx}: {e}")
            return self._get_default_item(label)

    def _load_features(self, feature_path: str) -> np.ndarray:
        """Load pre-computed .npy features"""
        try:
            features = np.load(feature_path)

            # Apply RIRS augmentation to features if enabled
            if self.rir_augmentation and np.random.rand() < self.config.rirs_probability:
                # Note: RIRS augmentation typically requires audio domain
                # For feature domain, we'll apply alternative augmentation
                features = self._apply_feature_augmentation(features)

            return features

        except Exception as e:
            print(f"Error loading features from {feature_path}: {e}")
            return self._get_default_features()

    def _load_and_process_audio(self, audio_path: str, label: int) -> Dict[str, torch.Tensor]:
        """Load audio file and process it"""
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

            # Extract features
            if self.feature_extractor:
                features = self.feature_extractor.extract_features(audio_path)
            else:
                features = self._extract_mel_spectrogram(audio)

            return {
                'features': torch.FloatTensor(features),
                'label': torch.LongTensor([label]),
                'source': 'audio',
                'path': audio_path
            }

        except Exception as e:
            print(f"Error processing audio {audio_path}: {e}")
            return self._get_default_item(label)

    def _apply_augmentations(self, audio: np.ndarray) -> np.ndarray:
        """Apply audio augmentations"""
        augmented = audio.copy()

        # Time shift
        if np.random.rand() < 0.5:
            augmented = self._time_shift(augmented)

        # RIRS augmentation
        if self.rir_augmentation and np.random.rand() < self.config.rirs_probability:
            augmented = self.rir_augmentation.apply_rir(
                augmented,
                sr=self.config.sample_rate,
                snr_range=self.config.rirs_snr_range
            )

        # Pitch shift
        if np.random.rand() < 0.3:
            augmented = self._pitch_shift(augmented)

        # Speed change
        if np.random.rand() < 0.3:
            augmented = self._speed_change(augmented)

        # Noise addition
        if np.random.rand() < 0.5:
            augmented = self._add_noise(augmented)

        return augmented

    def _apply_feature_augmentation(self, features: np.ndarray) -> np.ndarray:
        """Apply augmentations to pre-computed features"""
        augmented = features.copy()

        # Time shift in feature domain
        if np.random.rand() < 0.5:
            shift = int(np.random.uniform(-10, 10))
            augmented = np.roll(augmented, shift, axis=1)

        # Frequency masking
        if np.random.rand() < 0.3:
            fmask_param = int(features.shape[0] * 0.1)
            f0 = np.random.randint(0, features.shape[0] - fmask_param)
            augmented[f0:f0 + fmask_param, :] = 0

        # Time masking
        if np.random.rand() < 0.3:
            tmask_param = int(features.shape[1] * 0.1)
            t0 = np.random.randint(0, features.shape[1] - tmask_param)
            augmented[:, t0:t0 + tmask_param] = 0

        # Add noise
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.1, features.shape)
            augmented = augmented + noise

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
        """Extract mel-spectrogram from audio"""
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

        # Normalize
        log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

        return log_mel

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
        """Get dataset statistics"""
        stats = {
            'total_samples': len(self),
            'positive_samples': self.positive_count,
            'negative_samples': self.negative_count,
            'feature_samples': self.feature_count,
            'audio_samples': self.positive_count + self.negative_count,
            'balance_ratio': self.positive_count / (self.negative_count + 1e-8),
            'rirs_available': self.rir_augmentation is not None and self.rir_augmentation.is_available(),
            'feature_cache_available': self.feature_extractor is not None
        }

        if self.feature_extractor:
            stats['cache_stats'] = self.feature_extractor.get_cache_stats()

        return stats

    def save_dataset_info(self, save_path: str):
        """Save dataset information to JSON file"""
        stats = self.get_dataset_stats()
        stats['config'] = {
            'sample_rate': self.config.sample_rate,
            'duration': self.config.duration,
            'n_mels': self.config.n_mels,
            'use_precomputed_features': self.config.use_precomputed_features,
            'use_rirs_augmentation': self.config.use_rirs_augmentation,
            'mode': self.mode
        }

        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Dataset info saved to {save_path}")


def create_dataloaders(positive_dir: str,
                       negative_dir: str,
                       val_positive_dir: Optional[str] = None,
                       val_negative_dir: Optional[str] = None,
                       features_dir: str = None,
                       rirs_dir: str = None,
                       batch_size: int = 32,
                       config: Optional[EnhancedAudioConfig] = None) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create train and validation dataloaders with sensible fallbacks."""

    config = config or EnhancedAudioConfig()
    resolved_features_dir = features_dir or config.features_dir

    def _resolve_split_dir(train_dir: str, explicit: Optional[str]) -> str:
        if explicit:
            return explicit

        train_path = Path(train_dir)
        candidates = []

        if 'train' in train_path.name:
            candidates.append(train_path.with_name(train_path.name.replace('train', 'val')))

        candidates.append(train_path.parent / 'val' / train_path.name)

        for candidate in candidates:
            if candidate.exists():
                return str(candidate)

        return train_dir

    val_positive_dir = _resolve_split_dir(positive_dir, val_positive_dir)
    val_negative_dir = _resolve_split_dir(negative_dir, val_negative_dir)

    train_dataset = EnhancedWakewordDataset(
        positive_dir=positive_dir,
        negative_dir=negative_dir,
        features_dir=resolved_features_dir,
        rirs_dir=rirs_dir,
        config=config,
        mode='train'
    )

    val_dataset = EnhancedWakewordDataset(
        positive_dir=val_positive_dir,
        negative_dir=val_negative_dir,
        features_dir=resolved_features_dir,
        rirs_dir=rirs_dir,
        config=config,
        mode='validation'
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False
    )

    return train_loader, val_loader


def main():
    """Test enhanced dataset functionality"""
    # Create configuration
    config = EnhancedAudioConfig(
        use_precomputed_features=True,
        use_rirs_augmentation=False,  # Set to True when RIRS dataset is available
        features_dir="features/train",
        rirs_dataset_path="datasets/mit_rirs/rir_data"
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
    print(f"Dataset size: {len(dataset)}")
    print(f"Dataset stats: {dataset.get_dataset_stats()}")

    # Test data loading
    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample features shape: {sample['features'].shape}")
        print(f"Sample label: {sample['label']}")
        print(f"Sample source: {sample['source']}")

    # Save dataset info
    dataset.save_dataset_info("dataset_info.json")


if __name__ == "__main__":
    main()
