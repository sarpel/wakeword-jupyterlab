#!/usr/bin/env python3
"""
Enhanced Feature Extractor for Wakeword Training
Supports pre-computed .npy features and MIT RIRS augmentation
"""

import os
import numpy as np
import librosa
import soundfile as sf
import torch
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import time
from dataclasses import dataclass
import hashlib


@dataclass
class FeatureConfig:
    """Configuration class for feature extraction"""
    sample_rate: int = 16000
    duration: float = 2.0
    n_mels: int = 40
    n_fft: int = 1024
    hop_length: int = 160
    win_length: int = 400
    fmin: int = 20
    fmax: int = 8000
    power: float = 2.0
    normalized: bool = True
    delta: bool = True
    delta_delta: bool = False
    mean_norm: bool = True
    var_norm: bool = False


class FeatureExtractor:
    """Enhanced feature extractor with caching and pre-computed feature support"""

    def __init__(self, config_path: str = "config/feature_config.yaml"):
        self.config = self._load_config(config_path)
        self.feature_config = FeatureConfig(**self.config.get('mel', {}))
        self.cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}

        # Create cache directory
        self.cache_dir = Path("features/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load persistent cache if exists
        self._load_persistent_cache()

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"Config file not found: {config_path}")
            return {}

    def _load_persistent_cache(self):
        """Load persistent cache from disk"""
        cache_file = self.cache_dir / "feature_cache.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                print(f"Loaded {len(self.cache)} cached features")
            except Exception as e:
                print(f"Error loading cache: {e}")
                self.cache = {}

    def _save_persistent_cache(self):
        """Save persistent cache to disk"""
        cache_file = self.cache_dir / "feature_cache.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
        except Exception as e:
            print(f"Error saving cache: {e}")

    def _get_cache_key(self, audio_path: str) -> str:
        """Generate cache key for audio file"""
        stat = os.stat(audio_path)
        file_info = f"{audio_path}:{stat.st_mtime}:{stat.st_size}"
        return hashlib.md5(file_info.encode()).hexdigest()

    def _get_feature_path(self, audio_path: str, feature_type: str = "train") -> str:
        """Get feature file path for audio file"""
        audio_name = Path(audio_path).stem
        parent_dir = Path(audio_path).parent.name

        # Determine feature directory
        if 'positive' in parent_dir.lower():
            feature_dir = f"features/{feature_type}/positive"
        elif 'negative' in parent_dir.lower():
            feature_dir = f"features/{feature_type}/negative"
        else:
            feature_dir = f"features/{feature_type}"

        # Ensure directory exists
        Path(feature_dir).mkdir(parents=True, exist_ok=True)

        return f"{feature_dir}/{audio_name}.npy"

    def extract_features(self, audio_path: str, feature_type: str = "train") -> np.ndarray:
        """Extract mel-spectrogram features from audio file"""

        # Check cache first
        cache_key = self._get_cache_key(audio_path)
        if cache_key in self.cache:
            self.cache_stats['hits'] += 1
            return self.cache[cache_key]

        self.cache_stats['misses'] += 1

        try:
            # Load audio
            audio, sr = librosa.load(
                audio_path,
                sr=self.feature_config.sample_rate,
                duration=self.feature_config.duration
            )

            # Pad or trim to exact duration
            target_length = int(self.feature_config.sample_rate * self.feature_config.duration)
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
            else:
                audio = audio[:target_length]

            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=sr,
                n_mels=self.feature_config.n_mels,
                n_fft=self.feature_config.n_fft,
                hop_length=self.feature_config.hop_length,
                win_length=self.feature_config.win_length,
                fmin=self.feature_config.fmin,
                fmax=self.feature_config.fmax,
                power=self.feature_config.power
            )

            # Convert to log scale
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)

            # Apply delta features if enabled
            features = [log_mel]
            if self.feature_config.delta:
                delta = librosa.feature.delta(log_mel)
                features.append(delta)

            if self.feature_config.delta_delta:
                delta_delta = librosa.feature.delta(log_mel, order=2)
                features.append(delta_delta)

            # Stack features
            feature_stack = np.vstack(features)

            # Normalize features
            if self.feature_config.mean_norm:
                feature_stack = feature_stack - feature_stack.mean(axis=1, keepdims=True)

            if self.feature_config.var_norm:
                feature_stack = feature_stack / (feature_stack.std(axis=1, keepdims=True) + 1e-8)

            # Save as .npy file
            feature_path = self._get_feature_path(audio_path, feature_type)
            np.save(feature_path, feature_stack.astype(np.float32))

            # Cache result
            self.cache[cache_key] = feature_stack

            # Save cache periodically
            if len(self.cache) % 100 == 0:
                self._save_persistent_cache()

            return feature_stack

        except Exception as e:
            print(f"Error extracting features from {audio_path}: {e}")
            # Return zero features on error
            n_frames = int(self.feature_config.sample_rate * self.feature_config.duration / self.feature_config.hop_length)
            n_features = self.feature_config.n_mels
            if self.feature_config.delta:
                n_features *= 2
            if self.feature_config.delta_delta:
                n_features *= 3

            return np.zeros((n_features, n_frames), dtype=np.float32)

    def extract_features_batch(self, audio_paths: List[str], feature_type: str = "train") -> List[np.ndarray]:
        """Extract features from multiple audio files"""
        features = []
        for path in audio_paths:
            feat = self.extract_features(path, feature_type)
            features.append(feat)
        return features

    def load_features(self, feature_path: str) -> np.ndarray:
        """Load pre-computed features from .npy file"""
        try:
            return np.load(feature_path)
        except Exception as e:
            print(f"Error loading features from {feature_path}: {e}")
            return np.zeros((self.feature_config.n_mels, 126), dtype=np.float32)  # Default size

    def get_cache_stats(self) -> Dict:
        """Get cache performance statistics"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0

        return {
            'cache_size': len(self.cache),
            'hits': self.cache_stats['hits'],
            'misses': self.cache_stats['misses'],
            'hit_rate': hit_rate,
            'total_requests': total_requests
        }

    def clear_cache(self):
        """Clear feature cache"""
        self.cache.clear()
        self.cache_stats = {'hits': 0, 'misses': 0}

        # Clear persistent cache
        cache_file = self.cache_dir / "feature_cache.pkl"
        if cache_file.exists():
            cache_file.unlink()

    def preprocess_dataset(self, data_dir: str, feature_type: str = "train"):
        """Preprocess entire dataset directory"""
        print(f"Preprocessing {data_dir}...")

        # Find all audio files
        audio_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.wav', '.flac', '.mp3', '.ogg')):
                    audio_files.append(os.path.join(root, file))

        print(f"Found {len(audio_files)} audio files")

        # Extract features
        features = []
        for i, audio_path in enumerate(audio_files):
            if i % 100 == 0:
                print(f"Processing {i+1}/{len(audio_files)}")

            feat = self.extract_features(audio_path, feature_type)
            features.append(feat)

        print(f"Completed preprocessing {len(features)} files")
        print(f"Cache stats: {self.get_cache_stats()}")

        return features


class RIRAugmentation:
    """Room Impulse Response augmentation for acoustic simulation"""

    def __init__(self, rirs_dataset_path: str = "datasets/mit_rirs/rir_data"):
        self.rirs_path = rirs_dataset_path
        self.rir_files = self._load_rir_files()

    def _load_rir_files(self) -> List[str]:
        """Load all RIR files from dataset"""
        rir_files = []
        if os.path.exists(self.rirs_path):
            for root, dirs, files in os.walk(self.rirs_path):
                for file in files:
                    if file.endswith(('.wav', '.flac')):
                        rir_files.append(os.path.join(root, file))

        print(f"Loaded {len(rir_files)} RIR files")
        return rir_files

    def apply_rir(self, audio: np.ndarray, sr: int = 16000, snr_range: Tuple[float, float] = (5, 20)) -> np.ndarray:
        """Apply random RIR with specified SNR range"""
        if not self.rir_files:
            return audio  # No RIRS available, return original

        try:
            # Select random RIR
            rir_path = np.random.choice(self.rir_files)
            rir, sr_rir = librosa.load(rir_path, sr=sr)

            # Convolve audio with RIR
            reverberant = np.convolve(audio, rir, mode='same')

            # Normalize to target SNR
            target_snr = np.random.uniform(snr_range[0], snr_range[1])
            reverberant = self._adjust_snr(audio, reverberant, target_snr)

            return reverberant

        except Exception as e:
            print(f"Error applying RIR: {e}")
            return audio

    def _adjust_snr(self, clean: np.ndarray, noisy: np.ndarray, target_snr: float) -> np.ndarray:
        """Adjust SNR between clean and noisy signals"""
        clean_power = np.mean(clean ** 2)
        noisy_power = np.mean(noisy ** 2)

        if noisy_power > 0:
            scale = np.sqrt(clean_power / (noisy_power * 10 ** (target_snr / 10)))
            noisy = noisy * scale

        return noisy

    def is_available(self) -> bool:
        """Check if RIRS dataset is available"""
        return len(self.rir_files) > 0


def main():
    """Test feature extraction functionality"""
    # Initialize feature extractor
    extractor = FeatureExtractor()

    # Test on a sample file (if available)
    test_files = []
    for dataset_dir in ['positive_dataset', 'negative_dataset']:
        if os.path.exists(dataset_dir):
            for root, dirs, files in os.walk(dataset_dir):
                for file in files:
                    if file.endswith('.wav'):
                        test_files.append(os.path.join(root, file))
                        break
                if test_files:
                    break
        if test_files:
            break

    if test_files:
        print(f"Testing feature extraction on {test_files[0]}")
        features = extractor.extract_features(test_files[0])
        print(f"Feature shape: {features.shape}")
        print(f"Cache stats: {extractor.get_cache_stats()}")
    else:
        print("No audio files found for testing")


if __name__ == "__main__":
    main()