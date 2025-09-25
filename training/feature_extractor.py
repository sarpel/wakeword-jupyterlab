#!/usr/bin/env python3
"""
Enhanced Feature Extractor for Wakeword Training
Supports GPU-accelerated feature extraction and optimized caching
"""

import os
import numpy as np
import torch
import torchaudio
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pickle
import time
from dataclasses import dataclass
import hashlib
import logging
from datetime import datetime
import psutil
import gc
import librosa  # Keep for fallback and compatibility


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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
    use_gpu: bool = True  # New GPU acceleration flag


class GPUFeatureExtractor:
    """GPU-accelerated feature extractor with optimized caching"""

    def __init__(self, config_path: str = "config/feature_config.yaml", device: str = None):
        self.config = self._load_config(config_path)
        self.feature_config = FeatureConfig(**self.config.get('mel', {}))
        self.cache = {}
        self.cache_stats = {'hits': 0, 'misses': 0}

        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() and self.feature_config.use_gpu else 'cpu')
        else:
            self.device = torch.device(device)

        # Initialize GPU transforms if available
        self.gpu_transforms = None
        if self.device.type == 'cuda':
            self._initialize_gpu_transforms()

        # Performance tracking
        self.performance_stats = {
            'total_extraction_time': 0.0,
            'total_files_processed': 0,
            'avg_extraction_time': 0.0,
            'cpu_usage_peak': 0.0,
            'memory_usage_peak': 0.0,
            'gpu_available': torch.cuda.is_available(),
            'gpu_used': self.device.type == 'cuda',
            'cache_hit_rate': 0.0,
            'gpu_acceleration_enabled': self.device.type == 'cuda'
        }

        # Create cache directory
        self.cache_dir = Path("features/cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Load persistent cache if exists
        self._load_persistent_cache()

        logger.info(f"GPUFeatureExtractor initialized - Device: {self.device}")
        logger.info(f"GPU acceleration: {'Enabled' if self.device.type == 'cuda' else 'Disabled'}")

    def _initialize_gpu_transforms(self):
        """Initialize GPU-accelerated audio transforms"""
        try:
            self.gpu_transforms = {
                'mel_spectrogram': torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.feature_config.sample_rate,
                    n_fft=self.feature_config.n_fft,
                    hop_length=self.feature_config.hop_length,
                    win_length=self.feature_config.win_length,
                    n_mels=self.feature_config.n_mels,
                    f_min=self.feature_config.fmin,
                    f_max=self.feature_config.fmax,
                    power=self.feature_config.power
                ).to(self.device),

                'amplitude_to_db': torchaudio.transforms.AmplitudeToDB(
                    stype='power',
                    top_db=80.0
                ).to(self.device)
            }
            logger.info("GPU transforms initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize GPU transforms: {e}")
            self.gpu_transforms = None

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                # Ensure GPU acceleration is enabled by default if available
                if 'mel' in config and torch.cuda.is_available():
                    config['mel']['use_gpu'] = config['mel'].get('use_gpu', True)
                return config
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}")
            return {'mel': {}}

    def _load_persistent_cache(self):
        """Load persistent cache from disk"""
        cache_file = self.cache_dir / "feature_cache_gpu.pkl"
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    self.cache = pickle.load(f)
                logger.info(f"Loaded {len(self.cache)} cached features")
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                self.cache = {}

    def _save_persistent_cache(self):
        """Save persistent cache to disk"""
        cache_file = self.cache_dir / "feature_cache_gpu.pkl"
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(self.cache, f)
            logger.info(f"Saved {len(self.cache)} features to persistent cache")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")

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

    def _log_performance_metrics(self, start_time: float, audio_path: str, features_shape: Tuple, device_used: str):
        """Log performance metrics for feature extraction"""
        extraction_time = time.time() - start_time

        # Update performance stats
        self.performance_stats['total_extraction_time'] += extraction_time
        self.performance_stats['total_files_processed'] += 1
        self.performance_stats['avg_extraction_time'] = (
            self.performance_stats['total_extraction_time'] / self.performance_stats['total_files_processed']
        )

        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()

        # Update peak metrics
        self.performance_stats['cpu_usage_peak'] = max(self.performance_stats['cpu_usage_peak'], cpu_percent)
        self.performance_stats['memory_usage_peak'] = max(
            self.performance_stats['memory_usage_peak'],
            memory_info.percent
        )

        # Log detailed metrics
        logger.info(f"Feature extraction completed for {Path(audio_path).name}")
        logger.info(f"  Device used: {device_used}")
        logger.info(f"  Extraction time: {extraction_time:.3f}s")
        logger.info(f"  Features shape: {features_shape}")
        logger.info(f"  CPU usage: {cpu_percent:.1f}%")
        logger.info(f"  Memory usage: {memory_info.percent:.1f}%")
        logger.info(f"  Cache hit rate: {self.get_cache_stats()['hit_rate']:.2%}")

    def _extract_features_gpu(self, audio: torch.Tensor) -> np.ndarray:
        """Extract features using GPU acceleration"""
        try:
            if self.gpu_transforms is None:
                return self._extract_features_cpu_fallback(audio.cpu().numpy())

            # Ensure audio is on GPU
            audio_gpu = audio.to(self.device)

            # Extract mel-spectrogram on GPU
            mel_spec = self.gpu_transforms['mel_spectrogram'](audio_gpu)

            # Convert to dB scale on GPU
            log_mel = self.gpu_transforms['amplitude_to_db'](mel_spec)

            # Move back to CPU for numpy operations
            log_mel_cpu = log_mel.cpu().numpy()

            # Apply delta features if enabled
            features = [log_mel_cpu]
            if self.feature_config.delta:
                # Use torch for delta computation on GPU if possible
                delta = torchaudio.functional.compute_deltas(torch.from_numpy(log_mel_cpu).to(self.device), win_length=9)
                features.append(delta.cpu().numpy())

            if self.feature_config.delta_delta:
                delta_delta = torchaudio.functional.compute_deltas(torch.from_numpy(features[-1]).to(self.device), win_length=9)
                features.append(delta_delta.cpu().numpy())

            # Stack features
            feature_stack = np.vstack(features)

            # Normalize features
            if self.feature_config.mean_norm:
                feature_stack = feature_stack - feature_stack.mean(axis=1, keepdims=True)

            if self.feature_config.var_norm:
                feature_stack = feature_stack / (feature_stack.std(axis=1, keepdims=True) + 1e-8)

            return feature_stack

        except Exception as e:
            logger.warning(f"GPU extraction failed, falling back to CPU: {e}")
            return self._extract_features_cpu_fallback(audio.cpu().numpy())

    def _extract_features_cpu_fallback(self, audio: np.ndarray) -> np.ndarray:
        """Fallback CPU-based feature extraction using librosa"""
        try:
            # Extract mel-spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio,
                sr=self.feature_config.sample_rate,
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

            return feature_stack

        except Exception as e:
            logger.error(f"CPU fallback extraction failed: {e}")
            # Return zero features on error
            n_frames = int(self.feature_config.sample_rate * self.feature_config.duration / self.feature_config.hop_length)
            n_features = self.feature_config.n_mels
            if self.feature_config.delta:
                n_features += self.feature_config.n_mels
            if self.feature_config.delta_delta:
                n_features += self.feature_config.n_mels
            return np.zeros((n_features, n_frames), dtype=np.float32)

    def extract_features(self, audio_path: str, feature_type: str = "train") -> np.ndarray:
        """Extract mel-spectrogram features from audio file with GPU acceleration"""
        start_time = time.time()

        # Check cache first
        cache_key = self._get_cache_key(audio_path)
        if cache_key in self.cache:
            self.cache_stats['hits'] += 1
            logger.debug(f"Cache hit for {audio_path}")
            return self.cache[cache_key]

        self.cache_stats['misses'] += 1
        logger.debug(f"Cache miss for {audio_path}")

        try:
            # Load audio
            logger.debug(f"Loading audio: {audio_path}")

            # Try GPU-accelerated loading first
            if self.device.type == 'cuda':
                try:
                    # Load audio using torchaudio for GPU compatibility
                    audio_tensor, sr = torchaudio.load(audio_path)
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, sr, self.feature_config.sample_rate
                    )
                    audio = audio_tensor.squeeze().cpu().numpy()
                except Exception as e:
                    logger.debug(f"torchaudio loading failed, using librosa: {e}")
                    audio, sr = librosa.load(
                        audio_path,
                        sr=self.feature_config.sample_rate,
                        duration=self.feature_config.duration
                    )
            else:
                # Use librosa for CPU
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

            # Convert to tensor for GPU processing
            audio_tensor = torch.from_numpy(audio).float()

            # Extract features (GPU or CPU)
            if self.device.type == 'cuda' and self.gpu_transforms is not None:
                feature_stack = self._extract_features_gpu(audio_tensor)
                device_used = "GPU"
            else:
                feature_stack = self._extract_features_cpu_fallback(audio)
                device_used = "CPU"

            # Save as .npy file
            feature_path = self._get_feature_path(audio_path, feature_type)
            np.save(feature_path, feature_stack.astype(np.float32))
            logger.debug(f"Saved features to: {feature_path}")

            # Cache result
            self.cache[cache_key] = feature_stack

            # Save cache periodically
            if len(self.cache) % 100 == 0:
                self._save_persistent_cache()

            # Log performance metrics
            self._log_performance_metrics(start_time, audio_path, feature_stack.shape, device_used)

            return feature_stack

        except Exception as e:
            logger.error(f"Error extracting features from {audio_path}: {e}")
            # Return zero features on error
            n_frames = int(self.feature_config.sample_rate * self.feature_config.duration / self.feature_config.hop_length)
            n_features = self.feature_config.n_mels
            if self.feature_config.delta:
                n_features += self.feature_config.n_mels
            if self.feature_config.delta_delta:
                n_features += self.feature_config.n_mels
            return np.zeros((n_features, n_frames), dtype=np.float32)

    def extract_features_batch(self, audio_paths: List[str], feature_type: str = "train", batch_size: int = 32) -> List[np.ndarray]:
        """Extract features from multiple audio files with GPU batch processing"""
        logger.info(f"Starting batch feature extraction for {len(audio_paths)} files (batch_size={batch_size})")
        batch_start_time = time.time()

        features = []

        # Process in batches for GPU efficiency
        for i in range(0, len(audio_paths), batch_size):
            batch_paths = audio_paths[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(audio_paths) + batch_size - 1)//batch_size}")

            for j, path in enumerate(batch_paths):
                feat = self.extract_features(path, feature_type)
                features.append(feat)

        batch_time = time.time() - batch_start_time
        logger.info(f"Batch extraction completed in {batch_time:.3f}s")
        logger.info(f"Average time per file: {batch_time/len(audio_paths):.3f}s")

        return features

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

    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        cache_stats = self.get_cache_stats()

        return {
            'performance_metrics': self.performance_stats,
            'cache_statistics': cache_stats,
            'system_info': {
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / 1e9,
                'available_memory_gb': psutil.virtual_memory().available / 1e9,
                'gpu_available': torch.cuda.is_available(),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'gpu_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None
            }
        }

    def clear_cache(self):
        """Clear feature cache"""
        self.cache.clear()
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.performance_stats = {k: 0.0 if isinstance(v, float) else 0 for k, v in self.performance_stats.items() if k not in ['gpu_available', 'gpu_used', 'gpu_acceleration_enabled']}
        self.performance_stats['gpu_available'] = torch.cuda.is_available()
        self.performance_stats['gpu_used'] = self.device.type == 'cuda'
        self.performance_stats['gpu_acceleration_enabled'] = self.device.type == 'cuda'

        # Clear persistent cache
        cache_file = self.cache_dir / "feature_cache_gpu.pkl"
        if cache_file.exists():
            cache_file.unlink()

        logger.info("Cache cleared successfully")

    def preprocess_dataset(self, data_dir: str, feature_type: str = "train", batch_size: int = 64):
        """Preprocess entire dataset directory with GPU acceleration"""
        logger.info(f"Starting GPU-accelerated dataset preprocessing for {data_dir}")
        preprocess_start_time = time.time()

        # Get initial system stats
        initial_cpu = psutil.cpu_percent(interval=1)
        initial_memory = psutil.virtual_memory().percent
        initial_gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        logger.info(f"Initial system state - CPU: {initial_cpu:.1f}%, Memory: {initial_memory:.1f}%")
        if torch.cuda.is_available():
            logger.info(f"Initial GPU memory: {initial_gpu_memory:.3f} GB")

        # Find all audio files
        audio_files = []
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith(('.wav', '.flac', '.mp3', '.ogg')):
                    audio_files.append(os.path.join(root, file))

        logger.info(f"Found {len(audio_files)} audio files in {data_dir}")

        # Extract features with GPU batch processing
        features = []
        for i, audio_path in enumerate(audio_files):
            if i % 100 == 0:
                current_cpu = psutil.cpu_percent(interval=0.1)
                current_memory = psutil.virtual_memory().percent
                current_gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

                logger.info(f"Processing {i+1}/{len(audio_files)} - CPU: {current_cpu:.1f}%, Memory: {current_memory:.1f}%")

                if torch.cuda.is_available():
                    logger.info(f"GPU memory: {current_gpu_memory:.3f} GB")

            feat = self.extract_features(audio_path, feature_type)
            features.append(feat)

        # Final statistics
        total_time = time.time() - preprocess_start_time
        final_cpu = psutil.cpu_percent(interval=1)
        final_memory = psutil.virtual_memory().percent
        final_gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0

        logger.info(f"Dataset preprocessing completed in {total_time:.3f}s")
        logger.info(f"Final system state - CPU: {final_cpu:.1f}%, Memory: {final_memory:.1f}%")
        if torch.cuda.is_available():
            logger.info(f"Final GPU memory: {final_gpu_memory:.3f} GB")
        logger.info(f"Processed {len(features)} files")
        logger.info(f"Average processing time: {total_time/len(features):.3f}s per file")
        logger.info(f"Cache stats: {self.get_cache_stats()}")

        # Log comprehensive performance stats
        perf_stats = self.get_performance_stats()
        logger.info(f"Performance statistics: {perf_stats}")

        return features


# Keep the original FeatureExtractor for backward compatibility
class FeatureExtractor(GPUFeatureExtractor):
    """Backward compatibility wrapper for GPUFeatureExtractor"""
    pass


class RIRAugmentation:
    """Room Impulse Response augmentation for acoustic simulation (unchanged)"""

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

        logger.info(f"Loaded {len(rir_files)} RIR files")
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
            logger.error(f"Error applying RIR: {e}")
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
    """Test GPU-accelerated feature extraction"""
    logger.info("Starting GPU-accelerated feature extraction test...")

    # Initialize GPU feature extractor
    extractor = GPUFeatureExtractor()

    # Test GPU availability and performance
    logger.info(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU Device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

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
        logger.info(f"Testing GPU feature extraction on {test_files[0]}")

        # Test single file extraction
        start_time = time.time()
        features = extractor.extract_features(test_files[0])
        extraction_time = time.time() - start_time

        logger.info(f"Feature shape: {features.shape}")
        logger.info(f"Extraction time: {extraction_time:.3f}s")
        logger.info(f"Device used: {'GPU' if extractor.device.type == 'cuda' else 'CPU'}")
        logger.info(f"Cache stats: {extractor.get_cache_stats()}")

        # Test performance stats
        perf_stats = extractor.get_performance_stats()
        logger.info(f"Performance statistics:")
        for key, value in perf_stats.items():
            logger.info(f"  {key}: {value}")

    else:
        logger.warning("No audio files found for testing")

    logger.info("GPU feature extraction test completed")


if __name__ == "__main__":
    main()
