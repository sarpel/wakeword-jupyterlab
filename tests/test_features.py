#!/usr/bin/env python3
"""
Test Enhanced Wakeword Dataset with .npy Feature Files and RIRS Support
"""

import os
import sys
import torch
import numpy as np
import time
from pathlib import Path
import json

# Use relative imports or proper package structure instead

try:
    import unittest
    from enhanced_dataset import EnhancedWakewordDataset, EnhancedAudioConfig, create_dataloaders
    from feature_extractor import FeatureExtractor, RIRAugmentation
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please ensure enhanced_dataset.py and feature_extractor.py are in the current directory")
    sys.exit(1)
class TestEnhancedFeatures(unittest.TestCase):
    """Test class for enhanced wakeword dataset features"""

    def setUp(self):
        self.test_results = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def run_comprehensive_test(self):        """Run comprehensive test of all enhanced features"""
    # Remove this method and let unittest handle test discovery and execution        sources = {
            'positive_dataset': Path("positive_dataset").exists(),
            'negative_dataset': Path("negative_dataset").exists(),
            'background_noise': Path("background_noise").exists(),
            'features': Path("features").exists(),
            'config': Path("config/feature_config.yaml").exists(),
            'rirs_dataset': Path("datasets/mit_rirs/rir_data").exists()
        }

        # Count files in available directories
        if sources['positive_dataset']:
            pos_files = len(list(Path("positive_dataset").rglob("*.wav")))
            self.assertGreater(pos_files, 0, "No positive audio files found")

        if sources['negative_dataset']:
            neg_files = len(list(Path("negative_dataset").rglob("*.wav")))
            self.assertGreater(neg_files, 0, "No negative audio files found")

        if sources['features']:
            feature_files = len(list(Path("features").rglob("*.npy")))
            print(f"    📊 Feature files: {feature_files}")

        if sources['rirs_dataset']:
            rir_files = len(list(Path("datasets/mit_rirs/rir_data").rglob("*")))
            print(f"    📊 RIRS files: {rir_files}")

        self.test_results['data_sources'] = sources

        # Assert essential directories exist
        self.assertTrue(sources['negative_dataset'], "Negative dataset directory not found")

    def test_feature_extractor(self):        self.assertTrue(sources['negative_dataset'], "Negative dataset directory not found")    def test_feature_extractor(self):
        """Test feature extractor functionality"""
        print("  🔄 Testing feature extractor...")

        try:
            # Initialize feature extractor
            extractor = FeatureExtractor("config/feature_config.yaml")
            print("    ✅ Feature extractor initialized")

            # Test cache functionality
            cache_stats = extractor.get_cache_stats()
            print(f"    📊 Cache stats: {cache_stats}")

            # Find a test audio file
            test_files = []
            for dataset_dir in ['positive_dataset', 'negative_dataset']:
                if Path(dataset_dir).exists():
                    audio_files = list(Path(dataset_dir).rglob("*.wav"))
                    if audio_files:
                        test_files.append(audio_files[0])
                        break

            if not test_files:
                print("    ⚠️  No audio files found for feature extraction test")
                return True  # Pass if no files to test

            # Test feature extraction
            test_file = test_files[0]
            print(f"    🎵 Testing with: {test_file}")

            start_time = time.time()
            features = extractor.extract_features(str(test_file))
            extraction_time = time.time() - start_time

            print(f"    📊 Feature shape: {features.shape}")
            print(f"    ⏱️  Extraction time: {extraction_time:.3f}s")

            # Test caching
            start_time = time.time()
            cached_features = extractor.extract_features(str(test_file))
            cache_time = time.time() - start_time

            cache_speedup = extraction_time / cache_time if cache_time > 0 else float('inf')
            print(f"    📈 Cache speedup: {cache_speedup:.1f}x")

            # Verify features are identical
            features_match = np.allclose(features, cached_features)
            print(f"    ✅ Cache consistency: {features_match}")

            self.test_results['feature_extractor'] = {
                'extraction_time': extraction_time,
                'cache_time': cache_time,
                'cache_speedup': cache_speedup,
                'feature_shape': features.shape,
                'cache_hit_rate': extractor.cache_stats['hit_rate']
            }

            return features_match

        except Exception as e:
            print(f"    ❌ Feature extractor test failed: {e}")
            return False

    def test_rirs_augmentation(self):
        """Test RIRS augmentation functionality"""
        print("  🏠 Testing RIRS augmentation...")

        try:
            # Initialize RIRS augmentation
            rirs = RIRAugmentation("datasets/mit_rirs/rir_data")
            print(f"    📊 RIRS files loaded: {len(rirs.rir_files)}")

            if not rirs.rir_files:
                print("    ⚠️  No RIRS files available for testing")
                return True  # Pass if no RIRS files

            # Create test audio
            test_audio = np.random.randn(16000 * 2)  # 2 seconds of noise
            original_audio = test_audio.copy()

            # Test RIRS application
            print("    🔄 Applying RIRS augmentation...")
            start_time = time.time()
            augmented_audio = rirs.apply_rir(test_audio)
            augmentation_time = time.time() - start_time

            print(f"    ⏱️  Augmentation time: {augmentation_time:.3f}s")
            print(f"    📊 Original shape: {original_audio.shape}")
            print(f"    📊 Augmented shape: {augmented_audio.shape}")

            # Check that audio was modified
            audio_modified = not np.allclose(original_audio, augmented_audio)
            print(f"    ✅ Audio modified: {audio_modified}")

            self.test_results['rirs_augmentation'] = {
                'rirs_files_count': len(rirs.rir_files),
                'augmentation_time': augmentation_time,
                'audio_modified': audio_modified
            }

            return True

        except Exception as e:
            print(f"    ❌ RIRS augmentation test failed: {e}")
            return False

    def test_enhanced_dataset(self):
        """Test enhanced dataset functionality"""
        print("  📊 Testing enhanced dataset...")

        try:
            # Create configuration
            config = EnhancedAudioConfig()
            config.use_precomputed_features = True
            config.use_rirs_augmentation = True

            # Create dataset
            dataset = EnhancedWakewordDataset(
                positive_dir="positive_dataset",
                negative_dir="negative_dataset",
                features_dir="features",
                rirs_dir="datasets/mit_rirs/rir_data",
                config=config,
                mode="train"
            )

            print(f"    📊 Dataset size: {len(dataset)}")

            if len(dataset) == 0:
                print("    ⚠️  Empty dataset - no audio files found")
                return True  # Pass if no data

            # Test data loading
            print("    🔄 Testing data loading...")
            start_time = time.time()

            # Test multiple samples
            sample_times = []
            for i in range(min(5, len(dataset))):
                sample_start = time.time()
                sample = dataset[i]
                sample_time = time.time() - sample_start
                sample_times.append(sample_time)

                print(f"      Sample {i}: shape={sample['features'].shape}, "
                      f"label={sample['label'].item()}, source={sample['source']}")

            avg_sample_time = np.mean(sample_times)
            print(f"    ⏱️  Average sample loading time: {avg_sample_time:.3f}s")

            # Test dataset statistics
            stats = dataset.get_dataset_stats()
            print(f"    📊 Dataset stats: {stats}")

            # Test batch loading with DataLoader
            print("    🔄 Testing batch loading...")
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=4,
                shuffle=True,
                num_workers=0
            )

            batch_start = time.time()
            batch = next(iter(dataloader))
            batch_time = time.time() - batch_start

            print(f"    📊 Batch shape: {batch['features'].shape}")
            print(f"    📊 Labels: {batch['label']}")
            print(f"    ⏱️  Batch loading time: {batch_time:.3f}s")

            self.test_results['enhanced_dataset'] = {
                'dataset_size': len(dataset),
                'avg_sample_time': avg_sample_time,
                'batch_time': batch_time,
                'sample_shapes': [sample['features'].shape for sample in [dataset[i] for i in range(min(3, len(dataset)))]],
                'stats': stats
            }

            return True

        except Exception as e:
            print(f"    ❌ Enhanced dataset test failed: {e}")
            return False

    def test_performance(self):
        """Test performance improvements with enhanced features"""
        print("  ⚡ Testing performance improvements...")

        try:
            # Test with and without precomputed features
            configs = [
                ("Standard", EnhancedAudioConfig(use_precomputed_features=False, use_rirs_augmentation=False)),
                ("Features Only", EnhancedAudioConfig(use_precomputed_features=True, use_rirs_augmentation=False)),
                ("Full Enhanced", EnhancedAudioConfig(use_precomputed_features=True, use_rirs_augmentation=True))
            ]

            performance_results = {}

            for config_name, config in configs:
                print(f"    🔄 Testing {config_name} configuration...")

                dataset = EnhancedWakewordDataset(
                    positive_dir="positive_dataset",
                    negative_dir="negative_dataset",
                    features_dir="features",
                    rirs_dir="datasets/mit_rirs/rir_data",
                    config=config,
                    mode="train"
                )

                if len(dataset) == 0:
                    print(f"      ⚠️  No data for {config_name}")
                    continue

                # Time data loading
                start_time = time.time()
                samples = []
                for i in range(min(10, len(dataset))):
                    samples.append(dataset[i])
                loading_time = time.time() - start_time

                avg_time = loading_time / len(samples)
                performance_results[config_name] = {
                    'loading_time': avg_time,
                    'dataset_size': len(dataset)
                }

                print(f"      ⏱️  Average loading time: {avg_time:.3f}s")

            # Compare performance
            if "Standard" in performance_results and "Features Only" in performance_results:
                speedup = performance_results["Standard"]["loading_time"] / performance_results["Features Only"]["loading_time"]
                print(f"    📈 Feature speedup: {speedup:.1f}x")

            self.test_results['performance'] = performance_results
            return True

        except Exception as e:
            print(f"    ❌ Performance test failed: {e}")
            return False

    def test_integration(self):
        """Test full integration of all components"""
        print("  🔗 Testing full integration...")

        try:
            # Test complete workflow
            config = EnhancedAudioConfig(
                use_precomputed_features=True,
                use_rirs_augmentation=True,
                features_dir="features",
                rirs_dataset_path="datasets/mit_rirs/rir_data"
            )

            # Create datasets
            train_dataset = EnhancedWakewordDataset(
                positive_dir="positive_dataset",
                negative_dir="negative_dataset",
                features_dir="features",
                rirs_dir="datasets/mit_rirs/rir_data",
                config=config,
                mode="train"
            )

            val_dataset = EnhancedWakewordDataset(
                positive_dir="positive_dataset",
                negative_dir="negative_dataset",
                features_dir="features",
                rirs_dir="datasets/mit_rirs/rir_data",
                config=config,
                mode="validation"
            )

            # Create dataloaders
            train_loader, val_loader = create_dataloaders(
                positive_dir="positive_dataset",
                negative_dir="negative_dataset",
                features_dir="features",
                rirs_dir="datasets/mit_rirs/rir_data",
                batch_size=8,
                config=config
            )

            print(f"    📊 Train dataset size: {len(train_dataset)}")
            print(f"    📊 Validation dataset size: {len(val_dataset)}")

            # Test training batch
            if len(train_dataset) > 0:
                train_batch = next(iter(train_loader))
                print(f"    📊 Train batch: features={train_batch['features'].shape}, "
                      f"labels={train_batch['label'].shape}")

            # Test validation batch
            if len(val_dataset) > 0:
                val_batch = next(iter(val_loader))
                print(f"    📊 Validation batch: features={val_batch['features'].shape}, "
                      f"labels={val_batch['label'].shape}")

            # Test dataset info saving
            train_dataset.save_dataset_info("test_dataset_info.json")

            if Path("test_dataset_info.json").exists():
                print("    ✅ Dataset info saved successfully")
                os.remove("test_dataset_info.json")

            print("    ✅ Integration test completed successfully")

            self.test_results['integration'] = {
                'train_dataset_size': len(train_dataset),
                'val_dataset_size': len(val_dataset),
                'train_batch_shape': train_batch['features'].shape if len(train_dataset) > 0 else None,
                'val_batch_shape': val_batch['features'].shape if len(val_dataset) > 0 else None
            }

            return True

        except Exception as e:
            print(f"    ❌ Integration test failed: {e}")
            return False

    def save_results(self, filename="test_results.json"):
        """Save test results to JSON file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.test_results, f, indent=2)
            print(f"\n💾 Test results saved to {filename}")
        except Exception as e:
            print(f"\n❌ Error saving results: {e}")

    def print_summary(self):
        """Print test summary"""
        print("\n📋 Test Summary")
        print("=" * 30)

        for test_name, result in self.test_results.items():
            if isinstance(result, dict):
                print(f"📊 {test_name}:")
                for key, value in result.items():
                    print(f"    {key}: {value}")
            else:
                print(f"📝 {test_name}: {result}")


def main():
    """Main test function"""
    print("🧪 Enhanced Wakeword Dataset Feature Test")
    print("=" * 50)

    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")

    # Run tests
    tester = EnhancedFeatureTester()
    success = tester.run_comprehensive_test()

    # Save and print results
    tester.save_results()
    tester.print_summary()

    if success:
        print("\n🎉 All tests passed! Enhanced features are working correctly.")
        print("\n🚀 Ready for enhanced wakeword training!")
    else:
        print("\n⚠️  Some tests failed. Check the results above for details.")
        print("\n🔧 You may need to:")
        print("   - Install required datasets")
        print("   - Configure feature extraction")
        print("   - Set up RIRS datasets")

    return success


if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        sys.exit(1)
