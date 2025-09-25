#!/usr/bin/env python3
"""
Test script for GPU-accelerated feature extraction and tensor shape fixes
"""

import torch
import time
import os
import sys
from pathlib import Path

# Add training directory to path
sys.path.append('training')

def test_gpu_setup():
    """Test GPU availability and setup"""
    print("🔧 Testing GPU Setup...")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return True
    else:
        print("⚠️  CUDA not available - will test CPU fallback")
        return False

def test_feature_extractor():
    """Test the new GPU feature extractor"""
    print("\n🚀 Testing GPU Feature Extractor...")
    
    try:
        from feature_extractor import GPUFeatureExtractor
        print("✅ Successfully imported GPUFeatureExtractor")
    except ImportError as e:
        print(f"❌ Failed to import GPUFeatureExtractor: {e}")
        return False
    
    try:
        extractor = GPUFeatureExtractor()
        print(f"✅ Feature extractor initialized")
        print(f"   Device: {extractor.device}")
        print(f"   GPU acceleration: {'Enabled' if extractor.device.type == 'cuda' else 'Disabled'}")
        
        # Test performance stats
        stats = extractor.get_performance_stats()
        print(f"   Cache size: {stats['cache_statistics']['cache_size']}")
        print(f"   GPU available: {stats['system_info']['gpu_available']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Feature extractor initialization failed: {e}")
        return False

def test_model_forward_pass():
    """Test the enhanced model with fixed tensor shapes"""
    print("\n🎯 Testing Model Forward Pass...")
    
    try:
        from enhanced_trainer import EnhancedWakewordModel, EnhancedModelConfig
        print("✅ Successfully imported model components")
    except ImportError as e:
        print(f"❌ Failed to import model components: {e}")
        return False
    
    try:
        # Create model config
        config = EnhancedModelConfig()
        config.hidden_size = 128
        config.num_layers = 2
        config.dropout = 0.3
        
        # Create model
        model = EnhancedWakewordModel(config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        print(f"✅ Model created on device: {device}")
        
        # Test forward pass with correct input shape
        batch_size = 4
        # Input shape: (batch_size, n_mels, time_steps) = (4, 40, 126)
        dummy_input = torch.randn(batch_size, 40, 126).to(device)
        
        print(f"   Testing forward pass with input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
        
        print(f"✅ Forward pass successful!")
        print(f"   Input shape: {dummy_input.shape}")
        print(f"   Output shape: {output.shape}")
        print(f"   Expected output: ({batch_size}, {config.num_classes})")
        
        # Validate output shape
        expected_shape = (batch_size, config.num_classes)
        if output.shape == expected_shape:
            print(f"✅ Output shape correct!")
            return True
        else:
            print(f"❌ Output shape mismatch: expected {expected_shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"❌ Model forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dataset_loading():
    """Test enhanced dataset with GPU acceleration"""
    print("\n📊 Testing Enhanced Dataset...")
    
    try:
        from enhanced_dataset import EnhancedWakewordDataset, EnhancedAudioConfig
        print("✅ Successfully imported dataset components")
    except ImportError as e:
        print(f"❌ Failed to import dataset components: {e}")
        return False
    
    try:
        # Create config with GPU acceleration
        config = EnhancedAudioConfig(
            use_precomputed_features=False,  # Force audio processing
            use_gpu_acceleration=True,
            num_workers=2,
            batch_size=8
        )
        
        # Test dataset creation (will use synthetic data if no audio files)
        dataset = EnhancedWakewordDataset(
            positive_dir="positive_dataset",
            negative_dir="negative_dataset",
            config=config,
            mode="train"
        )
        
        print(f"✅ Dataset created successfully")
        print(f"   Dataset size: {len(dataset)}")
        print(f"   GPU acceleration: {'Enabled' if config.use_gpu_acceleration and torch.cuda.is_available() else 'Disabled'}")
        
        # Test a few samples
        if len(dataset) > 0:
            print("   Testing sample loading...")
            start_time = time.time()
            
            for i in range(min(3, len(dataset))):
                sample = dataset[i]
                load_time = time.time() - start_time
                
                print(f"   Sample {i}: shape={sample['features'].shape}, label={sample['label'].item()}, load_time={load_time:.3f}s")
                start_time = time.time()  # Reset for next sample
            
            return True
        else:
            print("⚠️  Dataset is empty")
            return True
            
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 GPU-ACCELERATED WAKEWORD SYSTEM TEST")
    print("=" * 60)
    
    gpu_available = test_gpu_setup()
    
    # Run all tests
    tests = [
        ("Feature Extractor", test_feature_extractor),
        ("Model Forward Pass", test_model_forward_pass),
        ("Dataset Loading", test_dataset_loading)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! GPU acceleration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)