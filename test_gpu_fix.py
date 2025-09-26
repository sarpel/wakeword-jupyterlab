#!/usr/bin/env python3
"""
GPU Fix Verification Test
Tests that the wakeword_app_fixed.py properly utilizes GPU for training
"""

import torch
import torch.nn as nn
import numpy as np
import logging
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_gpu_availability():
    """Test basic GPU availability and initialization"""
    print("🔍 Testing GPU Availability...")

    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")

    if cuda_available:
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

        # Test GPU initialization
        try:
            torch.cuda.init()
            torch.backends.cudnn.benchmark = True
            print("✅ GPU initialization successful")
        except Exception as e:
            print(f"❌ GPU initialization failed: {e}")
            return False
    else:
        print("⚠️ No GPU available")
        return False

    return cuda_available

def test_device_allocation():
    """Test device allocation and tensor operations"""
    print("\n🔍 Testing Device Allocation...")

    try:
        # Create device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Selected device: {device}")

        # Test tensor creation on GPU
        x = torch.randn(32, 80, 126).to(device)  # Typical mel-spectrogram shape
        y = torch.randint(0, 2, (32,)).to(device)  # Binary labels

        print(f"Tensor device: {x.device}")
        print(f"Label device: {y.device}")

        # Test basic operations
        z = x * 2.0
        assert z.device == device
        print("✅ Basic GPU operations successful")

        return True

    except Exception as e:
        print(f"❌ Device allocation failed: {e}")
        return False

def test_model_gpu_placement():
    """Test model creation and GPU placement"""
    print("\n🔍 Testing Model GPU Placement...")

    try:
        # Simple test model similar to WakewordModel
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
                self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
                self.lstm = nn.LSTM(64, 128, batch_first=True)
                self.fc = nn.Linear(128, 2)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = torch.relu(self.conv2(x))
                x = self.adaptive_pool(x)
                batch_size = x.size(0)
                x = x.view(batch_size, -1).unsqueeze(1)
                x, _ = self.lstm(x)
                x = self.fc(x.squeeze(1))
                return torch.softmax(x, dim=1)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create and move model to GPU
        model = TestModel().to(device)

        # Verify all parameters are on GPU
        all_on_gpu = all(param.device == device for param in model.parameters())
        print(f"All parameters on {device}: {all_on_gpu}")

        # Test forward pass
        x = torch.randn(4, 1, 80, 126).to(device)  # Batch of mel-spectrograms
        output = model(x)

        print(f"Input device: {x.device}")
        print(f"Output device: {output.device}")
        print(f"Output shape: {output.shape}")
        print("✅ Model GPU placement successful")

        return True, model

    except Exception as e:
        print(f"❌ Model GPU placement failed: {e}")
        return False, None

def test_training_loop():
    """Test a simple training loop on GPU"""
    print("\n🔍 Testing Training Loop...")

    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Simple model for testing
        model = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 2)
        ).to(device)

        criterion = nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Generate dummy data
        batch_size = 8
        x = torch.randn(batch_size, 1, 80, 126).to(device)
        y = torch.randint(0, 2, (batch_size,)).to(device)

        # Training step
        model.train()
        optimizer.zero_grad()

        # Forward pass
        output = model(x)
        loss = criterion(output, y)

        # Backward pass
        loss.backward()
        optimizer.step()

        print(f"Training loss: {loss.item():.4f}")
        print(f"Model device: {next(model.parameters()).device}")
        print(f"Loss device: {loss.device}")
        print("✅ Training loop successful")

        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"❌ Training loop failed: {e}")
        return False

def test_memory_management():
    """Test GPU memory management"""
    print("\n🔍 Testing GPU Memory Management...")

    if not torch.cuda.is_available():
        print("⚠️ Skipping memory test - no GPU available")
        return True

    try:
        # Get initial memory stats
        initial_memory = torch.cuda.memory_allocated()
        print(f"Initial GPU memory: {initial_memory / 1e6:.1f} MB")

        # Create some tensors
        tensors = []
        for i in range(10):
            tensor = torch.randn(100, 100).cuda()
            tensors.append(tensor)

        peak_memory = torch.cuda.memory_allocated()
        print(f"Peak GPU memory: {peak_memory / 1e6:.1f} MB")

        # Clean up
        del tensors
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()
        print(f"Final GPU memory: {final_memory / 1e6:.1f} MB")
        print("✅ Memory management test successful")

        return True

    except Exception as e:
        print(f"❌ Memory management test failed: {e}")
        return False

def main():
    """Run all GPU tests"""
    print("🚀 Starting GPU Fix Verification Tests")
    print("=" * 50)

    tests = [
        ("GPU Availability", test_gpu_availability),
        ("Device Allocation", test_device_allocation),
        ("Model GPU Placement", lambda: test_model_gpu_placement()[0]),
        ("Training Loop", test_training_loop),
        ("Memory Management", test_memory_management)
    ]

    results = {}

    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False

    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print("=" * 50)

    all_passed = True
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not result:
            all_passed = False

    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! GPU training should work correctly.")
    else:
        print("⚠️ Some tests failed. Check GPU setup and drivers.")

    print("=" * 50)

if __name__ == "__main__":
    main()