#!/usr/bin/env python3
"""
Quick GPU Detection and CUDA Check
"""
import torch
import sys

print("🔍 GPU Detection Report")
print("=" * 40)

# Check PyTorch version
print(f"PyTorch Version: {torch.__version__}")

# Check CUDA availability
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")

if cuda_available:
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"CUDA Device Count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")

    # Test tensor creation
    try:
        x = torch.tensor([1.0]).cuda()
        print("✅ GPU tensor creation successful")
        print(f"Test tensor device: {x.device}")
    except Exception as e:
        print(f"❌ GPU tensor creation failed: {e}")
else:
    print("❌ No GPU detected")
    print("\n🔍 Possible issues:")
    print("1. No NVIDIA GPU installed")
    print("2. NVIDIA drivers not installed")
    print("3. CUDA toolkit not installed")
    print("4. PyTorch CPU-only version installed")

    print("\n💡 Solutions:")
    print("1. Install NVIDIA GPU drivers")
    print("2. Install CUDA toolkit")
    print("3. Install PyTorch with CUDA support:")
    print("   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")

print("\n" + "=" * 40)