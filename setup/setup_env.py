#!/usr/bin/env python3
"""
Gradio GPU Environment Setup Script
This script will set up a proper virtual environment with CUDA-enabled PyTorch
"""

import subprocess
import sys
import os
import time

VENV_DIR = 'venv'
LEGACY_VENV_DIR = 'gradio_venv_gpu'

def run_command(cmd, description=""):
    """Run a command and return success status"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print(f"Output: {result.stdout}")
            return True
        else:
            print(f"❌ {description} failed")
            print(f"Error: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def check_cuda():
    """Check if CUDA is available on the system"""
    print("\n🔍 Checking CUDA availability...")

    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ NVIDIA GPU detected")
            print("GPU Info:")
            print(result.stdout)
            return True
        else:
            print("❌ NVIDIA GPU not found or nvidia-smi failed")
            return False
    except FileNotFoundError:
        print("❌ nvidia-smi not found - CUDA not installed")
        return False
    except Exception as e:
        print(f"❌ Error checking CUDA: {e}")
        return False

def create_virtual_environment():
    """Create and set up virtual environment"""
    print("\n🏗️ Creating virtual environment...")

    import shutil
    for candidate in (VENV_DIR, LEGACY_VENV_DIR):
        if os.path.exists(candidate):
            print(f"🗑️ Removing existing virtual environment: {candidate}")
            shutil.rmtree(candidate)
            print("✅ Existing venv removed")

    if not run_command(f'python -m venv {VENV_DIR}', "Creating virtual environment"):
        return False

    if os.name == 'nt':  # Windows
        python_exe = os.path.join(VENV_DIR, 'Scripts', 'python.exe')
        pip_exe = os.path.join(VENV_DIR, 'Scripts', 'pip.exe')
    else:  # Linux/Mac
        python_exe = os.path.join(VENV_DIR, 'bin', 'python')
        pip_exe = os.path.join(VENV_DIR, 'bin', 'pip')

    return python_exe, pip_exe

def install_packages(pip_exe):
    """Install required packages"""
    print("\n📦 Installing packages...")

    # Upgrade pip first
    if not run_command(f'{pip_exe} install --upgrade pip', "Upgrading pip"):
        return False


    # Install PyTorch with CUDA support per policy: prefer 2.1.2 cu118, fallback cu121
    print("\n🔥 Installing PyTorch (2.1.2) with CUDA 11.8...")
    cmd = f'{pip_exe} install torch==2.1.2+cu118 torchvision==0.16.2+cu118 torchaudio==2.1.2+cu118 --index-url https://download.pytorch.org/whl/cu118'
    if not run_command(cmd, "Installing PyTorch 2.1.2 cu118"):
        print("⚠️ cu118 installation failed, trying CUDA 12.1...")
        cmd = f'{pip_exe} install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --index-url https://download.pytorch.org/whl/cu121'
        if not run_command(cmd, "Installing PyTorch 2.1.2 cu121"):

            return False

    # Install other required packages
    packages = [
        'gradio>=4.0.0',
        'librosa>=0.10.0',
        'soundfile>=0.12.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'pandas>=1.3.0',
        'plotly>=5.0.0',
        'tqdm>=4.62.0',
        'numpy>=1.21.0'
    ]

    for package in packages:
        if not run_command(f'{pip_exe} install {package}', f"Installing {package}"):
            print(f"⚠️ Failed to install {package}, continuing...")

    return True

def test_installation(python_exe):
    """Test if the installation was successful"""
    print("\n🧪 Testing installation...")

    test_commands = [
        ('import torch; print(f"PyTorch: {torch.__version__}")', 'PyTorch import'),
        ('import torch; print(f"CUDA available: {torch.cuda.is_available()}")', 'CUDA availability'),
        ('import torch; print(f"GPU count: {torch.cuda.device_count()}")', 'GPU count'),
        ('import gradio; print("Gradio: OK")', 'Gradio import'),
        ('import librosa; print("Librosa: OK")', 'Librosa import'),
        ('import soundfile; print("Soundfile: OK")', 'Soundfile import'),
        ('import sklearn; print("Scikit-learn: OK")', 'Scikit-learn import'),
    ]

    results = []
    for cmd, desc in test_commands:
        try:
            result = subprocess.run([python_exe, '-c', cmd], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print(f"✅ {desc}: {result.stdout.strip()}")
                results.append(True)
            else:
                print(f"❌ {desc}: Failed")
                print(f"Error: {result.stderr}")
                results.append(False)
        except Exception as e:
            print(f"❌ {desc}: Exception - {e}")
            results.append(False)

    return all(results)


def create_launcher(python_exe):
    """Remind the user how to start the Gradio application using the venv."""
    print("\nLauncher script already provided: launch_app.py")
    print("Use the virtualenv Python to run the app for correct CUDA build:")
    print(f"  {python_exe} launch_app.py")


def main():
    print("🎯 Gradio GPU Environment Setup")
    print("=" * 50)

    # Check CUDA availability
    if not check_cuda():
        print("❌ CUDA not available. Please install CUDA and NVIDIA drivers.")
        print("Visit: https://developer.nvidia.com/cuda-downloads")
        return False

    # Create virtual environment
    venv_result = create_virtual_environment()
    if not venv_result:
        print("❌ Failed to create virtual environment")
        return False

    python_exe, pip_exe = venv_result

    # Install packages
    if not install_packages(pip_exe):
        print("❌ Failed to install packages")
        return False

    # Test installation
    if not test_installation(python_exe):
        print("❌ Installation test failed")
        return False

    # Create launcher

    create_launcher(python_exe)


    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the application: python launch_app.py")
    print("2. Open your web browser to the shown URL")
    print("3. Configure your data paths and start training")
    print("\n🔧 Environment location: venv/")
    print("?? Launcher script: launch_app.py")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
