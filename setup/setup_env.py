#!/usr/bin/env python3
"""
Gradio GPU Environment Setup Script
This script will set up a proper virtual environment with CUDA-enabled PyTorch
"""

import subprocess
import sys
import os
import time

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

    # Remove existing environment if it exists
    import shutil
    if os.path.exists('gradio_venv_gpu'):
        print("🗑️ Removing existing virtual environment...")
        shutil.rmtree('gradio_venv_gpu')
        print("✅ Existing venv removed")
    # Create new environment
    if not run_command('python -m venv gradio_venv_gpu', "Creating virtual environment"):
        return False

    # Get the python executable path
    if os.name == 'nt':  # Windows
        python_exe = 'gradio_venv_gpu\\Scripts\\python.exe'
        pip_exe = 'gradio_venv_gpu\\Scripts\\pip.exe'
    else:  # Linux/Mac
        python_exe = 'gradio_venv_gpu/bin/python'
        pip_exe = 'gradio_venv_gpu/bin/pip'

    return python_exe, pip_exe

def install_packages(pip_exe):
    """Install required packages"""
    print("\n📦 Installing packages...")

    # Upgrade pip first
    if not run_command(f'{pip_exe} install --upgrade pip', "Upgrading pip"):
        return False

    # Install PyTorch with CUDA support
    print("\n🔥 Installing PyTorch with CUDA support...")
    cmd = f'{pip_exe} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118'
    if not run_command(cmd, "Installing PyTorch with CUDA"):
        print("⚠️ PyTorch installation failed, trying alternative...")
        # Try with CUDA 11.7
        cmd = f'{pip_exe} install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117'
        if not run_command(cmd, "Installing PyTorch with CUDA 11.7"):
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

def create_launcher():
    """Create a launcher script for the Gradio app"""
    print("\n🚀 Creating launcher script...")

    launcher_content = '''#!/usr/bin/env python3
"""
Gradio Application Launcher for GPU Environment
"""

import subprocess
import sys
import os

def activate_and_run():
    """Activate virtual environment and run Gradio app"""

    # Get the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine the activation command based on OS
    if os.name == 'nt':  # Windows
        activate_cmd = 'gradio_venv_gpu\\Scripts\\activate'
        python_exe = 'gradio_venv_gpu\\Scripts\\python.exe'
    else:  # Linux/Mac
        activate_cmd = 'source gradio_venv_gpu/bin/activate'
        python_exe = 'gradio_venv_gpu/bin/python'

    # Change to script directory
    os.chdir(script_dir)

    # Run the Gradio app
    app_path = os.path.join(script_dir, 'wakeword_training_gradio.py')

    print("🚀 Starting Wakeword Training Gradio Application...")
    print("📍 App location:", app_path)
    print("🌐 The application will open in your web browser")
    print("⏹️ Press Ctrl+C to stop the application")
    print("=" * 60)

    try:
        subprocess.run([python_exe, app_path])
    except KeyboardInterrupt:
        print("\\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

if __name__ == "__main__":
    activate_and_run()
'''

    with open('launch_gradio_gpu.py', 'w') as f:
        f.write(launcher_content)

    print("✅ Launcher script created: launch_gradio_gpu.py")

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
    create_launcher()

    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Run the application: python launch_gradio_gpu.py")
    print("2. Open your web browser to the shown URL")
    print("3. Configure your data paths and start training")
    print("\n🔧 Environment location: gradio_venv_gpu/")
    print("🚀 Launcher script: launch_gradio_gpu.py")

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
