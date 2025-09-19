#!/usr/bin/env python3
"""
Gradio Application Launcher
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages for Gradio app"""
    print("📦 Installing required packages...")

    try:
        # Install from requirements file
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements_gradio.txt"])
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing packages: {e}")
        return False

def check_environment():
    """Check if the environment is properly set up"""
    print("🔍 Checking environment...")

    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Check Gradio
        import gradio as gr
        print(f"✅ Gradio version: {gr.__version__}")

        # Check other required packages
        import librosa, soundfile, numpy, sklearn, matplotlib, plotly
        print("✅ All required packages are available")

        return True

    except ImportError as e:
        print(f"❌ Missing package: {e}")
        return False

def launch_app():
    """Launch the Gradio application"""
    print("🚀 Launching Wakeword Training Gradio Application...")

    try:
        # Change to the directory containing the app
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

        # Launch the app
        subprocess.run([sys.executable, "wakeword_training_gradio.py"])

    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error launching application: {e}")

def main():
    print("🎯 Wakeword Training Gradio Application Launcher")
    print("=" * 50)

    # Check environment
    if not check_environment():
        print("\n⚠️  Environment check failed!")
        print("Please install required packages:")
        print("pip install -r requirements_gradio.txt")
        return

    # Ask user if they want to install requirements
    install_choice = input("\n📦 Do you want to install/update required packages? (y/N): ").lower().strip()

    if install_choice == 'y':
        if not install_requirements():
            print("❌ Failed to install packages. Please install manually:")
            print("pip install -r requirements_gradio.txt")
            return

    print("\n🚀 Starting application...")
    print("The application will open in your default web browser.")
    print("Press Ctrl+C to stop the application.")

    launch_app()

if __name__ == "__main__":
    main()