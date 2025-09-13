#!/bin/bash

# Wakeword Training Environment Setup Script
# Run this script in your WSL Ubuntu environment

set -e

echo "🚀 Setting up wakeword training environment..."

# Check if we're in WSL (multiple detection methods)
if ! (grep -q Microsoft /proc/version 2>/dev/null || grep -q WSL /proc/version 2>/dev/null || uname -r | grep -qi microsoft 2>/dev/null || [ -n "$WSL_DISTRO_NAME" ]); then
    echo "⚠️  Warning: This script is designed for WSL Ubuntu environment"
    echo "   Continuing anyway, but some features may not work as expected..."
    read -p "Press Enter to continue or Ctrl+C to abort: "
fi

# Check if python3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 is not installed. Please install Python3 first:"
    echo "   sudo apt update && sudo apt install python3 python3-pip python3-venv"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "wakeword_env" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv wakeword_env
    if [ $? -ne 0 ]; then
        echo "❌ Failed to create virtual environment. Make sure python3-venv is installed:"
        echo "   sudo apt install python3-venv"
        exit 1
    fi
fi

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source wakeword_env/bin/activate

# Upgrade pip
echo "📈 Upgrading pip..."
pip install --upgrade pip

# Install core packages
echo "🔧 Installing core packages..."
pip install jupyterlab jupyter

# Install PyTorch with GPU support (use latest available version)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install audio processing packages
echo "🎵 Installing audio processing packages..."
pip install librosa soundfile audioread numpy pandas matplotlib seaborn scikit-learn tqdm

# Install additional dependencies
echo "📦 Installing additional dependencies..."
pip install sympy filelock fsspec networkx

# Install Jupyter kernel support
echo "🔧 Installing Jupyter kernel support..."
pip install ipykernel

# Create Jupyter kernel for this environment
echo "🎯 Creating Jupyter kernel..."
python -m ipykernel install --user --name wakeword_env --display-name "Wakeword (GPU)"

# Create required directories
echo "📁 Creating data directories..."
mkdir -p wakeword_data negative_data background_noise

# Create a simple README
echo "📝 Creating setup guide..."
cat > SETUP_GUIDE.md << 'EOF'
# Wakeword Training Setup Guide

## Directories Created:
- `wakeword_data/` - Place your 500 wakeword recordings here
- `negative_data/` - Place your thousands of negative audio samples here  
- `background_noise/` - Place your 100 hours of background noise here

## To start JupyterLab:
```bash
source wakeword_env/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Then access from Windows at: `http://localhost:8888`

## Audio Format Support:
- WAV files (recommended)
- MP3 files
- FLAC files

## Next Steps:
1. Copy your audio files to the appropriate directories
2. Open `wakeword_training.ipynb` in JupyterLab
3. Run the cells to train your wakeword model

## GPU Support Status:
✅ GPU support is already installed with this script!
- Latest PyTorch with CUDA 11.8 support included
- To verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
EOF

echo "✅ Setup completed successfully!"
echo ""
echo "📁 Directories created:"
echo "   - wakeword_data/ (for your 500 wakeword recordings)"
echo "   - negative_data/ (for your negative audio samples)"  
echo "   - background_noise/ (for your 100 hours of background noise)"
echo ""
echo "📋 Next steps:"
echo "   1. Copy your audio files to the directories above"
echo "   2. Run: source wakeword_env/bin/activate"
echo "   3. Run: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser"
echo "   4. Open wakeword_training.ipynb in your browser"
echo "   5. Select 'Wakeword (GPU)' kernel from the notebook dropdown"
echo ""
echo "🎯 Environment is ready for training!"
echo ""
echo "🔍 GPU Verification:"
echo "   Run the following command to verify GPU support:"
echo "   source wakeword_env/bin/activate"
echo "   python -c \"import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')\""
echo ""
echo "   If GPU shows as False, you may need to:"
echo "   1. Install NVIDIA drivers: sudo apt install nvidia-driver-535"
echo "   2. Check CUDA installation: nvidia-smi"
echo "   3. Restart WSL: wsl --shutdown in Windows"