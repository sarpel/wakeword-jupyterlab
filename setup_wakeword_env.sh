#!/bin/bash

# Wakeword Training Environment Setup Script
# Run this script in your WSL Ubuntu environment

set -e

echo "🚀 Setting up wakeword training environment..."

# Check if we're in WSL
if ! grep -q Microsoft /proc/version; then
    echo "❌ This script is designed for WSL Ubuntu environment"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "wakeword_env" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv wakeword_env
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

# Install PyTorch (CPU version for now - replace with CUDA version when available)
echo "🔥 Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install audio processing packages
echo "🎵 Installing audio processing packages..."
pip install librosa soundfile audioread numpy pandas matplotlib seaborn scikit-learn tqdm

# Install additional dependencies
echo "📦 Installing additional dependencies..."
pip install sympy filelock fsspec networkx

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

## For GPU Support (when CUDA is properly installed):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
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
echo ""
echo "🎯 Environment is ready for training!"