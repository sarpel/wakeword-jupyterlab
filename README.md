# Wakeword Detection Training System

A complete GPU-accelerated wakeword detection training system for Ubuntu WSL with NVIDIA RTX 3060. This system includes audio processing, data augmentation, CNN+LSTM neural networks, and end-to-end training pipeline.

## 🚀 Quick Start

**System Status: ✅ FULLY OPERATIONAL**

```bash
# Activate environment and start JupyterLab
source wakeword_env/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Access JupyterLab: `http://127.0.0.1:8888/lab` (use token from terminal)

## 📋 Prerequisites

### Hardware Requirements
- **GPU**: NVIDIA RTX 3060 Ti (8GB VRAM) or similar
- **Memory**: 16GB+ RAM recommended
- **Storage**: 10GB+ free space for data and models

### Software Requirements
- **Windows 11** with WSL2 Ubuntu
- **NVIDIA Drivers** with CUDA support
- **Python 3.10** (automatically installed)

## 🔧 Installation

### Method 1: Automated Setup (Recommended)

1. **Download and run the setup script:**
   ```bash
   chmod +x setup_wakeword_env.sh
   ./setup_wakeword_env.sh
   ```

2. **Activate environment:**
   ```bash
   source wakeword_env/bin/activate
   ```

### Method 2: Manual Installation

1. **Create virtual environment:**
   ```bash
   python3 -m venv wakeword_env
   source wakeword_env/bin/activate
   ```

2. **Upgrade pip:**
   ```bash
   pip install --upgrade pip
   ```

3. **Install PyTorch with CUDA support:**
   ```bash
   pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
   ```

4. **Install audio processing packages:**
   ```bash
   pip install librosa soundfile audioread numpy pandas matplotlib seaborn scikit-learn tqdm
   ```

5. **Install JupyterLab:**
   ```bash
   pip install jupyterlab jupyter
   ```

6. **Install additional dependencies:**
   ```bash
   pip install sympy filelock fsspec networkx
   ```

### Verify Installation

```bash
source wakeword_env/bin/activate
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
GPU Available: True
Device: NVIDIA GeForce RTX 3060 Ti
```

## 📁 Data Organization

Create the following directory structure:

```
wakeword_project/
├── wakeword_data/           # Your 500 wakeword recordings
├── negative_data/           # Thousands of negative audio samples  
├── background_noise/        # 100 hours of background noise
├── wakeword_training.ipynb  # Main training notebook
├── test_gpu_training.py     # GPU verification script
└── wakeword_env/           # Virtual environment
```

### Data Requirements

**Wakeword Data (`wakeword_data/`)**
- **Quantity**: 500+ recordings of your target wakeword
- **Format**: WAV, MP3, or FLAC files
- **Quality**: Clear audio, minimal background noise
- **Recording**: Multiple speakers recommended (you and your wife)

**Negative Data (`negative_data/`)**
- **Quantity**: Thousands of non-wakeword audio samples
- **Content**: Speech, ambient sounds, other words
- **Format**: WAV, MP3, or FLAC files
- **Duration**: Variable lengths, 1-10 seconds recommended

**Background Noise (`background_noise/`)**
- **Quantity**: 100+ hours of background audio
- **Content**: Home environment noise, street sounds, etc.
- **Format**: WAV, MP3, or FLAC files
- **Purpose**: Data augmentation and training robustness

## 🧠 Training Pipeline

### Step 1: Start JupyterLab

```bash
source wakeword_env/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --no-browser
```

Access JupyterLab from Windows browser at: `http://localhost:8888`

### Step 2: Open Training Notebook

Open `wakeword_training.ipynb` in JupyterLab and run cells sequentially.

### Step 3: Configure Training Parameters

In the notebook, adjust these parameters as needed:

```python
# Audio parameters
SAMPLE_RATE = 16000
DURATION = 1.0  # seconds

# Model parameters  
N_MELS = 80
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 50

# Data augmentation
AUGMENTATION_PROB = 0.5
NOISE_FACTOR = 0.1
```

### Step 4: Run Training Pipeline

The notebook automatically:
1. **Processes Audio**: Converts all audio to mel-spectrograms
2. **Applies Augmentation**: Adds noise, time-shifting, pitch-shifting
3. **Creates Datasets**: Builds train/validation/test splits
4. **Trains Model**: GPU-accelerated CNN+LSTM training
5. **Evaluates Performance**: Shows accuracy, loss curves, confusion matrix
6. **Saves Model**: Exports trained model for deployment

## 🎯 Model Architecture

### CNN+LSTM Hybrid Network

```
Input: Mel-spectrogram (80x160)
│
├─ Conv2D (1→32 channels) + ReLU
├─ Conv2D (32→64 channels) + ReLU  
├─ Conv2D (64→128 channels) + ReLU
├─ AdaptiveAvgPool2D → (1, 1)
│
├─ LSTM (128→hidden_size, 2 layers)
├─ Dropout (0.5)
│
└─ Linear → Softmax (Binary classification)
```

### Key Features
- **Convolutional Layers**: Extract spectral patterns
- **LSTM Layers**: Model temporal dependencies
- **Dropout**: Prevent overfitting
- **GPU Acceleration**: CUDA-optimized training
- **Real-time Ready**: Optimized for deployment

## 📊 Training Features

### Data Augmentation
- **Background Noise Mixing**: Blend with background sounds
- **Time Shifting**: Random temporal shifts
- **Pitch Shifting**: Vary voice pitch ±2 semitones
- **Speed Changes**: Adjust playback speed (0.8-1.2x)
- **Volume Normalization**: Consistent audio levels

### Performance Monitoring
- **Real-time Training**: Live loss and accuracy tracking
- **GPU Memory Monitoring**: VRAM usage display
- **Confusion Matrix**: Detailed classification results
- **ROC Curves**: Model performance visualization
- **Learning Curves**: Training validation over epochs

### Model Evaluation
- **Accuracy**: Overall classification performance
- **Precision/Recall**: Detailed metrics
- **F1-Score**: Balanced performance measure
- **False Positive Rate**: Critical for wakeword detection
- **Inference Speed**: Real-time capability assessment

## 🔧 Advanced Usage

### Custom Wakeword Training

1. **Modify target classes** in the notebook:
   ```python
   TARGET_WORDS = ['your_wakeword', 'negative']
   ```

2. **Adjust model architecture**:
   ```python
   model_config = {
       'N_MELS': 80,
       'HIDDEN_SIZE': 256,  # Increase for complex wakewords
       'NUM_LAYERS': 3,     # Deeper LSTM
       'DROPOUT': 0.3       # Adjust regularization
   }
   ```

### Hyperparameter Tuning

Key parameters to optimize:
- **Learning Rate**: 0.0001 - 0.01
- **Batch Size**: 16 - 128 (GPU memory dependent)
- **Hidden Size**: 64 - 512
- **Number of Layers**: 1 - 4
- **Dropout Rate**: 0.2 - 0.7

### Multi-GPU Training (If Available)

```python
# Enable multi-GPU training
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

## 🚀 Deployment

### Export Trained Model

```python
# Save model for deployment
torch.save({
    'model_state_dict': model.state_dict(),
    'config': model_config,
    'audio_config': audio_config
}, 'wakeword_model.pth')
```

### Real-time Inference

```python
import torch
import librosa

def detect_wakeword(audio_file, model, threshold=0.8):
    # Load and process audio
    audio, sr = librosa.load(audio_file, sr=16000)
    mel_spec = audio_processor.audio_to_mel(audio)
    
    # Run inference
    with torch.no_grad():
        output = model(mel_spec.unsqueeze(0))
        probability = torch.softmax(output, dim=1)[0][1]
    
    return probability.item() > threshold
```

## 🐛 Troubleshooting

### GPU Issues

**Problem**: `CUDA not available`
```bash
# Check CUDA installation
nvidia-smi

# Verify PyTorch CUDA support
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**Problem**: Out of memory errors
- Reduce `BATCH_SIZE` in notebook
- Close other GPU applications
- Use smaller audio files

### Audio Processing Issues

**Problem**: Audio file not supported
```bash
# Install additional audio codecs
pip install ffmpeg-python
sudo apt install ffmpeg
```

**Problem**: Audio loading errors
- Verify file permissions
- Check audio file integrity
- Convert to WAV format if needed

### JupyterLab Issues

**Problem**: JupyterLab won't start
```bash
# Clear Jupyter configuration
jupyter lab --generate-config
jupyter lab clean
```

**Problem**: Can't access from Windows
- Check WSL networking: `ip addr show eth0`
- Use correct IP: `jupyter lab --ip=0.0.0.0`
- Check Windows Firewall settings

## 📈 Performance Optimization

### GPU Memory Management
```python
# Monitor GPU memory
print(f"Memory allocated: {torch.cuda.memory_allocated() / 1e6:.2f} MB")
print(f"Memory cached: {torch.cuda.memory_reserved() / 1e6:.2f} MB")

# Clear cache
torch.cuda.empty_cache()
```

### Training Speed Optimization
- Use mixed precision training: `torch.cuda.amp`
- Increase batch size to GPU limit
- Use data loading with multiple workers
- Enable CUDA graphs for repeated operations

### Model Optimization
- Quantize model for deployment: `torch.quantization`
- Use TorchScript for optimization
- Apply pruning for smaller model size

## 🔄 Version Compatibility

### Tested Configurations
- **OS**: Ubuntu 22.04 LTS (WSL2)
- **GPU**: NVIDIA RTX 3060 Ti (8GB)
- **CUDA**: 11.8
- **PyTorch**: 2.0.1
- **Python**: 3.10

### Dependency Versions
```
torch==2.0.1+cu118
torchvision==0.15.2+cu118
torchaudio==2.0.2+cu118
librosa==0.10.1
soundfile==0.12.1
numpy==1.24.3
pandas==2.0.3
jupyterlab==4.0.5
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **PyTorch Team** for the deep learning framework
- **Librosa** for audio processing capabilities
- **NVIDIA** for GPU computing platforms
- **WSL Team** for Windows Subsystem for Linux

## 📞 Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the Jupyter notebook documentation
3. Test GPU functionality with `test_gpu_training.py`
4. Verify data file formats and permissions

**System Status**: ✅ Ready for wakeword training with GPU acceleration!

---

*Last Updated: September 2025*  
*Version: 1.0.0*  
*GPU Accelerated: Yes*