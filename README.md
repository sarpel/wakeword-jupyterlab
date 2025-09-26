# 🎯 Enhanced Wakeword Training Studio

**Advanced wakeword detection training system with automated dataset management
and real-time monitoring**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ New Enhanced Features](#-new-enhanced-features)
- [🚀 Quick Start](#-quick-start)
- [📊 System Requirements](#-system-requirements)
- [🔧 Installation](#-installation)
- [📁 Project Structure](#-project-structure)
- [🤖 Automated Dataset Management](#-automated-dataset-management)
- [🧠 Model Architecture](#-model-architecture)
- [🏋️ Training Process](#️-training-process)
- [📊 Evaluation](#-evaluation)
- [🚀 Deployment](#-deployment)
- [🔍 Troubleshooting](#-troubleshooting)
- [📚 Advanced Features](#-advanced-features)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Overview

**Enhanced Wakeword Training Studio** is a comprehensive, production-ready
system for training custom wakeword detection models. It combines
state-of-the-art deep learning techniques with an intuitive web interface and
advanced automated dataset management, making wakeword training accessible to
both beginners and experts.

### Key Highlights:

- **🤖 Automated Dataset Management**: One-click dataset structure creation,
  auto-detection, and intelligent splitting
- **🎨 Single-file architecture**: All functionality consolidated into one clean
  Python file ([`wakeword_app.py`](wakeword_app.py))
- **🖥️ Web-based interface**: No coding required, everything accessible through
  browser
- **⚡ GPU acceleration**: Automatic GPU detection and optimization with live
  memory monitoring
- **📊 Real-time monitoring**: Live training progress with batch-level updates
  and performance metrics
- **🔧 Advanced features**: Data augmentation, early stopping, model evaluation,
  feature caching
- **📱 Production-ready**: Export models for deployment in real applications

## ✨ New Enhanced Features

### 🤖 Automated Dataset Management

- **One-click structure creation**: Automatically creates complete dataset
  folder structure
- **Smart file detection**: Scans and detects audio files across all categories
- **Intelligent auto-splitting**: Automatically splits datasets into
  train/validation/test (70/20/10)
- **Real-time validation**: Validates file counts and provides recommendations
- **Comprehensive reporting**: Detailed dataset statistics and health reports

### 🔥 Enhanced Training System

- **Batch-level monitoring**: Real-time updates for every training batch
- **GPU memory tracking**: Live GPU memory usage monitoring
- **Advanced error handling**: Fixed pickle serialization issues for Windows
- **Enhanced progress tracking**: Detailed epoch and batch progress with
  accuracy metrics
- **Automatic checkpointing**: Best model preservation with comprehensive
  metadata

### 🎯 Advanced Prediction & Testing

- **Dual prediction modes**: File upload and live microphone recording
- **Detailed probability analysis**: Shows confidence scores for both classes
- **Feature visualization**: Displays extracted mel-spectrogram features
- **Real-time audio processing**: Live recording and immediate prediction

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install torch torchaudio gradio librosa soundfile numpy scikit-learn matplotlib seaborn pandas plotly tqdm
```

### 2. Launch the Application

```bash
# Start the enhanced training studio
python wakeword_app.py
```

### 3. Automated Dataset Setup

1. Open your browser to `http://localhost:7860`
2. Go to **Dataset Management** tab
3. Click **"Create Dataset Structure"** to create folders
4. Add your audio files to the appropriate folders
5. Click **"Detect Dataset Status"** to check readiness
6. Click **"Auto-Split Dataset"** to organize files automatically
7. Start training in the **Model Training** tab

## 📊 System Requirements

### Minimum Requirements

- **Python**: 3.8 or higher
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Audio**: Microphone for recording (optional)

### Recommended Requirements

- **GPU**: NVIDIA GPU with CUDA support
- **RAM**: 16GB or more
- **Storage**: 10GB+ for large datasets
- **CPU**: Multi-core processor

### Supported Platforms

- **Windows**: 10/11
- **Linux**: Ubuntu 18.04+, CentOS 7+
- **macOS**: 10.14+

## 🔧 Installation

### Option 1: Manual Installation

```bash
# Clone or download the project
git clone <repository-url>
cd enhanced-wakeword-training-studio

# Install Python dependencies
pip install -r requirements.txt

# Launch application
python wakeword_app.py
```

### Option 2: Using conda

```bash
# Create conda environment
conda create -n wakeword python=3.9
conda activate wakeword

# Install dependencies
conda install pytorch torchaudio -c pytorch
pip install gradio librosa soundfile scikit-learn matplotlib seaborn pandas plotly tqdm
```

## 📁 Project Structure

```
enhanced-wakeword-training-studio/
├── wakeword_app.py              # 🎯 Main application (single file)
├── requirements.txt             # 📦 Dependencies
├── README.md                    # 📚 This documentation
├── GEMINI.md                    # 🔍 AI analysis report
│
├── data/                        # 📊 Dataset organization
│   ├── positive/               # ✅ Wakeword recordings
│   │   ├── train/              #   Training samples (70%)
│   │   ├── validation/         #   Validation samples (20%)
│   │   └── test/               #   Test samples (10%)
│   ├── negative/               # ❌ Negative samples
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── hard_negative/          # ⚠️ Hard negative samples (optional)
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── background/             # 🔊 Background noise
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── rirs/                   # 🏠 Room Impulse Responses (optional)
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── features/               # 🔍 Pre-extracted features (optional)
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── raw/                    # 📁 Original audio files
│   └── processed/              # ⚙️ Preprocessed audio
```

## 🤖 Automated Dataset Management

### Minimum Requirements

| Dataset Type      | Minimum Files | Recommended | Quality Requirements        |
| ----------------- | ------------- | ----------- | --------------------------- |
| **Positive**      | 100           | 500-1000    | Clean wakeword recordings   |
| **Negative**      | 450           | 2000-4000   | Random speech, no wakewords |
| **Background**    | 1000          | 5000+       | Environmental noise         |
| **Hard Negative** | 50            | 200-500     | Similar to wakeword         |

### Automated Workflow

1. **Structure Creation**: One-click creation of complete folder hierarchy
2. **File Detection**: Automatic scanning and counting of audio files
3. **Smart Splitting**: Intelligent 70/20/10 train/validation/test split
4. **Validation**: Real-time validation of dataset readiness
5. **Reporting**: Comprehensive statistics and health reports

### Audio Specifications

- **Format**: WAV, MP3, FLAC, M4A, OGG
- **Sample Rate**: 16kHz (automatically resampled)
- **Channels**: Mono (automatically converted)
- **Duration**: 1-3 seconds
- **Quality**: Clean, no clipping, minimal background noise

## 🧠 Model Architecture

### Enhanced CNN Architecture

```
Input (64×63 mel-spectrogram)
    ↓
Conv2D (32 filters, 3×3) + BatchNorm + ReLU + MaxPool + Dropout
    ↓
Conv2D (64 filters, 3×3) + BatchNorm + ReLU + MaxPool + Dropout
    ↓
Conv2D (128 filters, 3×3) + BatchNorm + ReLU + MaxPool + Dropout
    ↓
Flatten
    ↓
Dense (256) + ReLU + Dropout
    ↓
Dense (128) + ReLU + Dropout
    ↓
Dense (2 classes)
    ↓
Softmax Output
```

### Key Features:

- **CNN Layers**: Extract spatial features from mel-spectrograms
- **Batch Normalization**: Improved training stability
- **Dropout Regularization**: Prevents overfitting
- **Binary Classification**: Wakeword vs. background classification
- **GPU Acceleration**: Full CUDA support for training and inference

### Model Parameters: ~882K parameters

## 🏋️ Training Process

### Enhanced Training Pipeline

1. **Data Loading**: Automatic audio loading and preprocessing
2. **Feature Extraction**: GPU-accelerated mel-spectrogram computation
3. **Data Augmentation**: Real-time augmentation during training
4. **Model Training**: Enhanced CNN training with backpropagation
5. **Validation**: Continuous validation with detailed metrics
6. **Early Stopping**: Automatic stopping when validation plateaus
7. **Model Saving**: Best model checkpoint with comprehensive metadata

### Real-time Monitoring Features

- **Batch-level Progress**: Live updates for every training batch
- **Loss Curves**: Real-time training/validation loss visualization
- **Accuracy Tracking**: Training and validation accuracy monitoring
- **GPU Memory**: Live GPU memory usage tracking
- **Learning Rate**: Adaptive learning rate scheduling
- **Time Tracking**: Total training time and estimated completion

## 📊 Evaluation

### Performance Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Wakeword detection precision
- **Recall**: Wakeword detection recall rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

### Model Quality Assessment

| F1-Score  | Quality Level | Recommendation            |
| --------- | ------------- | ------------------------- |
| ≥ 0.90    | Excellent     | Ready for production      |
| 0.80-0.89 | Very Good     | Good for deployment       |
| 0.70-0.79 | Good          | Consider more data        |
| 0.60-0.69 | Average       | Review data quality       |
| < 0.60    | Poor          | Major improvements needed |

## 🚀 Deployment

### Model Export Options

1. **PyTorch Model**: Complete model with training configuration
2. **ONNX Format**: Cross-platform deployment format
3. **TorchScript**: Optimized for production inference

### Deployment Package Contents

- **Model Weights**: Trained neural network parameters
- **Configuration**: Audio processing and model parameters
- **Preprocessing Pipeline**: Audio feature extraction code
- **Usage Examples**: Sample inference code
- **Comprehensive Documentation**: Deployment and integration guide

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Dataset Not Detected

- **Check file formats**: Ensure supported audio formats (.wav, .mp3, .npy)
- **Verify folder structure**: Use automated structure creation
- **File permissions**: Ensure read access to audio files
- **Auto-detection**: Click "Detect Dataset Status" button

#### 3. Training Not Converging

- **Increase dataset size**: More training data often helps
- **Adjust learning rate**: Try lower learning rates (0.0001-0.001)
- **Check data balance**: Ensure adequate positive/negative samples
- **Verify data quality**: Ensure clean, consistent audio

#### 4. Auto-splitting Errors

- **Minimum files**: Ensure categories meet minimum requirements
- **File organization**: Place files directly in category folders
- **Disk space**: Ensure adequate storage for file operations
- **Permissions**: Check write permissions for data directories

## 📚 Advanced Features

### 🤖 Automated Dataset Management

- **Smart Detection**: Automatic file discovery and counting
- **Intelligent Splitting**: Optimal train/validation/test distribution
- **Health Monitoring**: Comprehensive dataset validation
- **Error Recovery**: Robust error handling and reporting

### 🚀 GPU Acceleration

- **Automatic Detection**: Automatic CUDA GPU detection
- **Memory Management**: Intelligent GPU memory management
- **Mixed Precision**: FP16 training for faster convergence
- **Multi-GPU Support**: Distributed training capabilities

### 📊 Advanced Monitoring

- **Real-time Updates**: Live training progress with 2-second refresh
- **Batch-level Tracking**: Detailed batch-by-batch monitoring
- **Performance Metrics**: Comprehensive training statistics
- **Export Capabilities**: Model information and training history export

## 🤝 Contributing

We welcome contributions to improve Enhanced Wakeword Training Studio!

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Add comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed
- Ensure backward compatibility

### Areas for Contribution

- **Model Architectures**: New neural network designs
- **Dataset Management**: Enhanced automation features
- **Training Strategies**: Advanced training methods
- **User Interface**: Enhanced web interface features
- **Performance Optimization**: Speed and memory improvements
- **Documentation**: Improved guides and tutorials

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

---

## 🎯 Summary

**Enhanced Wakeword Training Studio** provides a complete, production-ready
solution for training custom wakeword detection models. With its automated
dataset management, real-time monitoring, advanced features, and comprehensive
documentation, it democratizes wakeword training technology for developers,
researchers, and enthusiasts.

### Key Benefits:

- **🚀 Fast Setup**: Get started in minutes with automated dataset management
- **🤖 Intelligent**: Automated dataset organization and validation
- **📊 Comprehensive**: Real-time monitoring and detailed reporting
- **⚡ High Performance**: GPU-accelerated training and inference
- **🔧 User-Friendly**: Intuitive interface with advanced features
- **📱 Production-Ready**: Complete deployment pipeline

### Next Steps:

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Launch application**: `python wakeword_app.py`
3. **Create dataset structure**: Use automated dataset management
4. **Add your data**: Upload audio files to appropriate folders
5. **Auto-split datasets**: Let the system organize your data
6. **Start training**: Monitor progress in real-time
7. **Deploy**: Export your trained model for production use

---

**Happy Wakeword Training!** 🎉

For support and questions, please check the troubleshooting section or create an
issue in the repository.
