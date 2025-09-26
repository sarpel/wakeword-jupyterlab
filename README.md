# 🎯 Wakeword Training Studio

**Complete wakeword detection training system with advanced Gradio interface**

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/Gradio-4.0+-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📊 System Requirements](#-system-requirements)
- [🔧 Installation](#-installation)
- [📁 Project Structure](#-project-structure)
- [🎛️ Configuration](#️-configuration)
- [📈 Data Preparation](#-data-preparation)
- [🧠 Model Architecture](#-model-architecture)
- [🏋️ Training Process](#️-training-process)
- [📊 Evaluation](#-evaluation)
- [🚀 Deployment](#-deployment)
- [🔍 Troubleshooting](#-troubleshooting)
- [📚 Advanced Features](#-advanced-features)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Overview

**Wakeword Training Studio** is a comprehensive, user-friendly system for
training custom wakeword detection models. It combines state-of-the-art deep
learning techniques with an intuitive web interface, making wakeword training
accessible to both beginners and experts.

### Key Highlights:

- **🎨 Single-file architecture** - All functionality consolidated into one
  clean Python file
- **🖥️ Web-based interface** - No coding required, everything accessible through
  browser
- **⚡ GPU acceleration** - Automatic GPU detection and optimization
- **📊 Real-time monitoring** - Live training progress and performance metrics
- **🔧 Advanced features** - Data augmentation, early stopping, model evaluation
- **📱 Production-ready** - Export models for deployment in real applications

## ✨ Features

### Core Functionality

- **🎤 Audio Processing**: Automatic audio preprocessing, normalization, and
  feature extraction
- **🧠 Deep Learning**: CNN+LSTM architecture optimized for wakeword detection
- **📈 Training Management**: Complete training pipeline with validation and
  early stopping
- **📊 Evaluation**: Comprehensive model evaluation with multiple metrics
- **💾 Model Export**: Deployment-ready model packaging

### Advanced Features

- **🚀 GPU Acceleration**: Automatic CUDA detection and GPU optimization
- **🎨 Data Augmentation**: Time shifting, pitch shifting, speed changing, noise
  addition
- **🏠 Background Mixing**: Realistic background noise simulation
- **📈 Real-time Visualization**: Live training curves and performance plots
- **🔍 Audio Analysis**: Built-in audio visualization and feature inspection
- **⚙️ Configuration Management**: Comprehensive YAML-based configuration

### User Interface

- **🌐 Web Interface**: Modern Gradio-based web application
- **📱 Responsive Design**: Works on desktop, tablet, and mobile
- **🎨 Intuitive Controls**: Easy-to-use sliders, dropdowns, and visual feedback
- **📊 Live Updates**: Real-time training progress and system status
- **🧪 Interactive Testing**: Test models with audio uploads

## 🚀 Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install torch torchaudio gradio librosa soundfile numpy scikit-learn matplotlib seaborn pandas plotly tqdm
```

### 2. Prepare Your Data

```bash
# Create folder structure
python create_folder_structure.py

# Organize your audio files:
# - Positive samples: data/positive/train/
# - Negative samples: data/negative/train/
# - Background noise: data/background/train/
```

### 3. Launch the Application

```bash
# Start the training studio
python wakeword_app.py
```

### 4. Train Your Model

1. Open your browser to `http://localhost:7860`
2. Configure settings in the **Configuration** tab
3. Load your data and start training
4. Monitor progress in the **Training** tab
5. Evaluate results in the **Evaluation** tab

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
cd wakeword-training-studio

# Install Python dependencies
pip install -r requirements.txt

# Create folder structure
python create_folder_structure.py
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

### Option 3: Using pipenv

```bash
# Install pipenv if not available
pip install pipenv

# Install dependencies
pipenv install

# Activate environment
pipenv shell
```

## 📁 Project Structure

```
wakeword-training-studio/
├── wakeword_app.py              # 🎯 Main application (single file)
├── config.yaml                  # ⚙️ Unified configuration
├── create_folder_structure.py   # 📁 Folder structure creator
├── requirements.txt             # 📦 Dependencies
├── README.md                    # 📚 This documentation
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
│   ├── background/             # 🔊 Background noise
│   │   ├── train/
│   │   ├── validation/
│   │   └── test/
│   ├── raw/                    # 📁 Original audio files
│   └── processed/              # ⚙️ Preprocessed audio
│
├── features/                   # 🔍 Extracted features
│   ├── train/                  #   Training features
│   ├── validation/             #   Validation features
│   └── cache/                  #   Feature cache
│
├── results/                    # 📈 Training results
│   ├── plots/                  #   Training visualizations
│   ├── metrics/                #   Evaluation metrics
│   └── logs/                   #   Training logs
│
├── models/                     # 💾 Trained models
│   └── checkpoints/            #   Model checkpoints
│
├── datasets/                   # 📚 External datasets
│   └── mit_rirs/               #   Room impulse responses
│
└── temp/                       # 🗂️ Temporary files
    ├── cache/                  #   Application cache
    └── downloads/              #   Downloaded files
```

## 🎛️ Configuration

The system uses a comprehensive YAML configuration file (`config.yaml`) that
controls all aspects of training:

### Key Configuration Sections:

- **Audio Processing**: Sample rate, mel-spectrogram parameters
- **Model Architecture**: CNN layers, LSTM configuration, output settings
- **Training Parameters**: Batch size, learning rate, epochs, optimization
- **Data Augmentation**: Augmentation probabilities and parameters
- **System Settings**: GPU usage, memory management, logging

### Default Configuration:

```yaml
audio:
    sample_rate: 16000
    duration: 1.7
    n_mels: 80

model:
    hidden_size: 512
    num_layers: 2
    dropout: 0.6

training:
    batch_size: 32
    learning_rate: 0.0001
    epochs: 100

augmentation:
    probability: 0.85
    time_shift_max: 0.3
    pitch_shift_max: 1.5
```

## 📈 Data Preparation

### Dataset Requirements

| Dataset Type   | Minimum Files | Recommended | Quality Requirements        |
| -------------- | ------------- | ----------- | --------------------------- |
| **Positive**   | 100           | 500-1000    | Clean wakeword recordings   |
| **Negative**   | 450           | 2000-4000   | Random speech, no wakewords |
| **Background** | 1000          | 5000+       | Environmental noise         |

### Audio Specifications

- **Format**: WAV, MP3, FLAC, M4A, OGG
- **Sample Rate**: 16kHz (automatically resampled)
- **Channels**: Mono (automatically converted)
- **Duration**: 1-3 seconds
- **Quality**: Clean, no clipping, minimal background noise

### Data Organization

```
data/positive/train/     # 70% of positive samples
data/positive/validation/# 20% of positive samples
data/positive/test/      # 10% of positive samples
```

### Recording Tips

1. **Multiple Speakers**: Include different voices, ages, accents
2. **Various Environments**: Record in quiet, office, and noisy settings
3. **Different Devices**: Use phone, laptop, USB microphones
4. **Consistent Pronunciation**: Maintain consistent wakeword pronunciation
5. **Natural Speech**: Speak naturally, avoid robotic pronunciation

## 🧠 Model Architecture

### CNN + LSTM Architecture

```
Input (80×54 mel-spectrogram)
    ↓
Conv2D (32 filters, 3×3)
    ↓
Conv2D (64 filters, 3×3)
    ↓
Conv2D (128 filters, 3×3)
    ↓
Adaptive Pooling (1×1)
    ↓
LSTM (512 hidden, 2 layers)
    ↓
Dropout (0.6)
    ↓
Dense (2 classes)
    ↓
Softmax Output
```

### Key Features:

- **CNN Layers**: Extract spatial features from mel-spectrograms
- **LSTM Layers**: Model temporal dependencies in audio sequences
- **Attention Mechanism**: Focus on relevant time-frequency regions
- **Dropout**: Prevent overfitting during training
- **Binary Classification**: Wakeword vs. background classification

### Model Parameters: ~882K parameters

## 🏋️ Training Process

### Training Pipeline

1. **Data Loading**: Automatic audio loading and preprocessing
2. **Feature Extraction**: Mel-spectrogram computation with GPU acceleration
3. **Data Augmentation**: Real-time augmentation during training
4. **Model Training**: CNN+LSTM training with backpropagation
5. **Validation**: Continuous validation during training
6. **Early Stopping**: Automatic stopping when validation plateaus
7. **Model Saving**: Best model checkpoint preservation

### Training Monitoring

- **Live Loss Curves**: Real-time training/validation loss visualization
- **Accuracy Tracking**: Training and validation accuracy monitoring
- **Learning Rate**: Adaptive learning rate scheduling
- **GPU Utilization**: GPU memory and compute monitoring
- **Progress Bars**: Visual training progress indicators

### Training Tips

1. **Start Small**: Begin with smaller datasets to verify setup
2. **Monitor Overfitting**: Watch for large train/validation gaps
3. **Adjust Augmentation**: Tune augmentation based on dataset size
4. **Experiment with Parameters**: Try different learning rates, batch sizes
5. **Use Early Stopping**: Prevent overfitting with appropriate patience

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

### Evaluation Process

1. **Automatic Testing**: Built-in model evaluation after training
2. **Interactive Testing**: Upload audio files for manual testing
3. **Performance Visualization**: Confusion matrices and metric plots
4. **Model Comparison**: Compare different training runs
5. **Error Analysis**: Identify common misclassification patterns

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
- **Documentation**: Deployment and integration guide

### Production Considerations

- **Inference Speed**: Optimized for real-time processing
- **Memory Usage**: Efficient memory management
- **Model Size**: Compressed models for edge deployment
- **Hardware Requirements**: CPU and GPU deployment options
- **Error Handling**: Robust error handling and logging

## 🔍 Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues

```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CUDA-compatible PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Audio Loading Errors

- **Check file formats**: Ensure supported audio formats
- **Verify file paths**: Check absolute vs relative paths
- **Test audio files**: Try loading with librosa directly

#### 3. Training Not Converging

- **Increase dataset size**: More training data often helps
- **Adjust learning rate**: Try lower learning rates
- **Reduce model complexity**: Smaller models for small datasets
- **Check data quality**: Ensure clean, consistent audio

#### 4. Overfitting Issues

- **Increase augmentation**: More data augmentation
- **Add regularization**: Higher dropout rates
- **Reduce model size**: Smaller network architecture
- **Get more data**: Collect additional training samples

#### 5. Memory Issues

- **Reduce batch size**: Smaller batches use less memory
- **Enable mixed precision**: Use fp16 training
- **Clear cache**: Regular cache clearing during training
- **Use gradient accumulation**: Effective larger batches

### Performance Optimization

- **GPU Memory**: Monitor GPU memory usage
- **Data Loading**: Optimize data loading pipeline
- **Model Compilation**: Compile models for faster inference
- **Batch Processing**: Efficient batch processing strategies

## 📚 Advanced Features

### GPU Acceleration

- **Automatic Detection**: Automatic CUDA GPU detection
- **Memory Management**: Intelligent GPU memory management
- **Multi-GPU Support**: Distributed training across multiple GPUs
- **Mixed Precision**: FP16 training for faster convergence

### Data Augmentation

- **Time Shifting**: Temporal alignment variations
- **Pitch Shifting**: Frequency content modifications
- **Speed Changing**: Temporal scaling variations
- **Noise Addition**: Background noise simulation
- **Background Mixing**: Realistic acoustic environments

### Advanced Training

- **Learning Rate Scheduling**: Adaptive learning rate strategies
- **Early Stopping**: Automatic training termination
- **Model Checkpointing**: Regular model state preservation
- **Hyperparameter Optimization**: Automated parameter tuning

### Research Features

- **Attention Mechanisms**: Advanced attention layers
- **Transformer Integration**: Modern transformer architectures
- **Self-Supervised Learning**: Pre-training strategies
- **Transfer Learning**: Pre-trained model adaptation

## 🤝 Contributing

We welcome contributions to improve Wakeword Training Studio!

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
- **Data Augmentation**: Novel augmentation techniques
- **Training Strategies**: Advanced training methods
- **Evaluation Metrics**: Additional performance measures
- **User Interface**: Enhanced web interface features
- **Documentation**: Improved guides and tutorials

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file
for details.

---

## 🎯 Summary

**Wakeword Training Studio** provides a complete, production-ready solution for
training custom wakeword detection models. With its intuitive interface,
advanced features, and comprehensive documentation, it democratizes wakeword
training technology for developers, researchers, and enthusiasts.

### Key Benefits:

- **🚀 Fast Setup**: Get started in minutes, not hours
- **🎨 User-Friendly**: No deep learning expertise required
- **⚡ High Performance**: GPU-accelerated training and inference
- **📊 Comprehensive**: Complete training pipeline included
- **🔧 Customizable**: Extensive configuration options
- **📱 Production-Ready**: Deploy models to real applications

### Next Steps:

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Prepare your data**: Organize audio files in the data directory
3. **Launch the app**: `python wakeword_app.py`
4. **Start training**: Use the web interface to train your model
5. **Deploy**: Export your trained model for production use

---

**Happy Wakeword Training!** 🎉

For support and questions, please check the troubleshooting section or create an
issue in the repository.
