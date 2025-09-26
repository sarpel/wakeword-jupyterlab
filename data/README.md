# 📁 Enhanced Wakeword Dataset Management

**Automated dataset organization with intelligent splitting and comprehensive
validation**

## 🎯 Overview

This directory contains the complete dataset management system for the Enhanced
Wakeword Training Studio. The system features automated dataset organization,
intelligent file splitting, and comprehensive validation to streamline your
wakeword training workflow.

## ✨ New Automated Features

### 🤖 One-Click Dataset Management

- **Auto-structure creation**: Complete folder hierarchy in seconds
- **Smart file detection**: Automatic scanning and counting across all
  categories
- **Intelligent splitting**: Optimal 70/20/10 train/validation/test distribution
- **Real-time validation**: Instant feedback on dataset readiness
- **Comprehensive reporting**: Detailed statistics and health checks

### 📊 Enhanced Dataset Categories

- **Positive**: Wakeword recordings (minimum 100 files)
- **Negative**: Non-wakeword speech samples (minimum 450 files)
- **Hard Negative**: Challenging negative samples (minimum 50 files)
- **Background**: Environmental noise (minimum 1000 files)
- **RIRs**: Room Impulse Responses (optional)
- **Features**: Pre-extracted feature files (optional)

## 🚀 Quick Start Guide

### 1. Automated Setup (Recommended)

1. Launch the application: `python wakeword_app.py`
2. Navigate to **Dataset Management** tab
3. Click **"Create Dataset Structure"**
4. Add your audio files to appropriate folders
5. Click **"Detect Dataset Status"**
6. Click **"Auto-Split Dataset"**
7. Verify with **"Get Dataset Info"**

### 2. Manual Setup (Alternative)

```bash
# Create base structure manually
mkdir -p data/{positive,negative,hard_negative,background,rirs,features}/{train,validation,test}
mkdir -p data/{raw,processed}
```

## 📁 Directory Structure

```
data/
├── positive/                   # ✅ Wakeword recordings
│   ├── train/                  #   70% of files (training)
│   ├── validation/             #   20% of files (validation)
│   └── test/                   #   10% of files (testing)
│
├── negative/                   # ❌ Non-wakeword speech
│   ├── train/
│   ├── validation/
│   └── test/
│
├── hard_negative/              # ⚠️ Challenging negatives
│   ├── train/
│   ├── validation/
│   └── test/
│
├── background/                 # 🔊 Background noise
│   ├── train/
│   ├── validation/
│   └── test/
│
├── rirs/                       # 🏠 Room Impulse Responses
│   ├── train/
│   ├── validation/
│   └── test/
│
├── features/                   # 🔍 Pre-extracted features
│   ├── train/
│   │   └── before/            #   Can have subdirectories
│   ├── validation/
│   │   └── before/
│   └── test/
│       └── before/
│
├── raw/                        # 📁 Original unprocessed audio
└── processed/                  # ⚙️ Preprocessed audio files
```

## 📊 Dataset Requirements

### Minimum File Requirements

| Category          | Minimum Files | Recommended | Purpose                     |
| ----------------- | ------------- | ----------- | --------------------------- |
| **Positive**      | 100           | 500-1000    | Wakeword recordings         |
| **Negative**      | 450           | 2000-4000   | Random speech, no wakewords |
| **Background**    | 1000          | 5000+       | Environmental noise         |
| **Hard Negative** | 50            | 200-500     | Similar to wakeword         |
| **RIRs**          | 0             | 20+         | Room acoustics (optional)   |
| **Features**      | 0             | Variable    | Pre-extracted .npy files    |

### Audio Specifications

- **Formats**: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`, `.npy`
- **Sample Rate**: 16kHz (automatically resampled)
- **Channels**: Mono (automatically converted)
- **Duration**: 1-3 seconds optimal
- **Quality**: Clean, no clipping, minimal background noise

## 🔧 Automated Dataset Management

### Smart Detection Features

- **Recursive Scanning**: Searches all subdirectories for audio files
- **Format Recognition**: Supports all major audio formats
- **File Counting**: Accurate counting across complex directory structures
- **Validation Checking**: Real-time validation against minimum requirements

### Intelligent Splitting Algorithm

- **Random Distribution**: Shuffles files before splitting for unbiased
  distribution
- **Ratio Preservation**: Maintains 70/20/10 train/validation/test split
- **Minimum File Guarantee**: Ensures at least 1 file per split when possible
- **Error Handling**: Robust handling of edge cases and insufficient files

### Validation and Reporting

- **Readiness Status**: Clear indicators for dataset readiness
- **File Type Analysis**: Breakdown by audio format
- **Distribution Reports**: Detailed split statistics
- **Health Checks**: Comprehensive dataset validation
- **Recommendations**: Intelligent suggestions for improvements

## 📈 Data Collection Guidelines

### Positive Samples (Wakeword)

- **Multiple Speakers**: Include different voices, ages, accents
- **Various Environments**: Record in quiet, office, and noisy settings
- **Different Devices**: Use phone, laptop, USB microphones
- **Consistent Pronunciation**: Maintain consistent wakeword pronunciation
- **Natural Speech**: Speak naturally, avoid robotic pronunciation
- **File Naming**: Use descriptive names (e.g., `hey_computer_001.wav`)

### Negative Samples (Non-wakeword)

- **Speech Variety**: Include conversations, lectures, podcasts
- **Multiple Languages**: Consider multilingual environments
- **Different Contexts**: Office, home, outdoor recordings
- **Various Speakers**: Mix of genders, ages, accents
- **Background Variations**: Different acoustic environments

### Background Noise

- **Environmental Diversity**: Office, street, home, nature sounds
- **Consistent Levels**: Avoid extreme volume variations
- **Relevant Contexts**: Match your deployment environment
- **Quality Control**: Clean recordings without sudden noises

### Hard Negative Samples

- **Similar Sounds**: Words that sound like your wakeword
- **Partial Matches**: Incomplete wakeword pronunciations
- **Noisy Versions**: Wakewords with heavy background noise
- **Mispronunciations**: Common incorrect pronunciations

## 🚨 Common Issues & Solutions

### Dataset Not Detected

**Issue**: "No files found" or "Dataset not ready" **Solutions**:

- Ensure files are in supported formats (.wav, .mp3, etc.)
- Check file permissions and accessibility
- Verify files are placed directly in category folders
- Use "Detect Dataset Status" to get detailed feedback

### Auto-splitting Failures

**Issue**: "Auto-splitting failed" or errors during splitting **Solutions**:

- Ensure minimum file requirements are met
- Check available disk space for file operations
- Verify write permissions in data directories
- Review error messages for specific issues

### Training Readiness Issues

**Issue**: "Not ready for training" despite having files **Solutions**:

- Check that both positive and negative categories are ready
- Ensure proper train/validation/test distribution
- Verify total dataset size meets minimum requirements (100+ files)
- Use "Get Dataset Info" for detailed analysis

### File Organization Problems

**Issue**: Files not appearing in correct splits **Solutions**:

- Avoid deeply nested subdirectories (except for features)
- Ensure consistent file naming conventions
- Check for hidden or system files that might interfere
- Re-run auto-splitting after organizing files

## 💡 Pro Tips

### Quality Optimization

- **Start Simple**: Begin with positive and negative categories
- **Quality over Quantity**: 100 high-quality files > 1000 poor files
- **Balance**: Maintain similar file counts across categories
- **Diversity**: Include varied speakers, environments, and devices
- **Consistency**: Use consistent recording settings and formats

### Performance Enhancement

- **Feature Pre-processing**: Extract .npy features for faster training
- **Background Mixing**: Use diverse background noise for robustness
- **Hard Negatives**: Include challenging samples for better discrimination
- **Validation Sets**: Ensure representative validation splits
- **Testing**: Reserve high-quality samples for final testing

### Workflow Efficiency

- **Batch Upload**: Upload files in batches for better organization
- **Regular Validation**: Check dataset status frequently
- **Incremental Updates**: Add files incrementally and re-split
- **Backup**: Keep backups of original files before preprocessing
- **Documentation**: Document your data collection process

## 🔍 Advanced Features

### Dataset Health Monitoring

- **Comprehensive Analysis**: Detailed dataset statistics
- **Quality Metrics**: File format and quality validation
- **Distribution Analysis**: Split ratio verification
- **Error Detection**: Automatic identification of issues
- **Recommendations**: Intelligent improvement suggestions

### Integration with Training

- **Seamless Loading**: Automatic dataset loading for training
- **Feature Caching**: Pre-extracted feature support for speed
- **Real-time Updates**: Live dataset status during training
- **Export Integration**: Direct export of trained models
- **Performance Tracking**: Dataset impact on model performance

## 📚 Additional Resources

### File Naming Conventions

```
positive/
├── train/
│   ├── wake_word_001.wav
│   ├── wake_word_002.wav
│   └── hey_computer_003.wav
└── validation/
    ├── wake_word_val_001.wav
    └── validation_sample_002.wav
```

### Supported Audio Formats

- **WAV**: Uncompressed, highest quality
- **MP3**: Compressed, good for storage
- **FLAC**: Lossless compression
- **M4A**: Modern compressed format
- **OGG**: Open source compressed
- **NPY**: Pre-extracted feature files

### Quality Checklist

- [ ] Audio is clear without clipping
- [ ] Consistent volume levels
- [ ] No sudden noise interruptions
- [ ] Proper duration (1-3 seconds)
- [ ] Consistent sample rates
- [ ] Meaningful file names
- [ ] Appropriate folder organization

## 🔄 Workflow Summary

1. **Create Structure** → Use automated structure creation
2. **Add Files** → Place audio files in appropriate categories
3. **Detect Status** → Check dataset readiness and file counts
4. **Auto-Split** → Automatically organize into train/validation/test
5. **Verify** → Confirm proper distribution and readiness
6. **Train** → Use organized dataset for model training
7. **Monitor** → Track training progress and performance
8. **Deploy** → Export trained models for production use

---

**Happy Dataset Management!** 📊

For detailed troubleshooting and advanced configurations, consult the main
README.md file and check the System Status tab in the application.
