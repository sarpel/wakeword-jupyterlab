# ✅ Positive Samples - Wakeword Recordings

**High-quality wakeword recordings for training wake word detection models**

## 🎯 Purpose

This directory contains positive examples of your target wakeword. These
recordings are crucial for teaching the model to recognize when the wakeword is
spoken.

## 📊 Requirements

### Minimum Requirements

- **Files**: 100+ recordings (450+ recommended)
- **Duration**: 1-3 seconds per recording
- **Format**: WAV, MP3, FLAC, M4A, or OGG
- **Quality**: Clean, clear audio without clipping

### Directory Structure

```
positive/
├── train/              # 70% of files (training)
├── validation/         # 20% of files (validation)
└── test/              # 10% of files (testing)
```

## 🎤 Recording Guidelines

### Speaker Diversity

- **Multiple Speakers**: Include 10+ different voices minimum
- **Gender Balance**: Mix of male and female speakers
- **Age Range**: Include different age groups if applicable
- **Accents**: Consider regional accent variations
- **Speaking Styles**: Natural, conversational tone

### Environmental Variety

- **Quiet Environment**: Clean recordings for baseline
- **Office Noise**: Moderate background noise
- **Home Environment**: Typical household sounds
- **Outdoor**: Light outdoor background (if relevant)
- **Different Rooms**: Various acoustic environments

### Device Diversity

- **Smartphone**: Primary recording device
- **Laptop**: Built-in microphone recordings
- **USB Microphone**: Higher quality recordings
- **Bluetooth Headset**: Wireless device recordings
- **Tablet**: Alternative device recordings

## 📝 Recording Best Practices

### Before Recording

1. **Choose a quiet location** with minimal background noise
2. **Test your microphone** and adjust input levels
3. **Plan your wakeword** and ensure consistent pronunciation
4. **Prepare recording script** with variations

### During Recording

1. **Speak naturally** - avoid robotic or overly precise pronunciation
2. **Maintain consistent distance** from microphone (6-12 inches)
3. **Use normal speaking volume** - not too loud or soft
4. **Record multiple takes** of each variation
5. **Include natural pauses** before and after the wakeword

### Wakeword Variations

Record multiple variations of your wakeword:

- **Normal pronunciation**: Standard, clear pronunciation
- **Fast speech**: Quicker, natural speaking pace
- **Slow speech**: Deliberate, slower pronunciation
- **Soft voice**: Quiet, whisper-like pronunciation
- **Loud voice**: Clear, projected pronunciation
- **Questioning tone**: Rising intonation
- **Statement tone**: Falling intonation
- **Excited tone**: Energetic, enthusiastic pronunciation

## 🏷️ File Naming Conventions

### Recommended Format

```
[wakeword]_[speaker]_[environment]_[variation]_[number].wav
```

### Examples

```
hey_computer_john_quiet_normal_001.wav
wake_word_sarah_office_fast_002.wav
ai_assistant_mike_home_soft_003.wav
```

### Alternative Simple Format

```
wakeword_[number].wav
positive_[number].wav
```

## 🔍 Quality Control Checklist

### Audio Quality

- [ ] No clipping or distortion
- [ ] Consistent volume levels
- [ ] Clear pronunciation of wakeword
- [ ] Minimal background noise
- [ ] No sudden noises or interruptions
- [ ] Proper recording levels (no peaking)

### Content Quality

- [ ] Wakeword is clearly audible
- [ ] Consistent pronunciation across recordings
- [ ] Appropriate duration (1-3 seconds)
- [ ] Natural speaking style
- [ ] Good representation of variations
- [ ] Sufficient diversity in speakers/environments

## ⚠️ Common Mistakes to Avoid

### Recording Issues

- **Over-modulation**: Audio levels too high causing distortion
- **Under-modulation**: Audio levels too low causing poor quality
- **Inconsistent pronunciation**: Varying wakeword pronunciation
- **Too robotic**: Overly precise, unnatural speech
- **Background noise**: Excessive ambient noise
- **Sudden noises**: Coughs, chair movements, paper rustling

### Content Issues

- **Insufficient variety**: Not enough speaker/environment diversity
- **Wrong duration**: Recordings too short (<1s) or too long (>3s)
- **Poor labeling**: Inconsistent or unclear file names
- **Insufficient files**: Below minimum requirements
- **Quality inconsistency**: Mixed high/low quality recordings

## 🚀 Automated Processing

### Using the Automated System

1. Place all positive recordings in `positive/` folder
2. Click **"Detect Dataset Status"** in the application
3. Click **"Auto-Split Dataset"** to organize files
4. Verify distribution with **"Get Dataset Info"**

### Manual Organization (if needed)

```bash
# Count total files
ls positive/*.wav | wc -l

# Split manually (example for 100 files)
# Train: 70 files, Validation: 20 files, Test: 10 files
mv positive/wakeword_001.wav positive/train/
mv positive/wakeword_002.wav positive/train/
# ... continue for all files
```

## 📈 Performance Impact

### Good Positive Dataset

- **High Precision**: Few false positives
- **Good Recall**: Catches most wakeword instances
- **Robust Performance**: Works across environments
- **Speaker Independence**: Recognizes different voices
- **Noise Resilience**: Works with background noise

### Poor Positive Dataset

- **Overfitting**: Poor generalization to new speakers
- **Environment Bias**: Poor performance in new environments
- **False Negatives**: Missing actual wakeword instances
- **Inconsistent Results**: Variable performance

## 🔧 Advanced Tips

### Data Augmentation (Optional)

- **Speed Variation**: Slightly faster/slower versions
- **Pitch Shift**: Higher/lower pitch versions
- **Volume Variation**: Louder/softer versions
- **Background Addition**: Add light background noise
- **Time Stretch**: Slightly longer/shorter versions

### Quality Enhancement

- **Noise Reduction**: Clean recordings with audio tools
- **Normalization**: Consistent volume levels
- **Trimming**: Remove silence at beginning/end
- **Format Conversion**: Use consistent format (WAV recommended)
- **Metadata**: Add relevant tags and descriptions

### Collection Strategy

- **Incremental Collection**: Build dataset over time
- **Diversity First**: Prioritize variety over quantity initially
- **Quality Control**: Review and re-record poor samples
- **Regular Updates**: Add new samples based on performance
- **User Feedback**: Incorporate real-world usage feedback

## 📚 Integration with Training

### Automatic Loading

The system automatically:

- Scans for positive audio files
- Validates file formats and quality
- Extracts mel-spectrogram features
- Applies data augmentation during training
- Balances with negative samples

### Feature Extraction

- **Mel-spectrograms**: 64 mel-frequency bands
- **Duration**: 1-second windows
- **Sample Rate**: 16kHz processing
- **Normalization**: Automatic level adjustment
- **Caching**: Pre-extracted features for speed

---

**Happy Positive Sample Collection!** 🎤

For dataset management and automated processing, use the **Dataset Management**
tab in the main application.
