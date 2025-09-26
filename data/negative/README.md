# ❌ Negative Samples - Non-Wakeword Speech

**Diverse speech recordings that do NOT contain the wakeword**

## 🎯 Purpose

This directory contains negative examples - speech recordings that do NOT
contain your wakeword. These samples teach the model to distinguish between
wakeword and regular speech, preventing false positives.

## 📊 Requirements

### Minimum Requirements

- **Files**: 450+ recordings (2000+ recommended)
- **Duration**: 1-3 seconds per recording (variable lengths OK)
- **Format**: WAV, MP3, FLAC, M4A, or OGG
- **Content**: Speech WITHOUT wakeword instances

### Directory Structure

```
negative/
├── train/              # 70% of files (training)
├── validation/         # 20% of files (validation)
└── test/              # 10% of files (testing)
```

## 🎤 Content Guidelines

### Speech Variety

- **Conversations**: Natural dialogue between people
- **Lectures**: Educational or presentation speech
- **Podcasts**: Casual or formal podcast content
- **News**: Broadcast news recordings
- **Interviews**: Question and answer sessions
- **Phone Calls**: Telephone conversation recordings
- **Meetings**: Business or casual meeting recordings

### Speaker Diversity

- **Multiple Speakers**: 50+ different voices minimum
- **Gender Balance**: Good mix of male and female speakers
- **Age Range**: Various age groups (young, middle, senior)
- **Accents**: Different regional and international accents
- **Speaking Styles**: Formal, casual, fast, slow, emotional
- **Languages**: Primarily target language, some multilingual OK

### Environmental Variety

- **Quiet Rooms**: Clean speech recordings
- **Office Environments**: Moderate background noise
- **Home Settings**: Casual home recordings
- **Public Spaces**: Light public background noise
- **Transportation**: Car, bus, train environments
- **Outdoor**: Light outdoor background acceptable

## 📝 Recording Best Practices

### Content Selection

1. **Avoid Wakeword**: Ensure NO instances of your wakeword
2. **Natural Speech**: Use conversational, natural speaking patterns
3. **Varied Topics**: Cover diverse subjects and contexts
4. **Different Lengths**: Mix of short and longer utterances
5. **Multiple Contexts**: Various speaking situations and purposes

### Quality Standards

1. **Clear Audio**: Good signal-to-noise ratio
2. **Consistent Levels**: Avoid extreme volume variations
3. **Minimal Overlap**: Avoid multiple speakers talking simultaneously
4. **Clean Editing**: Remove long pauses and obvious edits
5. **Format Consistency**: Use consistent audio format and quality

### Content to Include

- **Daily Conversations**: "How was your day?", "What's for dinner?"
- **Work Discussions**: "Let's schedule a meeting", "Can you review this?"
- **Casual Chat**: "Did you see the game?", "The weather is nice"
- **Questions**: "Where are we going?", "What time is it?"
- **Statements**: "I think we should leave soon", "That sounds good"
- **Commands**: "Call mom", "Send email to John", "Set timer for 5 minutes"

### Content to AVOID

- **Your Wakeword**: Any instance of the target wakeword
- **Similar Sounds**: Words that sound very similar to wakeword
- **Music**: Songs, humming, or musical content
- **Non-speech**: Animal sounds, machine noises, etc.
- **Very Short**: Files shorter than 0.5 seconds
- **Very Long**: Files longer than 5 seconds (unless necessary)

## 🏷️ File Naming Conventions

### Recommended Format

```
negative_[speaker]_[environment]_[content]_[number].wav
```

### Examples

```
negative_john_office_conversation_001.wav
negative_sarah_home_daily_chat_002.wav
negative_mike_public_question_003.wav
```

### Alternative Format

```
speech_[number].wav
non_wakeword_[number].wav
```

## 📚 Suggested Data Sources

### Public Datasets

- **LibriSpeech**: Audiobook recordings
- **Common Voice**: Crowdsourced speech dataset
- **TED-LIUM**: TED talk recordings
- **VoxForge**: Open source speech corpus
- **TIMIT**: Acoustic-phonetic continuous speech
- **WSJ**: Wall Street Journal speech corpus

### Creative Commons Sources

- **Podcast Episodes**: With appropriate licensing
- **YouTube Audio**: Creative Commons licensed content
- **Audiobook Samples**: Public domain or licensed content
- **Educational Content**: Lectures and presentations
- **News Broadcasts**: Public domain news content

### Recording Your Own

- **Daily Conversations**: Record natural family/friend discussions
- **Work Meetings**: With permission from participants
- **Phone Calls**: With consent from all parties
- **Voice Messages**: Anonymized voicemail samples
- **Reading Aloud**: Reading books, articles, or scripts

## 🔍 Quality Control Checklist

### Audio Quality

- [ ] Clear speech without distortion
- [ ] Consistent volume levels
- [ ] Minimal background noise
- [ ] No clipping or over-modulation
- [ ] Proper recording levels
- [ ] Clean audio without artifacts

### Content Quality

- [ ] No wakeword instances present
- [ ] Natural, conversational speech
- [ ] Good variety of content and speakers
- [ ] Appropriate duration (1-3 seconds)
- [ ] Diverse speaking styles and contexts
- [ ] Sufficient speaker diversity

### Technical Quality

- [ ] Consistent sample rate (16kHz recommended)
- [ ] Mono channel format
- [ ] Clean file naming
- [ ] Proper file organization
- [ ] No corrupted files
- [ ] Complete metadata if available

## ⚠️ Common Mistakes to Avoid

### Content Issues

- **Including Wakeword**: Even accidental instances can harm training
- **Too Similar**: Words that sound very similar to wakeword
- **Insufficient Variety**: Limited speaker or content diversity
- **Poor Quality**: Distorted, clipped, or noisy recordings
- **Wrong Language**: Primarily non-target language speech
- **Music Content**: Including songs or musical speech

### Technical Issues

- **Inconsistent Format**: Mixed sample rates or channels
- **Extreme Volumes**: Too quiet or too loud recordings
- **Poor Editing**: Long silences or abrupt cuts
- **Corrupted Files**: Incomplete or damaged audio files
- **Wrong Duration**: Too short (<0.5s) or too long (>5s)
- **Naming Confusion**: Unclear or inconsistent file names

## 🚀 Automated Processing

### Using the Automated System

1. Place all negative recordings in `negative/` folder
2. Click **"Detect Dataset Status"** in the application
3. Click **"Auto-Split Dataset"** to organize files
4. Verify distribution with **"Get Dataset Info"**

### Manual Verification

```bash
# Count total files
ls negative/*.wav | wc -l

# Check for wakeword instances (manual review required)
# Listen to random samples to verify content
# Ensure good variety across files
```

## 📈 Performance Impact

### Good Negative Dataset

- **Low False Positives**: Minimal incorrect wakeword detection
- **High Precision**: Accurate wakeword identification
- **Robust Performance**: Works across different speech contexts
- **Speaker Independence**: Not biased to specific speakers
- **Environment Resilience**: Handles various acoustic conditions

### Poor Negative Dataset

- **High False Positives**: Regular speech triggers wakeword detection
- **Poor Precision**: Many incorrect positive predictions
- **Overfitting**: Poor generalization to new speech contexts
- **Speaker Bias**: Performance varies significantly across speakers
- **Environment Sensitivity**: Poor performance in new acoustic environments

## 🔧 Advanced Tips

### Dataset Balance

- **Quantity**: More negative samples than positive (4:1 ratio recommended)
- **Diversity**: Greater diversity in negative samples
- **Context Coverage**: Cover all contexts where wakeword might appear
- **Temporal Variety**: Include speech from different time periods
- **Source Diversity**: Mix of different recording sources and conditions

### Quality Enhancement

- **Consistent Processing**: Apply same preprocessing to all files
- **Noise Management**: Consistent approach to background noise
- **Volume Normalization**: Consistent loudness levels
- **Format Standardization**: Use consistent audio format
- **Metadata**: Maintain recording source and context information

### Collection Strategy

- **Systematic Collection**: Plan collection across different contexts
- **Regular Updates**: Add new samples based on performance feedback
- **User Feedback**: Incorporate real-world false positive examples
- **Continuous Improvement**: Regularly review and enhance dataset
- **Bias Testing**: Test for biases across different user groups

## 📊 Integration with Training

### Automatic Processing

The system automatically:

- Scans for negative audio files
- Validates file formats and content
- Extracts mel-spectrogram features
- Balances with positive samples using weighted sampling
- Applies appropriate data augmentation

### Feature Extraction

- **Mel-spectrograms**: 64 mel-frequency bands
- **Duration**: 1-second analysis windows
- **Sample Rate**: 16kHz processing
- **Normalization**: Automatic level adjustment
- **Caching**: Pre-extracted features for faster training

---

**Happy Negative Sample Collection!** 🎤

For automated dataset management and processing, use the **Dataset Management**
tab in the main application.
