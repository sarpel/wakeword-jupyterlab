# 🔊 Background Noise Samples

**Environmental noise recordings for robust wakeword detection training**

## 🎯 Purpose

This directory contains background noise recordings that simulate real-world
acoustic environments. These samples are mixed with positive and negative
samples during training to create more robust models that work well in noisy
conditions.

## 📊 Requirements

### Minimum Requirements

- **Files**: 1000+ recordings (5000+ recommended)
- **Duration**: 1-10 seconds per recording (variable lengths OK)
- **Format**: WAV, MP3, FLAC, M4A, or OGG
- **Content**: Environmental noise WITHOUT speech

### Directory Structure

```
background/
├── train/              # 70% of files (training)
├── validation/         # 20% of files (validation)
└── test/              # 10% of files (testing)
```

## 🌍 Noise Categories

### Indoor Environments

- **Office Noise**: Keyboard typing, printer sounds, chair movement
- **Home Sounds**: Appliances, TV background, cooking sounds
- **Kitchen Noise**: Refrigerator hum, microwave, dishes
- **Living Room**: TV, air conditioning, footsteps
- **Bathroom**: Fan, water running, echo sounds
- **Bedroom**: Fan, air purifier, quiet ambient sounds

### Outdoor Environments

- **Street Traffic**: Cars, buses, motorcycles, traffic signals
- **Pedestrian Areas**: Footsteps, conversations, street vendors
- **Parks**: Birds, wind, distant traffic, people walking
- **Construction**: Distant construction sounds, machinery
- **Weather**: Rain, wind, thunder (light)
- **Urban Ambient**: City background, distant sirens

### Transportation

- **Car Interior**: Engine noise, road sounds, turn signals
- **Public Transit**: Bus, train, subway ambient noise
- **Airport**: Distant announcements, crowd noise, aircraft
- **Train Station**: Platform sounds, trains, crowds
- **Car Radio**: Background music, news (low volume)

### Technical/Mechanical

- **Air Conditioning**: HVAC systems, fans, ventilation
- **Computer Fans**: Desktop, laptop cooling sounds
- **Refrigerator**: Compressor noise, cycling sounds
- **Washing Machine**: Operation sounds, spinning
- **Clocks**: Ticking, mechanical sounds
- **Electrical**: Transformer hum, fluorescent lights

## 🎤 Recording Guidelines

### Recording Setup

1. **Use Quality Equipment**: Decent microphone for clean recordings
2. **Consistent Levels**: Maintain similar recording volumes
3. **Avoid Clipping**: Ensure no distortion in loud environments
4. **Document Context**: Note location, time, and conditions
5. **Multiple Takes**: Record several samples of each environment

### Environmental Capture

1. **Representative Duration**: Capture typical noise patterns
2. **Stable Recording**: Minimize handling noise and movement
3. **Natural Levels**: Don't artificially boost or reduce noise
4. **Context Awareness**: Record during typical usage times
5. **Safety First**: Don't record in dangerous or private situations

### Content Guidelines

- **No Speech**: Avoid recordings with clear human speech
- **Natural Levels**: Representative of real-world conditions
- **Consistent Quality**: Avoid recordings with sudden loud noises
- **Diverse Intensity**: Mix of quiet and moderate noise levels
- **Relevant Contexts**: Focus on environments where wakeword will be used

## 🏷️ File Naming Conventions

### Recommended Format

```
background_[location]_[source]_[intensity]_[number].wav
```

### Examples

```
background_office_keyboard_moderate_001.wav
background_street_traffic_loud_002.wav
background_home_kitchen_quiet_003.wav
background_car_engine_moderate_004.wav
```

### Alternative Format

```
noise_[location]_[number].wav
ambient_[source]_[number].wav
```

## 📊 Noise Intensity Classification

### Quiet (20-40 dB)

- **Library**: Very quiet room, occasional page turning
- **Bedroom**: Nighttime, very quiet ambient
- **Office Late**: After hours, minimal activity
- **Countryside**: Remote outdoor, natural sounds

### Moderate (40-60 dB)

- **Office Day**: Normal office activity, typing
- **Home Living**: Normal household activities
- **Street Quiet**: Residential street, light traffic
- **Park**: Natural outdoor, people walking

### Loud (60-80 dB)

- **Busy Street**: Heavy traffic, urban environment
- **Restaurant**: Crowded dining area
- **Public Transit**: Bus, train interior
- **Shopping Mall**: Crowded commercial space

### Very Loud (80+ dB)

- **Construction**: Near active construction
- **Busy Highway**: Heavy traffic, close proximity
- **Airport**: Near aircraft, busy terminals
- **Factory**: Industrial environment

## 🔍 Quality Control Checklist

### Audio Quality

- [ ] Clear recording without distortion
- [ ] Consistent noise levels throughout
- [ ] No sudden loud noises or clipping
- [ ] Appropriate recording levels
- [ ] Minimal handling or equipment noise
- [ ] Clean audio without artifacts

### Content Quality

- [ ] No clear human speech present
- [ ] Representative of intended environment
- [ ] Natural, realistic noise patterns
- [ ] Appropriate duration (1-10 seconds)
- [ ] Consistent with naming description
- [ ] Good variety across collection

### Technical Quality

- [ ] Consistent sample rate (16kHz)
- [ ] Mono channel format
- [ ] Clean file naming
- [ ] Proper metadata if available
- [ ] No corrupted files
- [ ] Appropriate file size

## ⚠️ Common Mistakes to Avoid

### Content Issues

- **Including Speech**: Clear human speech in background
- **Too Quiet**: Barely audible noise levels
- **Too Loud**: Distorted or overwhelming noise
- **Sudden Noises**: Door slams, phone rings, alarms
- **Inconsistent**: Varying noise levels within recording
- **Irrelevant**: Noises from unrelated environments

### Technical Issues

- **Clipping**: Distorted loud sections
- **Poor Quality**: Excessive compression or artifacts
- **Inconsistent Format**: Mixed sample rates or channels
- **Wrong Duration**: Too short (<1s) or too long (>10s)
- **Equipment Noise**: Obvious microphone or handling noise
- **Environmental Interference**: Wind, rain on microphone

## 🚀 Automated Processing

### Using the Automated System

1. Place all background recordings in `background/` folder
2. Click **"Detect Dataset Status"** in the application
3. Click **"Auto-Split Dataset"** to organize files
4. Verify distribution with **"Get Dataset Info"**

### Manual Organization

```bash
# Count files by environment type
ls background/*office* | wc -l
ls background/*street* | wc -l
ls background/*home* | wc -l

# Verify intensity distribution
ls background/*quiet* | wc -l
ls background/*moderate* | wc -l
ls background/*loud* | wc -l
```

## 📈 Performance Impact

### Good Background Dataset

- **Noise Robustness**: Good performance in noisy environments
- **Environment Adaptation**: Works across different acoustic conditions
- **False Positive Reduction**: Better discrimination in noise
- **Real-world Performance**: Practical deployment success
- **User Satisfaction**: Reliable operation in daily use

### Poor Background Dataset

- **Noise Sensitivity**: Poor performance with any background noise
- **Overfitting**: Works only in quiet conditions
- **False Positives**: Background noise triggers detection
- **Limited Deployment**: Only suitable for quiet environments
- **User Frustration**: Unreliable in real-world conditions

## 🔧 Advanced Tips

### Collection Strategy

- **Systematic Approach**: Plan collection across different environments
- **Time Variation**: Record at different times of day
- **Seasonal Consideration**: Account for seasonal differences
- **Geographic Diversity**: Different locations if applicable
- **Usage Context**: Focus on environments where wakeword will be used

### Quality Enhancement

- **Consistent Equipment**: Use same recording setup when possible
- **Calibration**: Maintain consistent recording levels
- **Documentation**: Detailed notes about recording conditions
- **Batch Processing**: Process similar environments together
- **Quality Control**: Regular review and re-recording if needed

### Dataset Balance

- **Intensity Distribution**: Balance across quiet/moderate/loud categories
- **Environment Coverage**: Good representation of all relevant environments
- **Temporal Variety**: Different times and conditions
- **Source Diversity**: Multiple recording sources and methods
- **Relevance Focus**: Prioritize environments matching deployment

## 📊 Integration with Training

### Mixing During Training

The system automatically:

- Mixes background noise with speech samples
- Adjusts noise levels for optimal training
- Applies random noise selection
- Maintains appropriate signal-to-noise ratios
- Validates noise mixing effectiveness

### Noise Augmentation

- **Random Selection**: Different noise for each training batch
- **Level Adjustment**: Variable noise intensity
- **Temporal Alignment**: Proper timing with speech samples
- **Quality Control**: Ensures effective noise addition
- **Performance Monitoring**: Tracks impact on model performance

---

**Happy Background Noise Collection!** 🔊

For automated dataset management and background noise processing, use the
**Dataset Management** tab in the main application.
