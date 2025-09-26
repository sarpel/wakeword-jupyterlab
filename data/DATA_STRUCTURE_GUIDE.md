📁 Wakeword Training Studio - Data Structure

🎯 **NEW: Automated Dataset Management System!**

✨ **What's New:** • **Auto-detection**: Automatically detects when you place
files in dataset folders • **One-click splitting**: Single button to split all
datasets into train/test/validate • **Enhanced categories**: Support for
positive, negative, hard-negative, background, RIRs, and features • **Smart
validation**: Validates file counts and provides recommendations • **Visual
interface**: Easy-to-use Gradio interface with real-time feedback

📊 **Minimum Requirements:** • Positive samples: 100+ wakeword recordings •
Negative samples: 450+ speech samples • Background samples: 1000+ noise
recordings • Hard-negative samples: 50+ (optional but recommended) • RIRs: 20+
(optional, for room acoustics) • Features: Pre-extracted .npy files (optional)

📁 **Automated Structure (NEW):** data/ ├── positive/ # Place wakeword files
here │ ├── train/ # Auto-created: 70% of files │ ├── validation/ # Auto-created:
20% of files │ └── test/ # Auto-created: 10% of files ├── negative/ # Place
non-wakeword speech here │ ├── train/ # Auto-created: 70% of files │ ├──
validation/ # Auto-created: 20% of files │ └── test/ # Auto-created: 10% of
files ├── hard_negative/ # Place hard negative samples here (optional) │ ├──
train/ # Auto-created: 70% of files │ ├── validation/ # Auto-created: 20% of
files │ └── test/ # Auto-created: 10% of files ├── background/ # Place
background noise here │ ├── train/ # Auto-created: 70% of files │ ├──
validation/ # Auto-created: 20% of files │ └── test/ # Auto-created: 10% of
files ├── rirs/ # Place Room Impulse Responses here (optional) │ ├── train/ #
Auto-created: 70% of files │ ├── validation/ # Auto-created: 20% of files │ └──
test/ # Auto-created: 10% of files ├── features/ # Place pre-extracted .npy
features here (optional) │ ├── train/ │ │ └── before/ # Feature files can be
sub-foldered │ ├── validation/ │ │ └── before/ │ └── test/ │ └── before/ ├──
raw/ # Original unprocessed audio └── processed/ # Preprocessed audio files

🔧 **Audio Requirements:** • Format: WAV, 16kHz, mono • Duration: 1-3 seconds •
Quality: Clean, no clipping • Naming: descriptive (e.g., "hey_computer_001.wav")

⚡ **NEW: Quick Start with Automation:**

1. **Create Structure**: Click "Create Dataset Structure" button
2. **Add Files**: Place your audio files in the respective folders (positive,
   negative, background, etc.)
3. **Detect Status**: Click "Detect Dataset Status" to see readiness
4. **Auto-Split**: Click "Auto-Split Dataset" to automatically organize files
5. **Verify**: Check "Get Dataset Info" to confirm everything is ready
6. **Train**: Start training in the Model Training tab!

🤖 **How the Auto-Splitting Works:** • **Smart Detection**: Scans each category
folder for audio files • **Ratio Handling**: Automatically splits files 70%
train, 20% validation, 10% test • **Random Distribution**: Shuffles files before
splitting for unbiased distribution • **Validation**: Ensures minimum files per
split (at least 1 file per category if possible) • **Error Handling**: Provides
detailed feedback if issues occur • **File Types**: Supports .wav, .mp3, and
.npy files

📈 **Dataset Status Indicators:** • ✅ **Ready**: Category has sufficient files
for splitting • ⚠️ **Insufficient**: Category needs more files to meet minimum
requirements • ❌ **Missing**: Category folder doesn't exist or is empty • 🎉
**Ready for Auto-Splitting**: Multiple categories ready for automated processing

💡 **Pro Tips:** • **Start Simple**: Begin with positive and negative
categories, add others later • **Quality over Quantity**: Better to have 100
high-quality files than 1000 poor ones • **Balance**: Try to maintain similar
numbers of files across categories • **Hard Negatives**: Include samples that
sound similar to your wakeword • **Background Variety**: Use diverse background
noises (office, street, home) • **Feature Pre-processing**: Extract features
beforehand for faster training

🚨 **Common Issues & Solutions:** • **"Not ready for auto-splitting"**: Add more
files to meet minimum requirements • **"Files not detected"**: Ensure files are
in the root of category folders (not in subfolders except for features) •
**"Splitting errors"**: Check file permissions and disk space • **"Training
fails"**: Verify dataset info shows proper file distribution

🎯 **Training Readiness Checklist:** □ Dataset structure created □ Files placed
in correct folders □ Auto-splitting completed successfully □ At least positive
and negative categories ready □ Total dataset size ≥ 100 files □ Proper
train/validation/test distribution confirmed

📚 **Advanced Features:** • **Live Monitoring**: Real-time training progress
with batch-level updates • **GPU Acceleration**: Automatic CUDA detection and
optimization • **Feature Caching**: .npy preprocessing for faster subsequent
training • **Model Export**: Comprehensive model information and export
capabilities • **Health Reports**: Dataset validation and health checking

🔄 **Workflow Summary:**

1. Create structure → 2. Add files → 3. Detect status → 4. Auto-split → 5.
   Verify → 6. Train → 7. Test

For support and advanced configurations, check the System Status tab and consult
the comprehensive logging output.
