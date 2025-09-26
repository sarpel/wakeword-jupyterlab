#!/usr/bin/env python3
"""
🎯 Wakeword Training Studio - Folder Structure Creator
Creates organized folder structure for wakeword training datasets
"""

import os
from pathlib import Path

def create_folder_structure():
    """Create complete folder structure for wakeword training"""

    # Define the folder structure
    folders = [
        # Data directories
        "data/positive/train",
        "data/positive/validation",
        "data/positive/test",
        "data/negative/train",
        "data/negative/validation",
        "data/negative/test",
        "data/background/train",
        "data/background/validation",
        "data/background/test",
        "data/raw",
        "data/processed",

        # Feature directories
        "data/features/train",
        "data/features/validation",
        "data/features/cache",

        # Results directories
        "results/plots",
        "results/metrics",
        "results/logs",

        # Model directories
        "models/checkpoints",

        # Temporary directories
        "temp/cache",
        "temp/downloads",

        # RIRS dataset (optional)
        "datasets/mit_rirs/rir_data",

        # Feature cache
        "features/cache",
        "features/train",
        "features/validation"
    ]

    print("🎯 Creating Wakeword Training Studio folder structure...")

    created_count = 0
    for folder in folders:
        try:
            Path(folder).mkdir(parents=True, exist_ok=True)
            print(f"✅ Created: {folder}")
            created_count += 1
        except Exception as e:
            print(f"❌ Error creating {folder}: {e}")

    print(f"\n📊 Summary:")
    print(f"✅ Successfully created {created_count} folders")

    # Create placeholder README files
    readme_files = {
        "data/positive/train/README.md": "# Positive Training Data\n\nPlace your wakeword recordings here.\n\n## Requirements:\n- Audio format: WAV, 16kHz, mono\n- Duration: 1-2 seconds\n- Minimum files: 100\n- Naming: descriptive names (e.g., 'hey_computer_001.wav')",

        "data/negative/train/README.md": "# Negative Training Data\n\nPlace negative speech samples here.\n\n## Requirements:\n- Audio format: WAV, 16kHz, mono\n- Duration: 1-3 seconds\n- Content: Random speech, no wakewords\n- Minimum files: 450",

        "data/background/train/README.md": "# Background Training Data\n\nPlace background noise samples here.\n\n## Requirements:\n- Audio format: WAV, 16kHz, mono\n- Duration: Variable\n- Content: Environmental noise, fan sounds, etc.\n- Minimum files: 1000",

        "results/README.md": "# Training Results\n\nThis directory contains training results:\n- `plots/`: Training visualization plots\n- `metrics/`: Evaluation metrics and reports\n- `logs/`: Training logs and debugging information",

        "models/README.md": "# Trained Models\n\nThis directory contains trained models:\n- `checkpoints/`: Model checkpoints during training\n- `best_wakeword_model.pth`: Best model checkpoint\n- `wakeword_deployment_model.pth`: Deployment-ready model",

        "datasets/README.md": "# External Datasets\n\nThis directory contains external datasets:\n- `mit_rirs/`: MIT Room Impulse Response dataset (optional)\n\n## MIT RIRS Dataset\nDownload from: https://mcdermottlab.mit.edu/Reverb/IR_Survey.html\nImproves model robustness to room acoustics."
    }

    print("\n📝 Creating README files...")

    readme_count = 0
    for filepath, content in readme_files.items():
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Created: {filepath}")
            readme_count += 1
        except Exception as e:
            print(f"❌ Error creating {filepath}: {e}")

    print(f"\n📄 Summary:")
    print(f"✅ Successfully created {readme_count} README files")

    # Create sample data structure info
    structure_info = """
📁 Wakeword Training Studio - Data Structure

🎯 Recommended Dataset Organization:

📊 Minimum Requirements:
• Positive samples: 100+ wakeword recordings
• Negative samples: 450+ speech samples
• Background samples: 1000+ noise recordings

📁 Data Structure:
data/
├── positive/
│   ├── train/          # 70% of positive samples
│   ├── validation/     # 20% of positive samples
│   └── test/           # 10% of positive samples
├── negative/
│   ├── train/          # 70% of negative samples
│   ├── validation/     # 20% of negative samples
│   └── test/           # 10% of negative samples
├── background/
│   ├── train/          # 70% of background samples
│   ├── validation/     # 20% of background samples
│   └── test/           # 10% of background samples
├── raw/                # Original unprocessed audio
└── processed/          # Preprocessed audio files

🔧 Audio Requirements:
• Format: WAV, 16kHz, mono
• Duration: 1-3 seconds
• Quality: Clean, no clipping
• Naming: descriptive (e.g., "hey_computer_001.wav")

⚡ Quick Start:
1. Place your audio files in the appropriate directories
2. Run: python wakeword_app.py
3. Configure settings in the web interface
4. Start training!
"""

    with open("data/DATA_STRUCTURE_GUIDE.md", 'w', encoding='utf-8') as f:
        f.write(structure_info.strip())

    print("\n🎉 Folder structure creation completed!")
    print("📖 Check data/DATA_STRUCTURE_GUIDE.md for detailed information")

if __name__ == "__main__":
    create_folder_structure()
