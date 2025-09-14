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

## GPU Support Status:
✅ GPU support is already installed with this script!
- Latest PyTorch with CUDA 11.8 support included
- To verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`
