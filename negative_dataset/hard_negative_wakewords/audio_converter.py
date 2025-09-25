#!/usr/bin/env python3
"""
Audio Converter Script

Converts all audio files in the specified folder and its subfolders to WAV format.
"""

import os
import sys
import glob
import warnings
import librosa
import soundfile as sf
import numpy as np
from tqdm import tqdm

# Configuration
SAMPLE_RATE = 16000
CHANNELS = 1
SUPPORTED_FORMATS = ['.mp3', '.m4a', '.flac', '.ogg', '.opus', '.aac', '.wma', '.wav', '.aiff', '.au']

def convert_file(input_path, output_path):
    """Convert a single audio file to WAV format"""
    try:
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        # Load audio file
        audio, sr = librosa.load(
            input_path,
            sr=SAMPLE_RATE,
            mono=(CHANNELS == 1)
        )

        # Normalize audio to prevent clipping
        if len(audio) > 0:
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95

        # Save as WAV
        sf.write(output_path, audio, SAMPLE_RATE, format='WAV')
        return True

    except Exception as e:
        print(f"ERROR: {input_path}: {str(e)}")
        return False

def main():
    if len(sys.argv) != 2:
        print("Usage: python audio_converter.py <input_folder>")
        sys.exit(1)

    input_folder = sys.argv[1]

    if not os.path.exists(input_folder):
        print(f"ERROR: Folder does not exist: {input_folder}")
        sys.exit(1)

    if not os.path.isdir(input_folder):
        print(f"ERROR: Path is not a folder: {input_folder}")
        sys.exit(1)

    print(f"Converting audio files in: {input_folder}")
    print(f"Output will be saved in the same directory structure")

    # Find all audio files recursively
    audio_files = []
    for ext in SUPPORTED_FORMATS:
        pattern = os.path.join(input_folder, '**', f'*{ext}')
        files = glob.glob(pattern, recursive=True)
        audio_files.extend(files)

    # Remove duplicates
    audio_files = list(set(audio_files))

    if not audio_files:
        print("No audio files found.")
        sys.exit(0)

    print(f"Found {len(audio_files)} audio files")

    converted_count = 0
    error_count = 0

    # Convert files with progress bar
    for input_file in tqdm(audio_files, desc="Converting"):
        # Generate output path (same path but with .wav extension)
        output_file = os.path.splitext(input_file)[0] + '.wav'

        # Convert the file
        if convert_file(input_file, output_file):
            converted_count += 1
        else:
            error_count += 1

    print(f"\nConversion complete:")
    print(f"  Successfully converted: {converted_count}")
    print(f"  Errors: {error_count}")
    print(f"  Total files processed: {len(audio_files)}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    main()