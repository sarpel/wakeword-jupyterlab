#!/usr/bin/env python3
"""
MIT RIRS Dataset Setup Script
Downloads and sets up Room Impulse Response datasets for acoustic augmentation
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
import shutil
from pathlib import Path
import subprocess
import tempfile
from typing import List, Dict, Optional
import hashlib


class RIRSDatasetSetup:
    """Setup class for MIT RIRS and other Room Impulse Response datasets"""

    def __init__(self, base_dir: str = "datasets/mit_rirs"):
        self.base_dir = Path(base_dir)
        self.rir_data_dir = self.base_dir / "rir_data"
        self.metadata_dir = self.base_dir / "metadata"
        self.temp_dir = self.base_dir / "temp"

        # Create directories
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.rir_data_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    # Available RIRS datasets
    RIRS_DATASETS = {
        "mit_reverb": {
            "name": "MIT Acoustical Reverberation Scene Statistics",
            "url": "https://mcdermottlab.mit.edu/Reverb/IR_Survey/All_IRs.zip",
            "description": "271 environmental impulse responses from MIT",
            "size_mb": 15,
            "file_type": "zip"
        },
        "but_reverb": {
            "name": "BUT ReverbDB",
            "url": "https://data.but.vutbr.cz/public/reverb/BUT_ReverbDB.zip",
            "description": "Real room impulse responses with background noises",
            "size_mb": 120,
            "file_type": "zip",
            "alternate_url": "https://github.com/ButkoAV/ReverbDB/raw/master/BUT_ReverbDB.zip"
        },
        "air_dnn": {
            "name": "AIR DNN Dataset",
            "url": "https://www.openslr.org/resources/17/AIR_DNN_Reverb.zip",
            "description": "AIR impulse response dataset for DNN training",
            "size_mb": 200,
            "file_type": "zip"
        },
        "openair": {
            "name": "OpenAIR Impulse Responses",
            "url": "https://www.openslr.org/resources/22/openair-impulse-responses.zip",
            "description": "Various room impulse responses from OpenAIR",
            "size_mb": 50,
            "file_type": "zip"
        }
    }

    def download_file(self, url: str, destination: Path, chunk_size: int = 8192) -> bool:
        """Download file with progress bar"""
        try:
            print(f"Downloading {url}...")
            print(f"Destination: {destination}")

            def progress_hook(count, block_size, total_size):
                if total_size > 0:
                    percent = int(count * block_size * 100 / total_size)
                    if percent % 10 == 0:
                        mb_downloaded = count * block_size / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        print(f"\rProgress: {percent}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end='')
                    if count * block_size >= total_size:
                        print()  # New line when complete

            urllib.request.urlretrieve(url, destination, reporthook=progress_hook)
            return True

        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False

    def extract_archive(self, archive_path: Path, extract_to: Path) -> bool:
        """Extract zip or tar.gz archive"""
        try:
            print(f"Extracting {archive_path} to {extract_to}...")

            if archive_path.suffix.lower() == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_to)
            elif archive_path.suffixes[-2:] == ['.tar', '.gz'] or archive_path.suffix.lower() == '.tgz':
                with tarfile.open(archive_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(extract_to)
            else:
                print(f"Unsupported archive format: {archive_path}")
                return False

            print("Extraction completed successfully")
            return True

        except Exception as e:
            print(f"Error extracting {archive_path}: {e}")
            return False

    def find_audio_files(self, directory: Path) -> List[Path]:
        """Find all audio files in directory"""
        audio_extensions = {'.wav', '.flac', '.aiff', '.au', '.ogg', '.mp3'}
        audio_files = []

        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix.lower() in audio_extensions:
                    audio_files.append(file_path)

        return audio_files

    def organize_rir_files(self, source_dir: Path) -> int:
        """Organize RIR files into standard structure"""
        print("Organizing RIR files...")

        audio_files = self.find_audio_files(source_dir)
        organized_count = 0

        for audio_file in audio_files:
            # Copy to main RIR data directory
            dest_path = self.rir_data_dir / audio_file.name

            # Handle name conflicts
            counter = 1
            while dest_path.exists():
                stem = audio_file.stem
                suffix = audio_file.suffix
                dest_path = self.rir_data_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            shutil.copy2(audio_file, dest_path)
            organized_count += 1

        print(f"Organized {organized_count} RIR files")
        return organized_count

    def create_metadata(self, dataset_name: str, dataset_info: Dict):
        """Create metadata file for dataset"""
        from datetime import datetime

        metadata = {
            "dataset_name": dataset_name,
            "description": dataset_info["description"],
            "source_url": dataset_info["url"],
            "installation_date": datetime.now().isoformat(),
            "file_count": len(self.find_audio_files(self.rir_data_dir)),
            "total_size_mb": sum(f.stat().st_size for f in self.rir_data_dir.rglob("*") if f.is_file()) / (1024 * 1024),
            "compatible_with": ["wakeword_training", "speech_enhancement", "voice_activity_detection"]
        }

        metadata_file = self.metadata_dir / f"{dataset_name}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {metadata_file}")
    def setup_dataset(self, dataset_name: str) -> bool:
        """Setup a specific RIRS dataset"""
        if dataset_name not in self.RIRS_DATASETS:
            print(f"Unknown dataset: {dataset_name}")
            print(f"Available datasets: {list(self.RIRS_DATASETS.keys())}")
            return False

        dataset_info = self.RIRS_DATASETS[dataset_name]
        print(f"\nSetting up {dataset_info['name']}...")
        print(f"Description: {dataset_info['description']}")
        print(f"Size: ~{dataset_info['size_mb']} MB")

        # Download dataset
        archive_name = f"{dataset_name}.{dataset_info['file_type']}"
        archive_path = self.temp_dir / archive_name

        if not self.download_file(dataset_info["url"], archive_path):
            # Try alternate URL if available
            if "alternate_url" in dataset_info:
                print("Trying alternate URL...")
                if not self.download_file(dataset_info["alternate_url"], archive_path):
                    return False
            else:
                return False

        # Extract archive
        extract_dir = self.temp_dir / f"{dataset_name}_extracted"
        if not self.extract_archive(archive_path, extract_dir):
            return False

        # Organize files
        file_count = self.organize_rir_files(extract_dir)

        # Create metadata
        self.create_metadata(dataset_name, dataset_info)

        # Cleanup
        print("Cleaning up temporary files...")
        shutil.rmtree(extract_dir, ignore_errors=True)
        archive_path.unlink(missing_ok=True)

        print(f"\n✅ {dataset_info['name']} setup completed successfully!")
        print(f"RIR files available in: {self.rir_data_dir}")
        print(f"Total RIR files: {file_count}")

        return True

    def setup_all_datasets(self) -> bool:
        """Setup all available RIRS datasets"""
        print("Setting up all available RIRS datasets...")
        print("This may take a while due to large download sizes...")

        success_count = 0
        for dataset_name in self.RIRS_DATASETS.keys():
            if self.setup_dataset(dataset_name):
                success_count += 1
            print("-" * 50)

        print(f"\nSetup completed: {success_count}/{len(self.RIRS_DATASETS)} datasets installed")
        return success_count > 0

    def create_custom_rir_dataset(self, source_directory: str) -> bool:
        """Create custom RIR dataset from local directory"""
        source_path = Path(source_directory)
        if not source_path.exists():
            print(f"Source directory does not exist: {source_directory}")
            return False

        print(f"Creating custom RIR dataset from {source_path}...")

        # Find and copy audio files
        audio_files = self.find_audio_files(source_path)
        copied_count = 0

        for audio_file in audio_files:
            dest_path = self.rir_data_dir / audio_file.name

            # Handle name conflicts
            counter = 1
            while dest_path.exists():
                stem = audio_file.stem
                suffix = audio_file.suffix
                dest_path = self.rir_data_dir / f"{stem}_{counter}{suffix}"
                counter += 1

            shutil.copy2(audio_file, dest_path)
            copied_count += 1

        # Create metadata
        metadata = {
            "dataset_name": "custom_rirs",
            "description": "Custom RIR dataset from local directory",
            "source_directory": str(source_path),
            "installation_date": str(Path().cwd()),
            "file_count": copied_count,
            "total_size_mb": sum(f.stat().st_size for f in self.rir_data_dir.rglob("*") if f.is_file()) / (1024 * 1024),
            "compatible_with": ["wakeword_training", "speech_enhancement", "voice_activity_detection"]
        }

        metadata_file = self.metadata_dir / "custom_rirs_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"✅ Custom RIR dataset created successfully!")
        print(f"Copied {copied_count} RIR files")
        return True

    def verify_installation(self) -> Dict:
        """Verify RIRS installation and return statistics"""
        rir_files = self.find_audio_files(self.rir_data_dir)

        stats = {
            "installation_path": str(self.rir_data_dir),
            "total_rir_files": len(rir_files),
            "file_types": {},
            "total_size_mb": sum(f.stat().st_size for f in rir_files) / (1024 * 1024),
            "datasets_installed": [],
            "ready_for_use": len(rir_files) > 0
        }

        # Count file types
        for rir_file in rir_files:
            ext = rir_file.suffix.lower()
            stats["file_types"][ext] = stats["file_types"].get(ext, 0) + 1

        # Check metadata
        for metadata_file in self.metadata_dir.glob("*_metadata.json"):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
                stats["datasets_installed"].append(metadata["dataset_name"])

        return stats

    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            print("Temporary files cleaned up")


def print_help():
    """Print help information"""
    print("""
MIT RIRS Dataset Setup Tool

Usage: python setup_rirs_dataset.py [command] [options]

Commands:
  list                    List available RIRS datasets
  setup <dataset>         Setup specific dataset (mit_reverb, but_reverb, air_dnn, openair)
  setup-all               Setup all available datasets
  custom <directory>      Create custom dataset from local directory
  verify                  Verify current installation
  cleanup                 Clean up temporary files
  help                    Show this help message

Examples:
  python setup_rirs_dataset.py list
  python setup_rirs_dataset.py setup mit_reverb
  python setup_rirs_dataset.py setup-all
  python setup_rirs_dataset.py custom /path/to/your/rirs
  python setup_rirs_dataset.py verify
""")


def main():
    """Main function"""
    if len(sys.argv) < 2:
        print_help()
        return

    command = sys.argv[1].lower()
    setup = RIRSDatasetSetup()

    if command == "list":
        print("Available RIRS Datasets:")
        print("-" * 50)
        for key, info in setup.RIRS_DATASETS.items():
            print(f"{key}:")
            print(f"  Name: {info['name']}")
            print(f"  Description: {info['description']}")
            print(f"  Size: ~{info['size_mb']} MB")
            print()

    elif command == "setup":
        if len(sys.argv) < 3:
            print("Error: Dataset name required")
            print("Usage: python setup_rirs_dataset.py setup <dataset_name>")
            return

        dataset_name = sys.argv[2]
        setup.setup_dataset(dataset_name)

    elif command == "setup-all":
        setup.setup_all_datasets()

    elif command == "custom":
        if len(sys.argv) < 3:
            print("Error: Source directory required")
            print("Usage: python setup_rirs_dataset.py custom <source_directory>")
            return

        source_dir = sys.argv[2]
        setup.create_custom_rir_dataset(source_dir)

    elif command == "verify":
        stats = setup.verify_installation()
        print("RIRS Installation Status:")
        print("-" * 30)
        print(f"Installation path: {stats['installation_path']}")
        print(f"Total RIR files: {stats['total_rir_files']}")
        print(f"Total size: {stats['total_size_mb']:.1f} MB")
        print(f"File types: {stats['file_types']}")
        print(f"Datasets installed: {', '.join(stats['datasets_installed'])}")
        print(f"Ready for use: {'Yes' if stats['ready_for_use'] else 'No'}")

    elif command == "cleanup":
        setup.cleanup()

    elif command == "help":
        print_help()

    else:
        print(f"Unknown command: {command}")
        print_help()


if __name__ == "__main__":
    # Import required modules
    try:
        import json
        main()
    except ImportError as e:
        print(f"Missing required module: {e}")
        print("Please install required packages: pip install requests urllib3")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
