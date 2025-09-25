#!/usr/bin/env python3
"""
Audio File Analyzer
Analyzes audio files in current directory and all subdirectories.
Provides detailed statistics per folder including file counts, sizes, and durations.
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import time

try:
    from mutagen import File as MutagenFile
    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    print("Warning: mutagen library not found. Install with: pip install mutagen")
    print("Duration calculations will be skipped.\n")

# Common audio file extensions
AUDIO_EXTENSIONS = {
    '.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma', '.opus',
    '.aiff', '.au', '.ra', '.3gp', '.amr', '.mp2', '.mpc', '.ape'
}

class AudioAnalyzer:
    def __init__(self, root_path="."):
        self.root_path = Path(root_path).resolve()
        self.folder_stats = defaultdict(lambda: {
            'files': defaultdict(int),  # file_type: count
            'total_files': 0,
            'total_size_bytes': 0,
            'total_duration_seconds': 0,
            'files_with_duration': 0
        })

    def get_audio_duration(self, file_path):
        """Get audio file duration in seconds using mutagen."""
        if not MUTAGEN_AVAILABLE:
            return 0

        try:
            audio_file = MutagenFile(file_path)
            if audio_file is not None and hasattr(audio_file, 'info'):
                return getattr(audio_file.info, 'length', 0)
        except Exception:
            pass
        return 0

    def format_size(self, size_bytes):
        """Convert bytes to human readable format."""
        if size_bytes == 0:
            return "0 B"

        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.2f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.2f} PB"

    def format_duration(self, seconds):
        """Convert seconds to hours:minutes:seconds format."""
        if seconds == 0:
            return "0:00:00"

        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}:{minutes:02d}:{secs:02d}"

    def analyze_directory(self):
        """Analyze all audio files in the directory tree."""
        print(f"Analyzing audio files in: {self.root_path}")
        print("=" * 60)

        total_files_processed = 0

        for root, dirs, files in os.walk(self.root_path):
            current_folder = Path(root)
            relative_folder = current_folder.relative_to(self.root_path)
            folder_key = str(relative_folder) if str(relative_folder) != '.' else 'ROOT'

            for file in files:
                file_path = current_folder / file
                file_ext = file_path.suffix.lower()

                if file_ext in AUDIO_EXTENSIONS:
                    try:
                        # Get file size
                        file_size = file_path.stat().st_size

                        # Get duration
                        duration = self.get_audio_duration(file_path)

                        # Update statistics
                        stats = self.folder_stats[folder_key]
                        stats['files'][file_ext] += 1
                        stats['total_files'] += 1
                        stats['total_size_bytes'] += file_size
                        stats['total_duration_seconds'] += duration
                        if duration > 0:
                            stats['files_with_duration'] += 1

                        total_files_processed += 1

                    except (OSError, PermissionError) as e:
                        print(f"Warning: Could not process {file_path}: {e}")

        print(f"Processed {total_files_processed} audio files\n")
        return total_files_processed > 0

    def print_summary(self):
        """Print detailed summary of audio files per folder."""
        if not self.folder_stats:
            print("No audio files found in the directory tree.")
            return

        print("AUDIO FILE ANALYSIS SUMMARY")
        print("=" * 80)

        # Calculate totals
        grand_total_files = 0
        grand_total_size = 0
        grand_total_duration = 0
        all_extensions = set()

        for stats in self.folder_stats.values():
            grand_total_files += stats['total_files']
            grand_total_size += stats['total_size_bytes']
            grand_total_duration += stats['total_duration_seconds']
            all_extensions.update(stats['files'].keys())

        # Print folder-by-folder breakdown
        for folder, stats in sorted(self.folder_stats.items()):
            print(f"\n📁 FOLDER: {folder}")
            print("-" * 40)
            print(f"Total Files: {stats['total_files']}")
            print(f"Total Size: {self.format_size(stats['total_size_bytes'])} ({stats['total_size_bytes']:,} bytes)")

            if MUTAGEN_AVAILABLE and stats['files_with_duration'] > 0:
                print(f"Total Duration: {self.format_duration(stats['total_duration_seconds'])}")
                avg_duration = stats['total_duration_seconds'] / stats['files_with_duration']
                print(f"Average Duration: {self.format_duration(avg_duration)}")
            else:
                print("Total Duration: N/A (mutagen not available or no duration data)")

            print("\nFile Types:")
            for ext, count in sorted(stats['files'].items()):
                percentage = (count / stats['total_files']) * 100
                print(f"  {ext}: {count} files ({percentage:.1f}%)")

        # Print grand totals
        print("\n" + "=" * 80)
        print("GRAND TOTALS")
        print("=" * 80)
        print(f"Total Audio Files: {grand_total_files:,}")
        print(f"Total Size: {self.format_size(grand_total_size)} ({grand_total_size:,} bytes)")
        print(f"Total Size (GB): {grand_total_size / (1024**3):.2f} GB")

        if MUTAGEN_AVAILABLE:
            print(f"Total Duration: {self.format_duration(grand_total_duration)}")
            print(f"Total Duration (Hours): {grand_total_duration / 3600:.2f} hours")

        print(f"\nFile Type Summary Across All Folders:")
        total_by_extension = defaultdict(int)
        for stats in self.folder_stats.values():
            for ext, count in stats['files'].items():
                total_by_extension[ext] += count

        for ext, count in sorted(total_by_extension.items()):
            percentage = (count / grand_total_files) * 100
            print(f"  {ext}: {count:,} files ({percentage:.1f}%)")

def main():
    """Main function to run the audio analyzer."""
    print("Audio File Analyzer")
    print("=" * 60)

    if not MUTAGEN_AVAILABLE:
        print("For complete functionality, install mutagen:")
        print("pip install mutagen\n")

    analyzer = AudioAnalyzer()

    start_time = time.time()

    if analyzer.analyze_directory():
        analyzer.print_summary()
    else:
        print("No audio files found in the current directory and subdirectories.")
        print(f"Searched in: {analyzer.root_path}")
        print(f"Supported formats: {', '.join(sorted(AUDIO_EXTENSIONS))}")

    end_time = time.time()
    print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()