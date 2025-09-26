#!/usr/bin/env python3
"""
Test script for the automated dataset management system
"""

import os
import sys
import shutil
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import the dataset management functionality
from wakeword_app import DatasetManager, logger

def test_dataset_management():
    """Comprehensive test of the dataset management system"""
    
    print("🧪 Testing Automated Dataset Management System")
    print("=" * 60)
    
    # Initialize dataset manager
    dataset_manager = DatasetManager()
    
    # Test 1: Create folder structure
    print("\n1️⃣ Testing folder structure creation...")
    success = dataset_manager.create_folder_structure()
    if success:
        print("✅ Folder structure created successfully")
    else:
        print("❌ Folder structure creation failed")
        return False
    
    # Verify structure
    expected_dirs = [
        "data/positive", "data/negative", "data/hard_negative", 
        "data/background", "data/rirs", "data/features",
        "data/positive/train", "data/positive/validation", "data/positive/test",
        "data/negative/train", "data/negative/validation", "data/negative/test"
    ]
    
    for dir_path in expected_dirs:
        if os.path.exists(dir_path):
            print(f"   ✅ {dir_path}")
        else:
            print(f"   ❌ {dir_path} - Missing")
    
    # Test 2: Create sample files
    print("\n2️⃣ Creating sample dataset files...")
    
    # Create sample files in different categories
    sample_files = {
        "data/positive": ["wakeword_001.wav", "wakeword_002.wav", "wakeword_003.wav"],
        "data/negative": ["speech_001.wav", "speech_002.wav", "speech_003.wav", "speech_004.wav"],
        "data/background": ["noise_001.wav", "noise_002.wav", "noise_003.wav", "noise_004.wav", "noise_005.wav"]
    }
    
    for category, files in sample_files.items():
        for file in files:
            file_path = os.path.join(category, file)
            with open(file_path, 'w') as f:
                f.write("dummy audio content")
            print(f"   Created: {file_path}")
    
    # Test 3: Detect dataset structure
    print("\n3️⃣ Testing dataset structure detection...")
    structure_info = dataset_manager.detect_dataset_structure()
    
    print(f"   Total files detected: {structure_info['total_files']}")
    print(f"   Ready for splitting: {structure_info['ready_for_splitting']}")
    print(f"   Ready categories: {', '.join(structure_info['ready_categories'])}")
    
    for category, info in structure_info['categories'].items():
        if info['exists']:
            status_emoji = "✅" if info['status'] == 'ready' else "⚠️" if info['status'] == 'insufficient' else "❌"
            print(f"   {status_emoji} {category}: {info['file_count']} files")
    
    # Test 4: Auto-split dataset
    print("\n4️⃣ Testing auto-split functionality...")
    if structure_info['ready_for_splitting']:
        results = dataset_manager.organize_dataset_files(structure_info)
        
        print(f"   Files moved: {results['moved_files']}")
        
        if results['errors']:
            print("   Errors encountered:")
            for error in results['errors']:
                print(f"     ❌ {error}")
        else:
            print("   ✅ Auto-splitting completed without errors")
        
        # Verify splits
        print("\n   Verifying split distribution:")
        stats = dataset_manager.get_dataset_statistics()
        for category, cat_stats in stats['total_files_by_category'].items():
            if cat_stats['total'] > 0:
                print(f"   📁 {category}:")
                print(f"      Total: {cat_stats['total']}")
                print(f"      Train: {cat_stats['train']} ({cat_stats['train']/cat_stats['total']*100:.1f}%)")
                print(f"      Validation: {cat_stats['validation']} ({cat_stats['validation']/cat_stats['total']*100:.1f}%)")
                print(f"      Test: {cat_stats['test']} ({cat_stats['test']/cat_stats['total']*100:.1f}%)")
    else:
        print("   ⚠️ Dataset not ready for auto-splitting")
    
    # Test 5: Comprehensive dataset info
    print("\n5️⃣ Testing comprehensive dataset information...")
    stats = dataset_manager.get_dataset_statistics()
    
    print(f"   Total dataset size: {stats['total_files']} files")
    print("   Category breakdown:")
    for category, cat_stats in stats['total_files_by_category'].items():
        if cat_stats['total'] > 0:
            print(f"     {category}: {cat_stats['total']} files")
    
    print("\n🎉 All tests completed!")
    return True

def cleanup_test_files():
    """Clean up test files"""
    print("\n🧹 Cleaning up test files...")
    
    # Remove sample files from root category folders
    categories = ['positive', 'negative', 'background', 'hard_negative', 'rirs']
    
    for category in categories:
        category_path = f"data/{category}"
        if os.path.exists(category_path):
            for file in os.listdir(category_path):
                file_path = os.path.join(category_path, file)
                if os.path.isfile(file_path) and file.endswith('.wav'):
                    os.remove(file_path)
                    print(f"   Removed: {file_path}")
    
    print("   ✅ Cleanup completed")

if __name__ == "__main__":
    try:
        success = test_dataset_management()
        if success:
            print("\n✨ Dataset management system is working correctly!")
        else:
            print("\n❌ Some tests failed")
        
        # Ask if user wants to cleanup
        response = input("\nDo you want to clean up test files? (y/n): ").lower()
        if response == 'y':
            cleanup_test_files()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()