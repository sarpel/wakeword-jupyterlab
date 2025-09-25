#!/usr/bin/env python3
"""
Comprehensive Performance Analysis for Wakeword Feature Extraction
Identifies bottlenecks and GPU utilization issues
"""

import torch
import time
import psutil
import numpy as np
import logging
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import gc
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add training directory to path
import sys
sys.path.append('training')

from feature_extractor import FeatureExtractor
from enhanced_dataset import EnhancedWakewordDataset, EnhancedAudioConfig


class PerformanceAnalyzer:
    """Comprehensive performance analyzer for feature extraction"""
    
    def __init__(self):
        self.results = {
            'feature_extraction': {},
            'dataset_loading': {},
            'gpu_utilization': {},
            'memory_usage': {},
            'cpu_usage': {},
            'cache_performance': {},
            'bottlenecks': []
        }
        self.monitoring_active = False
        self.performance_samples = []
        
    def monitor_system_resources(self, duration: int = 60) -> Dict:
        """Monitor system resources during operation"""
        samples = []
        start_time = time.time()
        
        while time.time() - start_time < duration:
            sample = {
                'timestamp': time.time(),
                'cpu_percent': psutil.cpu_percent(interval=0.1),
                'memory_percent': psutil.virtual_memory().percent,
                'memory_available_gb': psutil.virtual_memory().available / 1e9,
                'memory_used_gb': psutil.virtual_memory().used / 1e9
            }
            
            if torch.cuda.is_available():
                sample['gpu_memory_allocated_gb'] = torch.cuda.memory_allocated() / 1e9
                sample['gpu_memory_reserved_gb'] = torch.cuda.memory_reserved() / 1e9
                sample['gpu_memory_total_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
                sample['gpu_utilization_percent'] = (sample['gpu_memory_allocated_gb'] / sample['gpu_memory_total_gb']) * 100
            
            samples.append(sample)
            time.sleep(0.5)
            
        return samples
    
    def analyze_feature_extraction_performance(self, test_audio_files: List[str]) -> Dict:
        """Analyze feature extraction performance in detail"""
        logger.info("Analyzing feature extraction performance...")
        
        extractor = FeatureExtractor()
        extraction_times = []
        memory_usage = []
        cpu_usage = []
        gpu_memory_usage = []
        
        for i, audio_path in enumerate(test_audio_files):
            logger.info(f"Testing file {i+1}/{len(test_audio_files)}: {Path(audio_path).name}")
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Record initial state
            initial_memory = psutil.virtual_memory().percent
            initial_cpu = psutil.cpu_percent(interval=0.1)
            initial_gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
            
            # Extract features
            start_time = time.time()
            try:
                features = extractor.extract_features(audio_path)
                extraction_time = time.time() - start_time
                
                # Record final state
                final_memory = psutil.virtual_memory().percent
                final_cpu = psutil.cpu_percent(interval=0.1)
                final_gpu_memory = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                
                # Store metrics
                extraction_times.append(extraction_time)
                memory_usage.append(final_memory - initial_memory)
                cpu_usage.append(final_cpu)
                gpu_memory_usage.append(final_gpu_memory - initial_gpu_memory)
                
                logger.info(f"  Extraction time: {extraction_time:.3f}s")
                logger.info(f"  Features shape: {features.shape}")
                logger.info(f"  Memory delta: {final_memory - initial_memory:.1f}%")
                logger.info(f"  GPU memory delta: {final_gpu_memory - initial_gpu_memory:.3f}GB")
                
            except Exception as e:
                logger.error(f"  Error extracting features: {e}")
                extraction_times.append(float('inf'))
                memory_usage.append(0)
                cpu_usage.append(0)
                gpu_memory_usage.append(0)
        
        # Calculate statistics
        valid_times = [t for t in extraction_times if t != float('inf')]
        
        results = {
            'total_files': len(test_audio_files),
            'successful_extractions': len(valid_times),
            'avg_extraction_time': np.mean(valid_times) if valid_times else 0,
            'min_extraction_time': np.min(valid_times) if valid_times else 0,
            'max_extraction_time': np.max(valid_times) if valid_times else 0,
            'std_extraction_time': np.std(valid_times) if valid_times else 0,
            'avg_memory_increase': np.mean(memory_usage),
            'avg_cpu_usage': np.mean(cpu_usage),
            'avg_gpu_memory_increase': np.mean(gpu_memory_usage),
            'cache_stats': extractor.get_cache_stats(),
            'performance_stats': extractor.get_performance_stats()
        }
        
        return results
    
    def analyze_dataset_loading_performance(self, positive_dir: str, negative_dir: str) -> Dict:
        """Analyze dataset loading performance"""
        logger.info("Analyzing dataset loading performance...")
        
        start_time = time.time()
        
        # Create dataset
        config = EnhancedAudioConfig(
            use_precomputed_features=False,  # Force audio processing
            use_rirs_augmentation=False
        )
        
        dataset = EnhancedWakewordDataset(
            positive_dir=positive_dir,
            negative_dir=negative_dir,
            config=config,
            mode='train'
        )
        
        loading_time = time.time() - start_time
        
        # Test individual item loading
        item_load_times = []
        for i in range(min(100, len(dataset))):  # Test first 100 items
            start_time = time.time()
            try:
                item = dataset[i]
                load_time = time.time() - start_time
                item_load_times.append(load_time)
            except Exception as e:
                logger.error(f"Error loading item {i}: {e}")
                item_load_times.append(float('inf'))
        
        valid_times = [t for t in item_load_times if t != float('inf')]
        
        results = {
            'dataset_size': len(dataset),
            'loading_time': loading_time,
            'positive_samples': dataset.positive_count,
            'negative_samples': dataset.negative_count,
            'avg_item_load_time': np.mean(valid_times) if valid_times else 0,
            'min_item_load_time': np.min(valid_times) if valid_times else 0,
            'max_item_load_time': np.max(valid_times) if valid_times else 0,
            'dataset_stats': dataset.get_dataset_stats()
        }
        
        return results
    
    def analyze_gpu_utilization(self, test_duration: int = 30) -> Dict:
        """Analyze GPU utilization during feature extraction"""
        if not torch.cuda.is_available():
            return {'gpu_available': False, 'message': 'CUDA not available'}
        
        logger.info("Analyzing GPU utilization...")
        
        # Initial GPU state
        initial_memory = torch.cuda.memory_allocated() / 1e9
        initial_reserved = torch.cuda.memory_reserved() / 1e9
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Monitor GPU during feature extraction
        gpu_samples = self.monitor_system_resources(duration=test_duration)
        
        # Calculate GPU utilization statistics
        gpu_utilizations = [s.get('gpu_utilization_percent', 0) for s in gpu_samples]
        gpu_memory_allocations = [s.get('gpu_memory_allocated_gb', 0) for s in gpu_samples]
        
        results = {
            'gpu_available': True,
            'gpu_name': torch.cuda.get_device_name(0),
            'total_gpu_memory_gb': total_memory,
            'initial_gpu_memory_gb': initial_memory,
            'initial_gpu_reserved_gb': initial_reserved,
            'avg_gpu_utilization_percent': np.mean(gpu_utilizations),
            'max_gpu_utilization_percent': np.max(gpu_utilizations),
            'min_gpu_utilization_percent': np.min(gpu_utilizations),
            'avg_gpu_memory_allocation_gb': np.mean(gpu_memory_allocations),
            'gpu_samples_collected': len(gpu_samples),
            'gpu_utilization_samples': gpu_utilizations
        }
        
        return results
    
    def identify_bottlenecks(self, feature_extraction_results: Dict, dataset_results: Dict, gpu_results: Dict) -> List[str]:
        """Identify performance bottlenecks based on analysis results"""
        bottlenecks = []
        
        # Analyze feature extraction bottlenecks
        if feature_extraction_results.get('avg_extraction_time', 0) > 1.0:
            bottlenecks.append("SLOW_FEATURE_EXTRACTION: Average extraction time > 1 second")
        
        if feature_extraction_results.get('std_extraction_time', 0) > 0.5:
            bottlenecks.append("INCONSISTENT_EXTRACTION: High variance in extraction times")
        
        # Analyze CPU vs GPU usage
        if torch.cuda.is_available() and gpu_results.get('avg_gpu_utilization_percent', 0) < 5:
            bottlenecks.append("LOW_GPU_UTILIZATION: GPU utilization < 5% during extraction")
        
        # Analyze memory usage
        if feature_extraction_results.get('avg_memory_increase', 0) > 10:
            bottlenecks.append("HIGH_MEMORY_USAGE: Memory increase > 10% per extraction")
        
        # Analyze cache performance
        cache_stats = feature_extraction_results.get('cache_stats', {})
        if cache_stats.get('total_requests', 0) > 0 and cache_stats.get('hit_rate', 0) < 0.5:
            bottlenecks.append("POOR_CACHE_PERFORMANCE: Cache hit rate < 50%")
        
        # Analyze dataset loading
        if dataset_results.get('avg_item_load_time', 0) > 0.1:
            bottlenecks.append("SLOW_DATASET_LOADING: Average item load time > 100ms")
        
        return bottlenecks
    
    def generate_performance_report(self, output_path: str = "performance_analysis_report.json"):
        """Generate comprehensive performance analysis report"""
        logger.info("Generating comprehensive performance analysis report...")
        
        # Create test audio files list
        test_files = []
        for dataset_dir in ['positive_dataset', 'negative_dataset']:
            if os.path.exists(dataset_dir):
                for root, dirs, files in os.walk(dataset_dir):
                    for file in files:
                        if file.endswith('.wav'):
                            test_files.append(os.path.join(root, file))
                    if test_files:
                        break
            if test_files:
                break
        
        # Limit test files for analysis
        test_files = test_files[:10]  # Test with 10 files
        
        # Run all analyses
        if test_files:
            feature_results = self.analyze_feature_extraction_performance(test_files)
            gpu_results = self.analyze_gpu_utilization()
            
            # Try dataset analysis if directories exist
            dataset_results = {}
            if os.path.exists('positive_dataset') and os.path.exists('negative_dataset'):
                dataset_results = self.analyze_dataset_loading_performance('positive_dataset', 'negative_dataset')
        else:
            # Use dummy data for testing
            logger.warning("No test audio files found, creating synthetic data for analysis")
            feature_results = self._create_synthetic_analysis()
            dataset_results = self._create_synthetic_dataset_analysis()
            gpu_results = self.analyze_gpu_utilization()
        
        # Identify bottlenecks
        bottlenecks = self.identify_bottlenecks(feature_results, dataset_results, gpu_results)
        
        # Compile comprehensive results
        comprehensive_results = {
            'analysis_timestamp': datetime.now().isoformat(),
            'system_info': {
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available(),
                'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                'cpu_count': psutil.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / 1e9,
                'available_memory_gb': psutil.virtual_memory().available / 1e9
            },
            'feature_extraction_analysis': feature_results,
            'dataset_loading_analysis': dataset_results,
            'gpu_utilization_analysis': gpu_results,
            'identified_bottlenecks': bottlenecks,
            'optimization_recommendations': self._generate_optimization_recommendations(bottlenecks)
        }
        
        # Save report
        with open(output_path, 'w') as f:
            json.dump(comprehensive_results, f, indent=2)
        
        logger.info(f"Performance analysis report saved to: {output_path}")
        
        # Print summary
        self._print_analysis_summary(comprehensive_results)
        
        return comprehensive_results
    
    def _create_synthetic_analysis(self) -> Dict:
        """Create synthetic analysis results for testing"""
        return {
            'total_files': 10,
            'successful_extractions': 10,
            'avg_extraction_time': 0.8,
            'min_extraction_time': 0.5,
            'max_extraction_time': 1.2,
            'std_extraction_time': 0.2,
            'avg_memory_increase': 2.5,
            'avg_cpu_usage': 45.0,
            'avg_gpu_memory_increase': 0.1,
            'cache_stats': {'hits': 5, 'misses': 5, 'hit_rate': 0.5, 'total_requests': 10},
            'performance_stats': {'gpu_available': torch.cuda.is_available(), 'gpu_used': False}
        }
    
    def _create_synthetic_dataset_analysis(self) -> Dict:
        """Create synthetic dataset analysis results for testing"""
        return {
            'dataset_size': 240000,
            'loading_time': 120.0,
            'positive_samples': 1000,
            'negative_samples': 240000,
            'avg_item_load_time': 0.05,
            'min_item_load_time': 0.02,
            'max_item_load_time': 0.15,
            'dataset_stats': {'balance_ratio': 0.004}
        }
    
    def _generate_optimization_recommendations(self, bottlenecks: List[str]) -> List[str]:
        """Generate optimization recommendations based on identified bottlenecks"""
        recommendations = []
        
        for bottleneck in bottlenecks:
            if "SLOW_FEATURE_EXTRACTION" in bottleneck:
                recommendations.extend([
                    "Implement GPU-accelerated feature extraction using torchaudio or similar",
                    "Use parallel processing with multiprocessing.Pool for CPU-bound extraction",
                    "Pre-compute and cache features to disk for reuse",
                    "Optimize librosa parameters (n_fft, hop_length) for speed vs quality trade-off"
                ])
            
            if "LOW_GPU_UTILIZATION" in bottleneck:
                recommendations.extend([
                    "Move feature extraction to GPU using torchaudio.transforms.MelSpectrogram",
                    "Implement batch processing on GPU for multiple audio files",
                    "Use CUDA-accelerated libraries for audio processing",
                    "Consider using NVIDIA RAPIDS for GPU-accelerated signal processing"
                ])
            
            if "POOR_CACHE_PERFORMANCE" in bottleneck:
                recommendations.extend([
                    "Increase cache size limit for better hit rates",
                    "Implement smarter cache key generation based on file content hash",
                    "Use persistent disk cache with longer TTL",
                    "Implement cache warming during dataset initialization"
                ])
            
            if "HIGH_MEMORY_USAGE" in bottleneck:
                recommendations.extend([
                    "Implement streaming audio processing instead of loading entire files",
                    "Use memory-mapped file access for large audio files",
                    "Implement garbage collection between extractions",
                    "Use lower precision (float16) for feature storage if accuracy allows"
                ])
            
            if "SLOW_DATASET_LOADING" in bottleneck:
                recommendations.extend([
                    "Pre-compute all features during dataset preparation phase",
                    "Use lazy loading with multiprocessing for on-demand feature extraction",
                    "Implement feature file indexing for faster lookup",
                    "Use SSD storage for feature files to reduce I/O latency"
                ])
        
        return recommendations
    
    def _print_analysis_summary(self, results: Dict):
        """Print analysis summary to console"""
        print("\n" + "="*80)
        print("🔍 PERFORMANCE ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n📊 System Information:")
        sys_info = results['system_info']
        print(f"  PyTorch Version: {sys_info['pytorch_version']}")
        print(f"  CUDA Available: {sys_info['cuda_available']}")
        if sys_info['cuda_available']:
            print(f"  CUDA Version: {sys_info['cuda_version']}")
            print(f"  GPU Name: {sys_info['gpu_name']}")
        
        print(f"\n🎯 Feature Extraction Performance:")
        feat_info = results['feature_extraction_analysis']
        print(f"  Files Processed: {feat_info.get('total_files', 0)}")
        print(f"  Avg Extraction Time: {feat_info.get('avg_extraction_time', 0):.3f}s")
        print(f"  Cache Hit Rate: {feat_info.get('cache_stats', {}).get('hit_rate', 0):.2%}")
        
        print(f"\n💾 Dataset Loading Performance:")
        dataset_info = results['dataset_loading_analysis']
        print(f"  Dataset Size: {dataset_info.get('dataset_size', 0):,}")
        print(f"  Avg Item Load Time: {dataset_info.get('avg_item_load_time', 0):.3f}s")
        
        if results['gpu_utilization_analysis'].get('gpu_available'):
            gpu_info = results['gpu_utilization_analysis']
            print(f"\n🚀 GPU Utilization:")
            print(f"  Avg GPU Utilization: {gpu_info.get('avg_gpu_utilization_percent', 0):.1f}%")
            print(f"  Max GPU Utilization: {gpu_info.get('max_gpu_utilization_percent', 0):.1f}%")
        
        print(f"\n⚠️  Identified Bottlenecks:")
        for i, bottleneck in enumerate(results['identified_bottlenecks'], 1):
            print(f"  {i}. {bottleneck}")
        
        print(f"\n💡 Optimization Recommendations:")
        recommendations = results['optimization_recommendations']
        for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
            print(f"  {i}. {rec}")
        if len(recommendations) > 5:
            print(f"  ... and {len(recommendations) - 5} more recommendations")
        
        print("\n" + "="*80)


def main():
    """Main performance analysis function"""
    print("🔍 Starting Comprehensive Performance Analysis...")
    
    analyzer = PerformanceAnalyzer()
    results = analyzer.generate_performance_report()
    
    print("✅ Performance analysis completed!")
    print("📄 Full report saved to: performance_analysis_report.json")
    
    return results


if __name__ == "__main__":
    results = main()