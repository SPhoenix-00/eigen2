"""
Memory Profiling Utilities for Debugging Memory Leaks
Provides detailed tracking of memory usage and object counts
"""

import gc
import sys
import psutil
import torch
import numpy as np
from typing import Dict, List, Any
from collections import defaultdict


class MemoryProfiler:
    """Tracks memory usage and provides detailed profiling information."""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None
        self.generation_snapshots = []

    def get_process_memory(self) -> Dict[str, float]:
        """Get current process memory usage in GB."""
        mem_info = self.process.memory_info()
        return {
            'rss_gb': mem_info.rss / 1e9,  # Resident Set Size (actual RAM used)
            'vms_gb': mem_info.vms / 1e9,  # Virtual Memory Size
        }

    def get_gpu_memory(self) -> Dict[str, float]:
        """Get GPU memory usage in GB."""
        if not torch.cuda.is_available():
            return {'allocated_gb': 0.0, 'reserved_gb': 0.0}

        return {
            'allocated_gb': torch.cuda.memory_allocated() / 1e9,
            'reserved_gb': torch.cuda.memory_reserved() / 1e9,
        }

    def count_objects_by_type(self) -> Dict[str, int]:
        """Count live objects by type."""
        # Force GC to get accurate counts
        gc.collect()

        type_counts = defaultdict(int)
        for obj in gc.get_objects():
            obj_type = type(obj).__name__
            type_counts[obj_type] += 1

        return dict(type_counts)

    def count_specific_objects(self) -> Dict[str, int]:
        """Count specific objects we care about for debugging."""
        gc.collect()

        from environment.trading_env import TradingEnvironment
        from models.ddpg_agent import DDPGAgent
        from models.replay_buffer import ReplayBuffer

        counts = {
            'TradingEnvironment': 0,
            'DDPGAgent': 0,
            'ReplayBuffer': 0,
            'list': 0,
            'dict': 0,
            'ndarray': 0,
            'Tensor': 0,
        }

        for obj in gc.get_objects():
            obj_type = type(obj)
            if obj_type.__name__ == 'TradingEnvironment':
                counts['TradingEnvironment'] += 1
            elif obj_type.__name__ == 'DDPGAgent':
                counts['DDPGAgent'] += 1
            elif obj_type.__name__ == 'ReplayBuffer':
                counts['ReplayBuffer'] += 1
            elif obj_type is list:
                counts['list'] += 1
            elif obj_type is dict:
                counts['dict'] += 1
            elif obj_type.__name__ == 'ndarray':
                counts['ndarray'] += 1
            elif obj_type.__name__ == 'Tensor':
                counts['Tensor'] += 1

        return counts

    def get_large_objects(self, min_size_mb: float = 10.0) -> List[Dict[str, Any]]:
        """Find large objects in memory (>= min_size_mb)."""
        gc.collect()

        large_objects = []
        for obj in gc.get_objects():
            try:
                size = sys.getsizeof(obj)
                size_mb = size / (1024 * 1024)

                if size_mb >= min_size_mb:
                    large_objects.append({
                        'type': type(obj).__name__,
                        'size_mb': size_mb,
                        'id': id(obj),
                    })
            except:
                # Some objects don't support getsizeof
                pass

        # Sort by size descending
        large_objects.sort(key=lambda x: x['size_mb'], reverse=True)
        return large_objects[:20]  # Top 20

    def take_snapshot(self, label: str) -> Dict[str, Any]:
        """Take a memory snapshot with a label."""
        snapshot = {
            'label': label,
            'process_memory': self.get_process_memory(),
            'gpu_memory': self.get_gpu_memory(),
            'object_counts': self.count_specific_objects(),
        }

        self.generation_snapshots.append(snapshot)
        return snapshot

    def print_snapshot(self, snapshot: Dict[str, Any], show_objects: bool = True):
        """Print a formatted memory snapshot."""
        print(f"\n{'='*70}")
        print(f"🔍 MEMORY SNAPSHOT: {snapshot['label']}")
        print(f"{'='*70}")

        # Process memory
        proc_mem = snapshot['process_memory']
        print(f"💾 Process Memory:")
        print(f"   RSS (actual RAM):  {proc_mem['rss_gb']:>8.2f} GB")
        print(f"   VMS (virtual):     {proc_mem['vms_gb']:>8.2f} GB")

        # GPU memory
        gpu_mem = snapshot['gpu_memory']
        if gpu_mem['allocated_gb'] > 0:
            print(f"🎮 GPU Memory:")
            print(f"   Allocated:         {gpu_mem['allocated_gb']:>8.2f} GB")
            print(f"   Reserved:          {gpu_mem['reserved_gb']:>8.2f} GB")

        # Object counts
        if show_objects:
            obj_counts = snapshot['object_counts']
            print(f"📊 Object Counts:")
            print(f"   TradingEnvironment: {obj_counts['TradingEnvironment']:>6}")
            print(f"   DDPGAgent:          {obj_counts['DDPGAgent']:>6}")
            print(f"   ReplayBuffer:       {obj_counts['ReplayBuffer']:>6}")
            print(f"   Lists:              {obj_counts['list']:>6}")
            print(f"   Dicts:              {obj_counts['dict']:>6}")
            print(f"   NumPy arrays:       {obj_counts['ndarray']:>6}")
            print(f"   PyTorch Tensors:    {obj_counts['Tensor']:>6}")

        print(f"{'='*70}\n")

    def print_memory_growth(self, baseline_label: str = None):
        """Print memory growth since baseline or first snapshot."""
        if len(self.generation_snapshots) < 2:
            print("⚠️  Need at least 2 snapshots to show growth")
            return

        # Find baseline
        if baseline_label:
            baseline = next((s for s in self.generation_snapshots if s['label'] == baseline_label), None)
            if not baseline:
                baseline = self.generation_snapshots[0]
        else:
            baseline = self.generation_snapshots[0]

        current = self.generation_snapshots[-1]

        # Calculate growth
        baseline_rss = baseline['process_memory']['rss_gb']
        current_rss = current['process_memory']['rss_gb']
        growth_gb = current_rss - baseline_rss
        growth_pct = (growth_gb / baseline_rss) * 100 if baseline_rss > 0 else 0

        print(f"\n{'='*70}")
        print(f"📈 MEMORY GROWTH ANALYSIS")
        print(f"{'='*70}")
        print(f"Baseline: {baseline['label']}")
        print(f"Current:  {current['label']}")
        print(f"")
        print(f"RAM Usage:")
        print(f"   Baseline:  {baseline_rss:>8.2f} GB")
        print(f"   Current:   {current_rss:>8.2f} GB")
        print(f"   Growth:    {growth_gb:>+8.2f} GB ({growth_pct:+.1f}%)")

        # Object count changes
        baseline_objs = baseline['object_counts']
        current_objs = current['object_counts']

        print(f"")
        print(f"Object Count Changes:")
        for obj_type in ['TradingEnvironment', 'DDPGAgent', 'ReplayBuffer', 'ndarray', 'Tensor']:
            baseline_count = baseline_objs.get(obj_type, 0)
            current_count = current_objs.get(obj_type, 0)
            change = current_count - baseline_count

            if abs(change) > 0:
                sign = '+' if change > 0 else ''
                print(f"   {obj_type:20s}: {baseline_count:6} → {current_count:6} ({sign}{change})")

        print(f"{'='*70}\n")

    def print_large_objects(self, min_size_mb: float = 10.0):
        """Print information about large objects in memory."""
        print(f"\n{'='*70}")
        print(f"🐘 LARGE OBJECTS (>= {min_size_mb} MB)")
        print(f"{'='*70}")

        large_objs = self.get_large_objects(min_size_mb)

        if not large_objs:
            print(f"No objects >= {min_size_mb} MB found")
        else:
            print(f"{'Type':<30} {'Size (MB)':>15}")
            print(f"{'-'*30} {'-'*15}")
            for obj in large_objs:
                print(f"{obj['type']:<30} {obj['size_mb']:>15.2f}")

        print(f"{'='*70}\n")

    def print_generation_trend(self):
        """Print memory trend across generations."""
        if len(self.generation_snapshots) < 2:
            return

        print(f"\n{'='*70}")
        print(f"📊 MEMORY TREND ACROSS GENERATIONS")
        print(f"{'='*70}")
        print(f"{'Label':<30} {'RAM (GB)':>12} {'Change':>12}")
        print(f"{'-'*30} {'-'*12} {'-'*12}")

        prev_rss = None
        for snapshot in self.generation_snapshots:
            rss = snapshot['process_memory']['rss_gb']

            if prev_rss is None:
                change_str = "baseline"
            else:
                change = rss - prev_rss
                change_str = f"{change:+.2f} GB"

            print(f"{snapshot['label']:<30} {rss:>12.2f} {change_str:>12}")
            prev_rss = rss

        print(f"{'='*70}\n")


# Global profiler instance
_profiler = None


def get_profiler() -> MemoryProfiler:
    """Get or create global profiler instance."""
    global _profiler
    if _profiler is None:
        _profiler = MemoryProfiler()
    return _profiler


def log_memory(label: str, show_objects: bool = True):
    """Convenience function to log memory at a point."""
    profiler = get_profiler()
    snapshot = profiler.take_snapshot(label)
    profiler.print_snapshot(snapshot, show_objects=show_objects)
    return snapshot


def print_memory_summary():
    """Print comprehensive memory summary."""
    profiler = get_profiler()
    profiler.print_generation_trend()
    profiler.print_memory_growth()
    profiler.print_large_objects(min_size_mb=50.0)
