"""
Display utilities for Project Eigen 2
Pretty printing and progress visualization
"""

import os
import numpy as np
from typing import List, Optional
from utils.config import Config
import psutil
import torch
import shutil


class ResourceTracker:
    """Tracks system resource usage throughout training."""

    def __init__(self, disk_path: str = "/workspace"):
        """
        Initialize resource tracker.

        Args:
            disk_path: Path to monitor for disk usage (default: /workspace for RunPod)
        """
        self.disk_path = disk_path
        # Fall back to current directory if /workspace doesn't exist
        if not os.path.exists(disk_path):
            self.disk_path = os.getcwd()

        self.peak_vram_gb = 0.0
        self.peak_ram_gb = 0.0
        self.peak_disk_gb = 0.0

    def update(self):
        """Update peak resource usage statistics."""
        # Track VRAM (GPU memory)
        if torch.cuda.is_available():
            vram_used = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
            self.peak_vram_gb = max(self.peak_vram_gb, vram_used)

        # Track system RAM
        ram_info = psutil.virtual_memory()
        ram_used_gb = ram_info.used / (1024**3)  # Convert to GB
        self.peak_ram_gb = max(self.peak_ram_gb, ram_used_gb)

        # Track disk usage
        disk_info = shutil.disk_usage(self.disk_path)
        disk_used_gb = disk_info.used / (1024**3)  # Convert to GB
        self.peak_disk_gb = max(self.peak_disk_gb, disk_used_gb)

    def get_current_stats(self) -> dict:
        """Get current resource usage statistics."""
        stats = {
            'peak_vram_gb': self.peak_vram_gb,
            'peak_ram_gb': self.peak_ram_gb,
            'peak_disk_gb': self.peak_disk_gb,
        }

        # Add current values for reference
        if torch.cuda.is_available():
            stats['current_vram_gb'] = torch.cuda.memory_allocated() / (1024**3)

        ram_info = psutil.virtual_memory()
        stats['current_ram_gb'] = ram_info.used / (1024**3)

        disk_info = shutil.disk_usage(self.disk_path)
        stats['current_disk_gb'] = disk_info.used / (1024**3)

        return stats

    def reset_peaks(self):
        """Reset peak tracking (useful for per-generation tracking)."""
        self.peak_vram_gb = 0.0
        self.peak_ram_gb = 0.0
        self.peak_disk_gb = 0.0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def plot_fitness_progress(fitness_history: List[List[float]]):
    """
    Create a simple ASCII plot of fitness over generations.
    
    Args:
        fitness_history: List of fitness scores per generation
    """
    if not fitness_history:
        return
    
    generations = len(fitness_history)
    max_fitness = [max(gen) for gen in fitness_history]
    mean_fitness = [np.mean(gen) for gen in fitness_history]
    
    print("\n" + "="*60)
    print("Fitness Progress Over Generations")
    print("="*60)
    
    # Normalize for plotting (0-20 scale)
    all_values = max_fitness + mean_fitness
    min_val = min(all_values)
    max_val = max(all_values)
    
    if max_val == min_val:
        # All same value
        scale = lambda x: 10
    else:
        scale = lambda x: int(((x - min_val) / (max_val - min_val)) * 20)
    
    # Plot
    for gen in range(generations):
        max_bar = '█' * scale(max_fitness[gen])
        mean_bar = '▓' * scale(mean_fitness[gen])
        
        print(f"Gen {gen+1:2d} | Max:  {max_bar:<20} {max_fitness[gen]:>8.1f}")
        print(f"      | Mean: {mean_bar:<20} {mean_fitness[gen]:>8.1f}")
        print()
    
    print(f"Value range: [{min_val:.1f}, {max_val:.1f}]")
    print("="*60)


def _print_progress_and_eta(gen: int, total_gens: int, avg_gen_time: float, gauntlet_info: Optional[dict] = None):
    """
    Print progress and ETA information based on training mode.

    Args:
        gen: Current generation number
        total_gens: Total number of generations (fallback limit)
        avg_gen_time: Average time per generation
        gauntlet_info: Optional gauntlet/consistency mode information
    """
    remaining = total_gens - (gen + 1)

    # Check if we're in gauntlet or consistency mode
    if gauntlet_info and gauntlet_info.get('gauntlet_enabled'):
        consistency_mode = gauntlet_info.get('consistency_mode', False)

        if consistency_mode:
            # Consistency mode: progress based on HoF turnovers
            current_turnovers = gauntlet_info.get('hof_turnover_count', 0)
            target_turnovers = gauntlet_info.get('target_hof_turnovers', 2)

            print(f"\n  🎯 CONSISTENCY PROGRESS")
            print(f"  HoF Turnovers:     {current_turnovers:>12} / {target_turnovers}")

            if current_turnovers < target_turnovers:
                turnovers_needed = target_turnovers - current_turnovers
                print(f"  Turnovers Needed:  {turnovers_needed:>12}")
                print(f"  Est. ETA:          {'Unknown':>11}  (depends on performance)")
            else:
                print(f"  Status:            {'🏆 Complete!':>11}")

            # Show fallback generation limit
            print(f"\n  Fallback Limit:    {remaining:>12} generations remaining")
            if remaining > 0:
                eta_seconds = avg_gen_time * remaining
                eta_minutes = eta_seconds / 60
                eta_hours = eta_seconds / 3600
                print(f"  Max Time Left:     {eta_minutes:>11.1f}m  ({eta_hours:.1f}h)")
        else:
            # Normal gauntlet mode: progress based on breakthroughs
            current_breakthroughs = gauntlet_info.get('confirmed_breakthroughs', 0)
            target_breakthroughs = gauntlet_info.get('target_breakthroughs', 4)

            print(f"\n  🎯 BREAKTHROUGH PROGRESS")
            print(f"  Confirmed:         {current_breakthroughs:>12} / {target_breakthroughs}")

            if current_breakthroughs < target_breakthroughs:
                breakthroughs_needed = target_breakthroughs - current_breakthroughs
                print(f"  Needed:            {breakthroughs_needed:>12}")

                # Estimate ETA based on breakthrough history if available
                history = gauntlet_info.get('breakthrough_history', [])
                if len(history) >= 2:
                    # Calculate average generations between breakthroughs
                    gen_diffs = []
                    for i in range(1, len(history)):
                        gen_diffs.append(history[i].get('generation', 0) - history[i-1].get('generation', 0))
                    avg_gens_per_breakthrough = np.mean(gen_diffs)

                    est_gens_remaining = avg_gens_per_breakthrough * breakthroughs_needed
                    est_time_seconds = est_gens_remaining * avg_gen_time
                    est_time_minutes = est_time_seconds / 60
                    est_time_hours = est_time_seconds / 3600

                    print(f"  Est. Gens Left:    {est_gens_remaining:>11.1f}  (based on {len(history)} breakthroughs)")
                    print(f"  Est. ETA:          {est_time_minutes:>11.1f}m  ({est_time_hours:.1f}h)")
                else:
                    print(f"  Est. ETA:          {'Unknown':>11}  (insufficient data)")
            else:
                print(f"  Status:            {'🏆 Complete!':>11}")

            # Show fallback generation limit
            print(f"\n  Fallback Limit:    {remaining:>12} generations remaining")
            if remaining > 0:
                eta_seconds = avg_gen_time * remaining
                eta_minutes = eta_seconds / 60
                eta_hours = eta_seconds / 3600
                print(f"  Max Time Left:     {eta_minutes:>11.1f}m  ({eta_hours:.1f}h)")
    else:
        # Traditional fixed-generation mode
        print(f"\n  Remaining Gens:    {remaining:>12}")

        # Only show ETA if there are remaining generations
        if remaining > 0:
            eta_seconds = avg_gen_time * remaining
            eta_minutes = eta_seconds / 60
            eta_hours = eta_seconds / 3600
            print(f"  ETA:               {eta_minutes:>11.1f}m  ({eta_hours:.1f}h)")
        else:
            print(f"  ETA:               {'Complete':>11}  (0.0h)")


def print_generation_summary(gen: int, total_gens: int,
                             fitness_scores: List[float],
                             pop_stats: dict,
                             buffer_size: int,
                             best_fitness: float,
                             gen_time: float,
                             avg_gen_time: float,
                             resource_stats: Optional[dict] = None,
                             gauntlet_info: Optional[dict] = None):
    """
    Print a comprehensive summary of the generation.

    Args:
        gen: Current generation number
        total_gens: Total number of generations
        fitness_scores: Fitness scores for this generation
        pop_stats: Population statistics dictionary
        buffer_size: Current replay buffer size
        best_fitness: Best fitness ever achieved
        gen_time: Time taken for this generation
        avg_gen_time: Average time per generation
        resource_stats: Optional dictionary with resource usage statistics
        gauntlet_info: Optional dictionary with gauntlet/consistency mode info:
            - gauntlet_enabled: bool
            - consistency_mode: bool
            - breakthrough_state: str (e.g., 'NORMAL', 'STABILIZATION', 'GAUNTLET')
            - confirmed_baseline: float
            - confirmed_breakthroughs: int
            - target_breakthroughs: int
            - hof_turnover_count: int
            - target_hof_turnovers: int
            - hof_current_median: float or None
            - hof_size: int (current number of agents in Hall of Fame)
            - hof_capacity: int (maximum Hall of Fame size)
            - queue_size: int or None (number of candidates in queue, heroes mode only)
            - stabilization_progress: tuple (current, total) or None
            - breakthrough_history: list of breakthrough events
    """
    mean_fitness = np.mean(fitness_scores)
    max_fitness = np.max(fitness_scores)
    min_fitness = np.min(fitness_scores)
    
    print("\n" + "="*70)
    print(f"{'GENERATION ' + str(gen+1) + ' / ' + str(total_gens):^70}")
    print("="*70)
    
    # Fitness section
    print("\n📊 FITNESS METRICS")
    print("-" * 70)
    print(f"  Mean:              {mean_fitness:>12.2f}")
    print(f"  Maximum:           {max_fitness:>12.2f}  {'🌟 NEW BEST!' if max_fitness > best_fitness else ''}")
    print(f"  Minimum:           {min_fitness:>12.2f}")
    print(f"  Std Dev:           {np.std(fitness_scores):>12.2f}")
    print(f"  Positive Agents:   {pop_stats['agents_with_positive_fitness']:>12} / {len(fitness_scores)}")
    
    # Fitness distribution
    print(f"\n  Distribution: ", end="")
    for f in sorted(fitness_scores, reverse=True):
        if f > 0:
            print("█", end="")
        elif f > -100:
            print("▓", end="")
        else:
            print("░", end="")
    print()

    # Gauntlet State Machine section (if in gauntlet mode)
    if gauntlet_info and gauntlet_info.get('gauntlet_enabled'):
        print("\n🎮 GAUNTLET STATE")
        print("-" * 70)

        state = gauntlet_info.get('breakthrough_state', 'UNKNOWN')
        confirmed_baseline = gauntlet_info.get('confirmed_baseline', 0.0)
        consistency_mode = gauntlet_info.get('consistency_mode', False)

        # State indicator with visual progress
        state_icons = {
            'NORMAL': '🔵 NORMAL',
            'DETECTION': '🟡 DETECTION',
            'STABILIZATION': '🟠 STABILIZATION',
            'GAUNTLET': '🔴 GAUNTLET',
            'CONFIRMED': '🟢 CONFIRMED',
            'REJECTED': '⚫ REJECTED'
        }
        print(f"  Current State:     {state_icons.get(state, state):>12}")
        print(f"  Baseline:          {confirmed_baseline:>12.2f}")

        # State-specific progress information
        if state == 'STABILIZATION':
            stab_progress = gauntlet_info.get('stabilization_progress')
            if stab_progress:
                current, total = stab_progress
                progress_bar = '█' * current + '░' * (total - current)
                print(f"  Stab Progress:     [{progress_bar}] {current}/{total}")

        # Consistency mode: show HoF turnover info
        if consistency_mode:
            current_turnovers = gauntlet_info.get('hof_turnover_count', 0)
            target_turnovers = gauntlet_info.get('target_hof_turnovers', 2)
            hof_median = gauntlet_info.get('hof_current_median')
            hof_size = gauntlet_info.get('hof_size', 0)
            hof_capacity = gauntlet_info.get('hof_capacity', 10)

            print(f"  HoF Size:          {hof_size:>12} / {hof_capacity}")
            print(f"  HoF Turnovers:     {current_turnovers:>12} / {target_turnovers}")
            if hof_median is not None:
                print(f"  HoF Median ROI:    {hof_median:>11.2f}%")

            # Show queue size if using candidate queue (heroes mode)
            queue_size = gauntlet_info.get('queue_size')
            if queue_size is not None:
                print(f"  Queue Size:        {queue_size:>12}")
        else:
            # Normal gauntlet mode: show breakthrough count
            current_breakthroughs = gauntlet_info.get('confirmed_breakthroughs', 0)
            target_breakthroughs = gauntlet_info.get('target_breakthroughs', 4)
            hof_size = gauntlet_info.get('hof_size', 0)
            hof_capacity = gauntlet_info.get('hof_capacity', 10)

            print(f"  Breakthroughs:     {current_breakthroughs:>12} / {target_breakthroughs}")
            print(f"  HoF Size:          {hof_size:>12} / {hof_capacity}")

            # Show queue size if using candidate queue (heroes mode)
            queue_size = gauntlet_info.get('queue_size')
            if queue_size is not None:
                print(f"  Queue Size:        {queue_size:>12}")

        # Global Hall of Fame section (if enabled)
        global_hof_enabled = gauntlet_info.get('global_hof_enabled', False)
        if global_hof_enabled:
            global_hof_size = gauntlet_info.get('global_hof_size', 0)
            global_hof_threshold = gauntlet_info.get('global_hof_threshold', float('-inf'))

            print("\n🌍 GLOBAL HALL OF FAME")
            print("-" * 70)
            print(f"  Status:            {'ENABLED ✓' if global_hof_enabled else 'DISABLED'}")
            print(f"  Current Size:      {global_hof_size:>12} / 50")
            if global_hof_threshold != float('-inf'):
                print(f"  Entry Threshold:   {global_hof_threshold:>12.2f}  (Rank #50)")
            else:
                print(f"  Entry Threshold:   {'None (Open)':>12}")

    # Trading section
    print("\n📈 TRADING ACTIVITY")
    print("-" * 70)
    print(f"  Total Trades:      {pop_stats['total_trades']:>12}")
    print(f"  Avg per Agent:     {pop_stats['avg_trades_per_agent']:>12.1f}")
    print(f"  Wins:              {pop_stats['total_wins']:>12}  ({pop_stats['total_wins']/max(pop_stats['total_trades'],1)*100:.1f}%)")
    print(f"  Losses:            {pop_stats['total_losses']:>12}  ({pop_stats['total_losses']/max(pop_stats['total_trades'],1)*100:.1f}%)")
    print(f"  Avg Win Rate:      {pop_stats['avg_win_rate']:>11.1%}")
    
    # System section
    print("\n⚙️  SYSTEM STATUS")
    print("-" * 70)
    print(f"  Replay Buffer:     {buffer_size:>12,} / {Config.BUFFER_SIZE:,}")
    # Show correct minimum based on sweep vs regular training
    is_sweep = os.environ.get("WANDB_SWEEP_ID") is not None
    min_size = Config.MIN_BUFFER_SIZE_SWEEP if is_sweep else Config.MIN_BUFFER_SIZE
    print(f"  Buffer Ready:      {' '*10}{'✓ Yes' if buffer_size >= min_size else '✗ No (needs ' + str(min_size-buffer_size) + ' more)'}")
    print(f"  Generation Time:   {gen_time:>11.1f}s")
    print(f"  Avg Gen Time:      {avg_gen_time:>11.1f}s")

    # Progress and ETA - context-aware based on training mode
    _print_progress_and_eta(gen, total_gens, avg_gen_time, gauntlet_info)

    # Add one-line resource summary if provided
    if resource_stats:
        print("\n💻 RESOURCE USAGE")
        print("-" * 70)
        print(f"  Peak VRAM: {resource_stats['peak_vram_gb']:.1f}GB  |  "
              f"Peak RAM: {resource_stats['peak_ram_gb']:.1f}GB  |  "
              f"Peak Disk: {resource_stats['peak_disk_gb']:.1f}GB  |  "
              f"Avg Gen Time: {avg_gen_time:.1f}s")

    print("\n" + "="*70)


def visualize_gauntlet_slices(fitness_scores: List[float], mean_score: float, min_score: float,
                               max_score: float, gauntlet_score: float):
    """
    Visualize gauntlet slice scores with a detailed breakdown.

    Shows:
    - Bar chart of all 20 slice scores
    - Statistical metrics (std dev, coefficient of variation)
    - Identification of problematic slices
    - Score distribution

    Args:
        fitness_scores: List of fitness scores for each of the 20 slices
        mean_score: Mean fitness across all slices
        min_score: Minimum fitness score
        max_score: Maximum fitness score
        gauntlet_score: Final aggregated gauntlet score (0.5*mean + 0.5*min)
    """
    print("\n" + "="*70)
    print(f"{'GAUNTLET SLICE ANALYSIS':^70}")
    print("="*70)

    # Statistical analysis
    std_dev = np.std(fitness_scores)
    # Coefficient of Variation: measure of volatility (std dev / mean)
    # Higher CV = more volatile performance across slices
    cv = (std_dev / abs(mean_score)) * 100 if abs(mean_score) > 0.001 else 0

    print("\n📊 STATISTICAL SUMMARY")
    print("-" * 70)
    print(f"  Gauntlet Score:    {gauntlet_score:>12.2f}  (0.5*mean + 0.5*min)")
    print(f"  Mean:              {mean_score:>12.2f}")
    print(f"  Min:               {min_score:>12.2f}")
    print(f"  Max:               {max_score:>12.2f}")
    print(f"  Std Dev:           {std_dev:>12.2f}")
    print(f"  Coefficient of Variation: {cv:>8.1f}%  {'⚠️  High volatility!' if cv > 50 else '✓ Stable' if cv < 25 else '~ Moderate'}")
    print(f"  Range:             {max_score - min_score:>12.2f}")

    # Identify problematic slices (below mean - 0.5*std_dev)
    threshold = mean_score - 0.5 * std_dev
    problematic_slices = [(i, score) for i, score in enumerate(fitness_scores) if score < threshold]

    if problematic_slices:
        print(f"\n⚠️  PROBLEMATIC SLICES (below {threshold:.2f})")
        print("-" * 70)
        for slice_idx, score in sorted(problematic_slices, key=lambda x: x[1]):
            deviation = score - mean_score
            print(f"  Slice #{slice_idx+1:2d}:          {score:>12.2f}  ({deviation:+.2f} from mean)")

    # Bar chart visualization
    print("\n📊 SLICE SCORE DISTRIBUTION")
    print("-" * 70)

    # Determine scale for visualization (normalize to 0-40 character width)
    score_range = max_score - min_score
    if score_range < 0.01:
        # All scores are essentially the same
        scale_func = lambda x: 20
    else:
        scale_func = lambda x: int(((x - min_score) / score_range) * 40)

    # Sort slices by score for better visualization
    sorted_slices = sorted(enumerate(fitness_scores), key=lambda x: x[1], reverse=True)

    # Show bar chart
    for slice_idx, score in sorted_slices:
        bar_length = scale_func(score)
        bar = '█' * bar_length

        # Color code based on performance
        if score >= mean_score:
            marker = '✓'
        elif score < threshold:
            marker = '⚠'
        else:
            marker = '·'

        deviation = score - mean_score
        print(f"  {marker} Slice #{slice_idx+1:2d}  {bar:<40}  {score:>8.2f}  ({deviation:+.2f})")

    # Add reference lines
    print(f"\n  {'Reference Lines':}")
    mean_pos = scale_func(mean_score)
    mean_line = ' ' * mean_pos + '↑'
    print(f"    Mean:      {mean_line} {mean_score:.2f}")

    if min_score != max_score:
        min_line = '↑'
        max_pos = scale_func(max_score)
        max_line = ' ' * max_pos + '↑'
        print(f"    Min:       {min_line} {min_score:.2f}")
        print(f"    Max:       {max_line} {max_score:.2f}")

    # Quartile analysis
    q1 = np.percentile(fitness_scores, 25)
    q2 = np.percentile(fitness_scores, 50)  # median
    q3 = np.percentile(fitness_scores, 75)

    print(f"\n📈 QUARTILE BREAKDOWN")
    print("-" * 70)
    print(f"  Q1 (25th percentile):  {q1:>12.2f}")
    print(f"  Q2 (50th / median):    {q2:>12.2f}")
    print(f"  Q3 (75th percentile):  {q3:>12.2f}")
    print(f"  IQR (Q3-Q1):           {q3-q1:>12.2f}")

    # Performance consistency rating
    print(f"\n🎯 PERFORMANCE RATING")
    print("-" * 70)
    if cv < 15:
        rating = "EXCELLENT"
        emoji = "🌟"
        desc = "Very consistent across all slices"
    elif cv < 25:
        rating = "GOOD"
        emoji = "✓"
        desc = "Stable performance with minor variations"
    elif cv < 40:
        rating = "MODERATE"
        emoji = "~"
        desc = "Some volatility, improvement possible"
    elif cv < 60:
        rating = "POOR"
        emoji = "⚠️"
        desc = "High volatility, inconsistent performance"
    else:
        rating = "VERY POOR"
        emoji = "❌"
        desc = "Extremely volatile, major consistency issues"

    print(f"  Consistency:       {emoji} {rating}")
    print(f"  Assessment:        {desc}")

    # Analyze if one slice is ruining the score
    if len(problematic_slices) > 0:
        worst_score = min(fitness_scores)
        score_without_worst = [s for s in fitness_scores if s != worst_score or fitness_scores.count(worst_score) > 1]
        if score_without_worst:
            mean_without_worst = np.mean(score_without_worst)
            impact = mean_without_worst - mean_score
            print(f"\n💡 WORST SLICE IMPACT")
            print("-" * 70)
            print(f"  Mean without worst slice:  {mean_without_worst:>12.2f}")
            print(f"  Impact on mean:            {impact:>12.2f}  ({impact/abs(mean_score)*100:+.1f}%)" if abs(mean_score) > 0.001 else "  Impact on mean:            N/A")

            if abs(impact) > std_dev:
                print(f"  ⚠️  Single slice is disproportionately affecting the score!")

    print("\n" + "="*70)


def print_final_summary(trainer):
    """
    Print final training summary.

    Args:
        trainer: ERLTrainer instance
    """
    print("\n" + "="*70)
    print(f"{'TRAINING COMPLETE':^70}")
    print("="*70)

    print("\n🏆 BEST RESULTS")
    print("-" * 70)
    print(f"  Best Training Fitness:     {trainer.best_fitness:>12.2f}")
    print(f"  Best Validation Fitness:   {trainer.best_validation_fitness:>12.2f}")
    print(f"  Total Generations: {len(trainer.fitness_history):>12}")
    print(f"  Total Transitions: {trainer.replay_buffer.total_added:>12,}")
    print(f"  Avg Gen Time:      {np.mean(trainer.generation_times):>11.1f}s")
    print(f"  Total Time:        {sum(trainer.generation_times)/60:>11.1f}m")

    # Fitness improvement
    if len(trainer.fitness_history) > 1:
        first_gen_max = max(trainer.fitness_history[0])
        last_gen_max = max(trainer.fitness_history[-1])
        improvement = last_gen_max - first_gen_max

        print(f"\n📈 IMPROVEMENT")
        print("-" * 70)
        print(f"  First Gen Max:     {first_gen_max:>12.2f}")
        print(f"  Last Gen Max:      {last_gen_max:>12.2f}")
        perc_str = f"({improvement/abs(first_gen_max)*100:+.1f}%)" if abs(first_gen_max) > 0 else "(N/A)"
        print(f"  Improvement:       {improvement:>12.2f}  {perc_str}")

    print("\n📁 OUTPUT FILES")
    print("-" * 70)
    print(f"  Checkpoints:       checkpoints/")
    print(f"  TensorBoard Logs:  logs/")
    print(f"\n  View logs with:    tensorboard --logdir=logs")

    print("\n" + "="*70)