"""
Display utilities for Project Eigen 2
Pretty printing, progress visualization, and structured logging
"""

import os
import sys
import numpy as np
from typing import List, Optional, Dict
from utils.config import Config
import psutil
import torch
import shutil


# ============================================================
# Verbosity Levels
# ============================================================
QUIET = 0    # Errors/warnings only
NORMAL = 1   # Dashboard + key events
VERBOSE = 2  # Full detail (legacy behavior)

_verbosity = NORMAL


def set_verbosity(level: int):
    """Set the global verbosity level."""
    global _verbosity
    _verbosity = level


def get_verbosity() -> int:
    """Get the current verbosity level."""
    return _verbosity


def log(msg: str = "", level: int = NORMAL):
    """
    Print to console if level <= verbosity, always write to log file.
    
    Args:
        msg: Message to print
        level: Verbosity level required for console output
    """
    if level <= _verbosity:
        print(msg)
    else:
        # Write to log file only (if TeeLogger is active)
        stdout = sys.stdout
        if hasattr(stdout, 'log_file') and not stdout.log_file.closed:
            stdout.log_file.write(msg + '\n')
            stdout.log_file.flush()


def log_event(msg: str):
    """Print a highlighted event line (always shown at NORMAL+)."""
    log(f"  >>> {msg}", NORMAL)


# ============================================================
# Generation Tracker for Trend Display
# ============================================================
class GenerationTracker:
    """Tracks metrics across generations for trend display."""
    
    def __init__(self):
        self.prev: Dict[str, float] = {}
        self.best_ever: Dict[str, float] = {}
    
    def update(self, metrics: dict) -> dict:
        """
        Store current metrics and return deltas vs previous generation.
        
        Args:
            metrics: Current generation's metrics
            
        Returns:
            dict of deltas (key -> delta value)
        """
        deltas = {}
        for key, val in metrics.items():
            if isinstance(val, (int, float)) and not np.isnan(val) and not np.isinf(val):
                if key in self.prev:
                    deltas[key] = val - self.prev[key]
                # Track all-time bests (higher is better for most metrics)
                if key not in self.best_ever or val > self.best_ever[key]:
                    self.best_ever[key] = val
        self.prev = {k: v for k, v in metrics.items() 
                     if isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v)}
        return deltas
    
    def format_delta(self, key: str, deltas: dict, fmt: str = ".2f", 
                     suffix: str = "", invert: bool = False) -> str:
        """
        Format a delta value with arrow indicator.
        
        Args:
            key: Metric key
            deltas: Dict of deltas from update()
            fmt: Format string for the number
            suffix: Suffix like '%' or '$'
            invert: If True, negative is good (e.g., for losses)
        """
        if key not in deltas:
            return ""
        d = deltas[key]
        if abs(d) < 0.001:
            return f"  ={suffix}"
        sign = "+" if d > 0 else ""
        return f"  {sign}{d:{fmt}}{suffix}"


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


def _fmt_time(seconds: float) -> str:
    """Format seconds into a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def print_generation_dashboard(gen: int, total_gens: int,
                               fitness_scores: List[float],
                               buffer_size: int,
                               gen_time: float,
                               avg_gen_time: float,
                               best_agent_info: Optional[dict] = None,
                               population_info: Optional[dict] = None,
                               hof_info: Optional[dict] = None,
                               gauntlet_info: Optional[dict] = None,
                               timing_info: Optional[dict] = None,
                               resource_stats: Optional[dict] = None,
                               deltas: Optional[dict] = None,
                               local_mode: bool = False,
                               events: Optional[List[str]] = None):
    """
    Print a structured per-generation dashboard.
    
    This replaces the old print_generation_summary and the scattered prints
    throughout the generation loop. All information is consolidated here.
    
    Args:
        gen: Current generation (0-indexed)
        total_gens: Fallback generation limit
        fitness_scores: Population fitness scores
        buffer_size: Current replay buffer size
        gen_time: Time for this generation
        avg_gen_time: Average generation time
        best_agent_info: Dict with best agent's trading profile:
            - combined_fitness, roi, win_rate, num_wins, num_losses,
              quality_ratio, quality_count, total_trades, expectancy, pnl
        population_info: Dict with population health:
            - positive_count, mean_roi, mean_win_rate
        hof_info: Dict with Hall of Fame state:
            - size, capacity, best, worst, median_roi, roi_hurdle_ema
        gauntlet_info: Dict with gauntlet state (same as before)
        timing_info: Dict with phase breakdown:
            - eval_time, val_time, train_time, evolve_time
            - train_compute_pct, train_data_load_pct, train_gpu_transfer_pct
        resource_stats: Dict with resource usage
        deltas: Dict of metric deltas from GenerationTracker.update()
        local_mode: Whether running in local mode
        events: List of event strings to highlight
    """
    deltas = deltas or {}
    
    def _delta(key, fmt=".2f", suffix="", pct=False):
        """Format delta for a metric key."""
        if key not in deltas:
            return ""
        d = deltas[key]
        if abs(d) < 0.001:
            return ""
        sign = "+" if d > 0 else ""
        if pct:
            return f"  {sign}{d:{fmt}}%"
        return f"  {sign}{d:{fmt}}{suffix}"

    # ---- Header ----
    header_parts = [f"Gen {gen+1}/{total_gens}"]
    
    if gauntlet_info and gauntlet_info.get('gauntlet_enabled'):
        consistency_mode = gauntlet_info.get('consistency_mode', False)
        if consistency_mode:
            turnovers = gauntlet_info.get('hof_turnover_count', 0)
            target = gauntlet_info.get('target_hof_turnovers', 3)
            runway = gauntlet_info.get('runway_remaining', '?')
            header_parts.append(f"Turnover {turnovers}/{target}")
            header_parts.append(f"Runway {runway}")
        else:
            bt = gauntlet_info.get('confirmed_breakthroughs', 0)
            bt_target = gauntlet_info.get('target_breakthroughs', 4)
            header_parts.append(f"BT {bt}/{bt_target}")
        
        state = gauntlet_info.get('breakthrough_state', 'NORMAL')
        header_parts.append(f"State: {state}")
    
    header_parts.append(_fmt_time(gen_time))
    
    print(f"\n{'='*70}")
    print(f"  {' | '.join(header_parts)}")
    print(f"{'='*70}")
    
    # ---- Best Agent ----
    if best_agent_info:
        ba = best_agent_info
        combined = ba.get('combined_fitness', 0)
        roi = ba.get('roi', 0)
        wr = ba.get('win_rate', 0)
        wins = ba.get('num_wins', 0)
        losses = ba.get('num_losses', 0)
        qr = ba.get('quality_ratio', 0)
        qc = ba.get('quality_count', 0)
        tt = ba.get('total_trades', 0)
        exp = ba.get('expectancy', 0)
        pnl = ba.get('pnl', 0)
        
        print(f"\n  BEST AGENT (Agent {ba.get('idx', '?')})")
        print(f"  {'-'*55}")
        print(f"  Combined Fitness: {combined:>10.2f}{_delta('best_combined_fitness')}")
        print(f"  ROI:              {roi:>9.2f}%{_delta('best_roi', pct=True)}")
        print(f"  Win Rate:         {wr:>9.1f}%   ({wins}W / {losses}L){_delta('best_win_rate', '.1f', '%')}")
        print(f"  Quality Ratio:    {qr:>9.1f}%   ({qc}/{tt} trades){_delta('best_quality_ratio', '.1f', '%')}")
        print(f"  Expectancy:       {exp:>10.2f}{_delta('best_expectancy')}")
        print(f"  PnL:              ${pnl:>9.2f}{_delta('best_pnl', '.2f', '')}")

    # ---- Population Health ----
    mean_fit = np.mean(fitness_scores) if len(fitness_scores) > 0 else 0
    max_fit = np.max(fitness_scores) if len(fitness_scores) > 0 else 0
    std_fit = np.std(fitness_scores) if len(fitness_scores) > 0 else 0
    pop_size = len(fitness_scores)
    
    print(f"\n  POPULATION ({pop_size} agents)")
    print(f"  {'-'*55}")
    print(f"  Fitness:   best={max_fit:>8.2f}  mean={mean_fit:>8.2f}  std={std_fit:>7.2f}{_delta('mean_fitness', '.2f', ' mean')}")
    
    if population_info:
        pos_count = population_info.get('positive_count', 0)
        mean_roi = population_info.get('mean_roi', 0)
        mean_wr = population_info.get('mean_win_rate', 0)
        pos_pct = (pos_count / pop_size * 100) if pop_size > 0 else 0
        print(f"  Positive:  {pos_count}/{pop_size} agents ({pos_pct:.0f}%){_delta('positive_count', '.0f', ' agents')}")
        print(f"  Mean ROI:  {mean_roi:>8.2f}%{_delta('mean_roi', '.2f', '%')}     Mean WR: {mean_wr:>5.1f}%{_delta('mean_win_rate', '.1f', '%')}")

    # ---- Hall of Fame ----
    if hof_info:
        hi = hof_info
        print(f"\n  HALL OF FAME")
        print(f"  {'-'*55}")
        print(f"  Size: {hi.get('size', 0)}/{hi.get('capacity', 10)}"
              f"  |  Best: {hi.get('best', 0):.2f}  Worst: {hi.get('worst', 0):.2f}")
        median_roi = hi.get('median_roi', 0)
        hurdle = hi.get('roi_hurdle_ema', 0)
        print(f"  Median ROI: {median_roi:.2f}%  |  ROI Hurdle EMA: {hurdle:.2f}%{_delta('roi_hurdle_ema', '.2f', '%')}")
        
        # Global HoF
        if gauntlet_info and gauntlet_info.get('global_hof_enabled'):
            g_size = gauntlet_info.get('global_hof_size', 0)
            g_thresh = gauntlet_info.get('global_hof_threshold', float('-inf'))
            thresh_str = f"{g_thresh:.2f}" if g_thresh != float('-inf') else "Open"
            print(f"  Global 50: {g_size}/50  |  Entry: {thresh_str}")

    # ---- Gauntlet State (when not NORMAL) ----
    if gauntlet_info and gauntlet_info.get('gauntlet_enabled'):
        state = gauntlet_info.get('breakthrough_state', 'NORMAL')
        if state != 'NORMAL':
            print(f"\n  GAUNTLET: {state}")
            print(f"  {'-'*55}")
            baseline = gauntlet_info.get('confirmed_baseline', 0)
            print(f"  Baseline: {baseline:.2f}")
            if state == 'STABILIZATION':
                stab = gauntlet_info.get('stabilization_progress')
                if stab:
                    c, t = stab
                    bar = '█' * c + '░' * (t - c)
                    print(f"  Progress: [{bar}] {c}/{t}")
            queue = gauntlet_info.get('queue_size')
            if queue is not None:
                print(f"  Queue: {queue} candidates")
    
    # ---- Loop Performance ----
    print(f"\n  LOOP PERFORMANCE")
    print(f"  {'-'*55}")
    
    if timing_info:
        ti = timing_info
        et = ti.get('eval_time', 0)
        vt = ti.get('val_time', 0)
        tt = ti.get('train_time', 0)
        ev = ti.get('evolve_time', 0)
        total = et + vt + tt + ev
        if total > 0:
            print(f"  Eval: {_fmt_time(et)} ({et/total*100:.0f}%)"
                  f"  |  Val: {_fmt_time(vt)} ({vt/total*100:.0f}%)"
                  f"  |  Train: {_fmt_time(tt)} ({tt/total*100:.0f}%)"
                  f"  |  Evolve: {_fmt_time(ev)} ({ev/total*100:.0f}%)")
        
        # Training bottleneck breakdown (if available)
        comp_pct = ti.get('train_compute_pct', 0)
        load_pct = ti.get('train_data_load_pct', 0)
        xfer_pct = ti.get('train_gpu_transfer_pct', 0)
        if comp_pct > 0 or load_pct > 0:
            print(f"  Train split: compute={comp_pct:.0f}%  data_load={load_pct:.0f}%  gpu_xfer={xfer_pct:.0f}%")
    
    # Buffer + GPU
    buffer_capacity = Config.LOCAL_BUFFER_SIZE if local_mode else Config.BUFFER_SIZE
    buf_pct = buffer_size / buffer_capacity * 100 if buffer_capacity > 0 else 0
    gpu_str = ""
    if resource_stats and 'peak_vram_gb' in resource_stats:
        gpu_str = f"  |  GPU: {resource_stats['peak_vram_gb']:.2f} GB peak"
    print(f"  Buffer: {buffer_size:,}/{buffer_capacity:,} ({buf_pct:.0f}%){gpu_str}")
    
    # ETA
    remaining = total_gens - (gen + 1)
    if gauntlet_info and gauntlet_info.get('gauntlet_enabled'):
        consistency_mode = gauntlet_info.get('consistency_mode', False)
        if consistency_mode:
            runway = gauntlet_info.get('runway_remaining', remaining)
            if isinstance(runway, (int, float)) and runway > 0 and avg_gen_time > 0:
                eta = _fmt_time(avg_gen_time * runway)
                print(f"  Max ETA: {eta} ({runway} gens remaining)")
        else:
            if remaining > 0 and avg_gen_time > 0:
                eta = _fmt_time(avg_gen_time * remaining)
                print(f"  ETA: {eta} ({remaining} gens remaining)")
    else:
        if remaining > 0 and avg_gen_time > 0:
            eta = _fmt_time(avg_gen_time * remaining)
            print(f"  ETA: {eta} ({remaining} gens remaining)")

    # ---- Events ----
    if events:
        print()
        for event in events:
            print(f"  >>> {event}")
    
    print(f"\n{'='*70}")


# Keep old function signature as a thin wrapper for backward compatibility
def print_generation_summary(gen: int, total_gens: int,
                             fitness_scores: List[float],
                             pop_stats: dict,
                             buffer_size: int,
                             best_fitness: float,
                             gen_time: float,
                             avg_gen_time: float,
                             resource_stats: Optional[dict] = None,
                             gauntlet_info: Optional[dict] = None,
                             local_mode: bool = False):
    """Legacy wrapper -- calls new dashboard with available data."""
    print_generation_dashboard(
        gen=gen, total_gens=total_gens,
        fitness_scores=fitness_scores,
        buffer_size=buffer_size,
        gen_time=gen_time, avg_gen_time=avg_gen_time,
        gauntlet_info=gauntlet_info,
        resource_stats=resource_stats,
        local_mode=local_mode,
    )


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
        gauntlet_score: Final aggregated gauntlet score (Penalized Median = Median - 0.5*StdDev)
    """
    print("\n" + "="*70)
    print(f"{'GAUNTLET SLICE ANALYSIS':^70}")
    print("="*70)

    # Statistical analysis
    median_score = float(np.median(fitness_scores))
    std_dev = float(np.std(fitness_scores))
    # Coefficient of Variation: measure of volatility (std dev / |mean|)
    # Higher CV = more volatile performance across slices
    cv = (std_dev / abs(mean_score)) * 100 if abs(mean_score) > 0.001 else 0

    print("\n📊 STATISTICAL SUMMARY")
    print("-" * 70)
    print(f"  Gauntlet Score:    {gauntlet_score:>12.2f}  (Penalized Median = Median - 0.5*StdDev)")
    print(f"  Median:            {median_score:>12.2f}")
    print(f"  Mean:              {mean_score:>12.2f}")
    print(f"  Min:               {min_score:>12.2f}")
    print(f"  Max:               {max_score:>12.2f}")
    print(f"  Std Dev:           {std_dev:>12.2f}")
    print(f"  CV (StdDev/|Mean|):{cv/100:>12.3f}  {'⚠️  High volatility!' if cv > 50 else '✓ Stable' if cv < 25 else '~ Moderate'}")
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