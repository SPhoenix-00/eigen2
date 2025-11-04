"""
Display utilities for Project Eigen 2
Pretty printing and progress visualization
"""

import numpy as np
from typing import List
from utils.config import Config


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


def print_generation_summary(gen: int, total_gens: int, 
                             fitness_scores: List[float],
                             pop_stats: dict,
                             buffer_size: int,
                             best_fitness: float,
                             gen_time: float,
                             avg_gen_time: float):
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
    print(f"  Maximum:           {max_fitness:>12.2f}  {'🌟 NEW BEST!' if max_fitness >= best_fitness else ''}")
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
    print(f"  Buffer Ready:      {' '*10}{'✓ Yes' if buffer_size >= 10000 else '✗ No (needs ' + str(10000-buffer_size) + ' more)'}")
    print(f"  Generation Time:   {gen_time:>11.1f}s")
    print(f"  Avg Gen Time:      {avg_gen_time:>11.1f}s")
    
    remaining = total_gens - (gen + 1)
    eta_seconds = avg_gen_time * remaining
    eta_minutes = eta_seconds / 60
    eta_hours = eta_seconds / 3600
    
    print(f"  Remaining Gens:    {remaining:>12}")
    print(f"  ETA:               {eta_minutes:>11.1f}m  ({eta_hours:.1f}h)")
    
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
    print(f"  Best Fitness:      {trainer.best_fitness:>12.2f}")
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