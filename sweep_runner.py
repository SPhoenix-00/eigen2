"""
W&B Sweep Runner for Eigen2
Runs hyperparameter optimization experiments locally
"""

import wandb
import torch
import numpy as np
from pathlib import Path

from utils.config import Config
from training.erl_trainer import ERLTrainer
from data.data_loader import load_stock_data


def train_with_config(config_override=None):
    """
    Run training with W&B sweep configuration.

    Args:
        config_override: Dictionary of hyperparameters from W&B sweep
    """
    # Initialize W&B run (will be created by sweep agent)
    run = wandb.init()

    # Get sweep config
    sweep_config = wandb.config

    # Override Config class attributes with sweep parameters
    if sweep_config:
        # Learning rates
        if hasattr(sweep_config, 'actor_lr'):
            Config.ACTOR_LR = sweep_config.actor_lr
        if hasattr(sweep_config, 'critic_lr'):
            Config.CRITIC_LR = sweep_config.critic_lr

        # DDPG parameters
        if hasattr(sweep_config, 'gamma'):
            Config.GAMMA = sweep_config.gamma
        if hasattr(sweep_config, 'tau'):
            Config.TAU = sweep_config.tau

        # Training intensity
        if hasattr(sweep_config, 'gradient_steps_per_generation'):
            Config.GRADIENT_STEPS_PER_GENERATION = sweep_config.gradient_steps_per_generation
        if hasattr(sweep_config, 'gradient_accumulation_steps'):
            Config.GRADIENT_ACCUMULATION_STEPS = sweep_config.gradient_accumulation_steps
        if hasattr(sweep_config, 'batch_size'):
            Config.BATCH_SIZE = sweep_config.batch_size

        # Exploration noise
        if hasattr(sweep_config, 'noise_scale'):
            Config.NOISE_SCALE = sweep_config.noise_scale
        if hasattr(sweep_config, 'noise_decay'):
            Config.NOISE_DECAY = sweep_config.noise_decay

        # Genetic algorithm
        if hasattr(sweep_config, 'mutation_rate'):
            Config.MUTATION_RATE = sweep_config.mutation_rate
        if hasattr(sweep_config, 'mutation_std'):
            Config.MUTATION_STD = sweep_config.mutation_std

        # Trading environment
        if hasattr(sweep_config, 'loss_penalty_multiplier'):
            Config.LOSS_PENALTY_MULTIPLIER = sweep_config.loss_penalty_multiplier
        if hasattr(sweep_config, 'inaction_penalty'):
            Config.INACTION_PENALTY = sweep_config.inaction_penalty
        if hasattr(sweep_config, 'max_holding_period'):
            Config.MAX_HOLDING_PERIOD = sweep_config.max_holding_period

        # Fixed sweep parameters
        if hasattr(sweep_config, 'num_generations'):
            Config.NUM_GENERATIONS = sweep_config.num_generations
        if hasattr(sweep_config, 'population_size'):
            Config.POPULATION_SIZE = sweep_config.population_size
        if hasattr(sweep_config, 'buffer_size'):
            Config.BUFFER_SIZE = sweep_config.buffer_size

    # Update W&B config with all parameters (for tracking)
    wandb.config.update({
        "actor_lr": Config.ACTOR_LR,
        "critic_lr": Config.CRITIC_LR,
        "gamma": Config.GAMMA,
        "tau": Config.TAU,
        "gradient_steps": Config.GRADIENT_STEPS_PER_GENERATION,
        "gradient_accum": Config.GRADIENT_ACCUMULATION_STEPS,
        "batch_size": Config.BATCH_SIZE,
        "noise_scale": Config.NOISE_SCALE,
        "noise_decay": Config.NOISE_DECAY,
        "mutation_rate": Config.MUTATION_RATE,
        "mutation_std": Config.MUTATION_STD,
        "loss_penalty": Config.LOSS_PENALTY_MULTIPLIER,
        "inaction_penalty": Config.INACTION_PENALTY,
        "max_holding": Config.MAX_HOLDING_PERIOD,
        "num_generations": Config.NUM_GENERATIONS,
        "population_size": Config.POPULATION_SIZE,
        "buffer_size": Config.BUFFER_SIZE,
    })

    print("\n" + "="*70)
    print("Starting W&B Sweep Run")
    print("="*70)
    print(f"Run ID: {run.id}")
    print(f"Run Name: {run.name}")
    print("\nHyperparameters:")
    print(f"  Actor LR: {Config.ACTOR_LR}")
    print(f"  Critic LR: {Config.CRITIC_LR}")
    print(f"  Gamma: {Config.GAMMA}")
    print(f"  Tau: {Config.TAU}")
    print(f"  Gradient Steps/Gen: {Config.GRADIENT_STEPS_PER_GENERATION}")
    print(f"  Batch Size: {Config.BATCH_SIZE}")
    print(f"  Noise Scale: {Config.NOISE_SCALE}")
    print(f"  Mutation Rate: {Config.MUTATION_RATE}")
    print(f"  Loss Penalty: {Config.LOSS_PENALTY_MULTIPLIER}x")
    print(f"  Max Holding: {Config.MAX_HOLDING_PERIOD} days")
    print("="*70 + "\n")

    # Validate config
    if not Config.validate():
        print("Configuration validation failed!")
        return

    # Set random seeds
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)

    # Load data
    print("Loading stock data...")
    train_data, val_data = load_stock_data(
        Config.DATA_PATH,
        Config.CONTEXT_WINDOW_DAYS,
        Config.TRAIN_TEST_SPLIT
    )
    print(f"✓ Loaded {len(train_data)} training samples, {len(val_data)} validation samples")

    # Create trainer (NO cloud sync for local sweeps)
    trainer = ERLTrainer(
        train_data=train_data,
        val_data=val_data,
        cloud_sync=None  # Disable cloud sync for local runs
    )

    # Run training
    try:
        trainer.train()

        # Log final metrics
        wandb.log({
            "final/best_fitness": trainer.best_fitness,
            "final/total_generations": len(trainer.fitness_history),
            "final/avg_generation_time": np.mean(trainer.generation_times),
        })

        print("\n" + "="*70)
        print("Sweep Run Complete!")
        print(f"Best Fitness: {trainer.best_fitness:.2f}")
        print("="*70 + "\n")

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        wandb.log({"error": str(e)})
        raise

    finally:
        # Clean up
        wandb.finish()


if __name__ == "__main__":
    train_with_config()
