"""
Project Eigen 2 - Main Entry Point
Evolutionary Reinforcement Learning for Stock Trading
"""

import argparse
import torch
import numpy as np
import random
from pathlib import Path

from data.loader import StockDataLoader
from training.erl_trainer import ERLTrainer
from utils.config import Config


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    """Main training function."""
    print("="*60)
    print("Project Eigen 2: ERL Stock Trading")
    print("="*60)

    # --- 1. ADD ARGUMENT PARSER ---
    parser = argparse.ArgumentParser(description="Run Project Eigen 2 ERL Training")
    parser.add_argument(
        '--resume',
        action='store_true',  # This makes it a True/False flag
        help='Resume training from the last checkpoint'
    )
    args = parser.parse_args()
    # --------------------------------

    # Set seed for reproducibility
    set_seed(Config.SEED)
    print(f"\nRandom seed: {Config.SEED}")

    # Display configuration
    Config.display()

    # Validate configuration
    if not Config.validate():
        print("\n❌ Configuration validation failed!")
        return

    print("\n" + "="*60)
    print("Phase 1: Data Loading")
    print("="*60)

    # Load data
    loader = StockDataLoader()
    data_array, stats = loader.load_and_prepare()

    print("\n" + "="*60)
    print("Phase 2: ERL Training")
    print("="*60)

    # Create trainer
    trainer = ERLTrainer(loader)

    # --- 2. ADD CHECKPOINT LOADING ---
    # If the --resume flag was used, load the checkpoint
    if args.resume:
        trainer.load_checkpoint()
    # --------------------------------

    # Start or resume training
    trainer.train()

    print("\n" + "="*60)
    print("✓ Training Complete!")
    print("="*60)
    print(f"\nBest fitness achieved: {trainer.best_fitness:.2f}")
    print(f"Total generations: {Config.NUM_GENERATIONS}")
    print(f"Final buffer size: {len(trainer.replay_buffer)}")

    # Show where results are saved
    print(f"\nResults saved to:")
    print(f"  Checkpoints: {Config.CHECKPOINT_DIR}")
    print(f"  Logs: {Config.LOG_DIR}")
    print(f"\nView training progress:")
    print(f"  tensorboard --logdir={Config.LOG_DIR}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        raise