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
        action='store_true',
        help='Resume training from the last run (reads from last_run.json). '
             'This will download checkpoints from GCS and reconnect to the wandb run.'
    )
    parser.add_argument(
        '--resume-run',
        type=str,
        default=None,
        metavar='RUN_NAME',
        help='Resume training from a specific wandb run (e.g., "azure-thunder-123"). '
             'Overrides --resume. Useful for resuming older runs.'
    )
    args = parser.parse_args()
    # --------------------------------

    # NOTE: Seed will be set AFTER wandb init in ERLTrainer to ensure unique seeds per run
    # This prevents parallel runs from having identical behavior

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

    # Determine resume run name
    resume_run_name = None
    if args.resume_run:
        # Specific run provided via --resume-run
        resume_run_name = args.resume_run
        print(f"Resuming from specific run: {resume_run_name}")
    elif args.resume:
        # Resume from last run (read from last_run.json)
        import json
        last_run_file = Path("last_run.json")
        if last_run_file.exists():
            try:
                with open(last_run_file, 'r') as f:
                    last_run_info = json.load(f)
                resume_run_name = last_run_info.get('run_name')
                print(f"Resuming from last run: {resume_run_name}")
            except Exception as e:
                print(f"⚠ Could not read last_run.json: {e}")
                print("Starting new training run instead.")
        else:
            print("⚠ No last_run.json found. Starting new training run.")

    # Create trainer (pass resume_run_name if resuming)
    trainer = ERLTrainer(loader, resume_run_name=resume_run_name)

    # --- 2. CHECKPOINT LOADING IS NOW HANDLED IN ERLTrainer.__init__ ---
    # If resume_run_name was provided, checkpoints are automatically loaded
    # and wandb run is reconnected during trainer initialization
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