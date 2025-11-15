"""
Project Eigen 2 - Main Entry Point
Evolutionary Reinforcement Learning for Stock Trading
"""

import os
# Fix PyTorch memory fragmentation (must be set before importing torch)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import argparse
import torch
import numpy as np
import random
import sys
from pathlib import Path
from datetime import datetime

from data.loader import StockDataLoader
from training.erl_trainer import ERLTrainer
from utils.config import Config


class TeeLogger:
    """Logger that writes to both file and stdout."""

    def __init__(self, filepath: Path):
        """
        Initialize TeeLogger.

        Args:
            filepath: Path to the log file
        """
        self.terminal = sys.stdout
        self.log_file = open(filepath, 'w', buffering=1)  # Line buffered

    def write(self, message):
        """Write message to both terminal and file."""
        self.terminal.write(message)
        self.log_file.write(message)

    def flush(self):
        """Flush both outputs."""
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        """Check if the terminal is a TTY (needed by wandb and other libraries)."""
        return self.terminal.isatty()

    def fileno(self):
        """Return the file descriptor (needed by some libraries)."""
        return self.terminal.fileno()

    def close(self):
        """Close the log file."""
        self.log_file.close()

    def __del__(self):
        """Ensure file is closed on deletion."""
        if hasattr(self, 'log_file') and not self.log_file.closed:
            self.log_file.close()


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
    # Setup logging to capture all outputs to file
    # Create evaluation_results directory (same folder where evaluations go)
    log_dir = Path("evaluation_results")
    log_dir.mkdir(exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"training_log_{timestamp}.txt"

    # Redirect stdout to both terminal and log file
    tee_logger = TeeLogger(log_file)
    sys.stdout = tee_logger
    sys.stderr = tee_logger  # Also capture error messages

    print("="*60)
    print("Project Eigen 2: ERL Stock Trading")
    print(f"Logging to: {log_file}")
    print("="*60)

    try:
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
        print(f"  Training log: {log_file}")
        print(f"\nView training progress:")
        print(f"  tensorboard --logdir={Config.LOG_DIR}")

    finally:
        # Restore original stdout/stderr and close log file
        sys.stdout = tee_logger.terminal
        sys.stderr = tee_logger.terminal
        tee_logger.close()
        print(f"\n✓ Complete training log saved to: {log_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        raise