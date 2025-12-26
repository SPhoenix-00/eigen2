"""
Project Eigen 2 - Main Entry Point
Evolutionary Reinforcement Learning for Stock Trading
"""

import os
# GPU environment setup must happen before importing torch
# This sets PYTORCH_CUDA_ALLOC_CONF (NVIDIA) or PYTORCH_HIP_ALLOC_CONF (AMD)
from utils.device import setup_gpu_environment
setup_gpu_environment()

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
from utils.cleanup_orphans import cleanup_orphans


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # cuDNN settings (works for both CUDA and ROCm)
    if torch.cuda.is_available():
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
        parser.add_argument(
            '--leverage',
            action='store_true',
            help='Use with --resume to enable leverage mode. Replaces bottom 5 agents with '
                 'top 5 Hall of Fame agents modified to trade with 1.5x coefficients for 5 generations.'
        )
        parser.add_argument(
            '--consistency',
            action='store_true',
            help='Enable consistency mode. Evaluates agents over 5 episodes with sum (not avg), '
                 'and magnifies losses to focus training on reducing drawdowns.'
        )
        parser.add_argument(
            '--heroes',
            type=str,
            nargs='?',
            const='auto',
            default=None,
            metavar='HOF_DIR',
            help='Path to a Hall of Fame directory. Loads all agents, evaluates them with current '
                 'reward function, and selects best as initial population. Uses reduced mutant ratio. '
                 'If flag is used without a path, automatically uses global50 directory for current context window.'
        )
        parser.add_argument(
            '--single',
            type=str,
            default=None,
            metavar='AGENT_FILENAME',
            help='Load a specific Global50 agent by filename (e.g., "azure-thunder-123_42") '
                 'and run focused refinement training. Auto-enables consistency mode. '
                 'Training seeks 4 breakthroughs of 5%% improvement each.'
        )
        parser.add_argument(
            '--multi',
            action='store_true',
            help='Multi-agent mode: Train all 9 committee members sequentially. '
                 'Trains one member at a time until breakthrough (5%% improvement), then rotates to next. '
                 'Ends after 3 turnovers (all 9 members achieve 3 breakthroughs each).'
        )
        parser.add_argument(
            '--cleanup',
            action='store_true',
            help='Clean up orphaned replay buffer files before resuming. Use with --resume or --resume-run. '
                 'Scans buffer_storage directory and removes files not tracked in metadata (zombie files from crashes).'
        )
        parser.add_argument(
            '--buffer',
            type=str,
            default=None,
            metavar='BUFFER_PATH',
            help='Path to an existing buffer_storage folder to reuse (e.g., "checkpoints/run-123/buffer_storage"). '
                 'Allows starting a new run with pre-filled replay buffer instead of starting from zero.'
        )
        parser.add_argument(
            '--reset-limit',
            action='store_true',
            help='Use with --resume to reset the fallback generation counter to max. '
                 'In consistency mode, this resets the "generations since last turnover" counter, '
                 'giving the run a fresh runway of MAX_GENERATIONS_GAUNTLET generations.'
        )
        args = parser.parse_args()
        # --------------------------------

        # Auto-detect global50 directory if --heroes was used without a path
        if args.heroes == 'auto':
            import glob

            # Context window identifier (e.g., "cw151" for 151-day window)
            context_window_id = f"cw{Config.CONTEXT_WINDOW_DAYS}"

            # Build list of ALL available directories in priority order
            # Heroes loading will aggregate agents from all of them
            global50_candidates = [
                (Path("global50") / context_window_id, f"current context ({context_window_id})"),
                (Path("workspace/global50") / context_window_id, f"current context ({context_window_id})"),
                (Path("global50") / "cw504", "fallback (cw504)"),
                (Path("workspace/global50") / "cw504", "fallback (cw504)"),
                (Path("global50"), "legacy"),
                (Path("workspace/global50"), "legacy")
            ]

            # Collect all directories with agents
            found_dirs = []
            total_agents = 0

            for candidate, desc in global50_candidates:
                if candidate.exists():
                    # Check all possible agent subdirectories
                    for subdir in ["agents", "hall_of_fame", "."]:
                        agent_dir = candidate / subdir if subdir != "." else candidate
                        if agent_dir.exists():
                            agent_files = glob.glob(str(agent_dir / "*.pth"))
                            if agent_files:
                                found_dirs.append(str(candidate))
                                total_agents += len(agent_files)
                                print(f"  ✓ Found {len(agent_files)} agents in {candidate} ({desc})")
                                break  # Found agents in this candidate, move to next

            if found_dirs:
                # Join all directories with '|' separator for heroes loading
                args.heroes = '|'.join(found_dirs)
                print(f"\n✓ Auto-detected global50 directories")
                print(f"  Total agents available: {total_agents}")
                print(f"  Training context window: {Config.CONTEXT_WINDOW_DAYS} days")
            else:
                print("\n⚠ WARNING: --heroes auto-detect failed")
                print(f"  Tried: global50/{context_window_id}/, global50/cw504/, and legacy paths")
                print(f"  You may need to run: python download_global50.py")
                print("  Continuing with random initialization.")
                args.heroes = None

        # Validate --single agent exists BEFORE loading data (fail fast)
        if args.single:
            # Construct expected filename
            agent_filename = args.single if args.single.endswith('.pth') else f"{args.single}.pth"

            # Context window identifier
            context_window_id = f"cw{Config.CONTEXT_WINDOW_DAYS}"

            # Check possible paths for the agent
            possible_paths = [
                Path(f"global50/{context_window_id}/agents/{agent_filename}"),
                Path(f"workspace/global50/{context_window_id}/agents/{agent_filename}"),
            ]

            agent_found = False
            for path in possible_paths:
                if path.exists():
                    agent_found = True
                    args.single_agent_path = str(path)
                    break

            if not agent_found:
                print(f"\n❌ ERROR: Agent '{args.single}' not found in Global50")
                print(f"  Searched: {[str(p) for p in possible_paths]}")
                print(f"  Available agents can be listed with: ls global50/{context_window_id}/agents/")
                sys.exit(1)

            # Auto-enable consistency mode for single-agent training
            args.consistency = True
            print(f"\n🎯 SINGLE AGENT MODE: {agent_filename}")
            print(f"  Agent path: {args.single_agent_path}")
            print(f"  Consistency mode auto-enabled")
            print(f"  Target: {Config.SINGLE_TARGET_BREAKTHROUGHS} breakthroughs of 5% each")

        # Validate --multi mode (committee roster must exist)
        if args.multi:
            from committee import CommitteeManager

            context_window_id = f"cw{Config.CONTEXT_WINDOW_DAYS}"
            manager = CommitteeManager(Config.CONTEXT_WINDOW_DAYS)
            roster = manager.load_roster()

            if roster is None:
                print(f"\n❌ ERROR: No committee roster found for {context_window_id}")
                print(f"  Run: python committee.py --draft")
                sys.exit(1)

            if len(roster.get('members', [])) != Config.COMMITTEE_SIZE:
                print(f"\n❌ ERROR: Committee has {len(roster['members'])} members, expected {Config.COMMITTEE_SIZE}")
                sys.exit(1)

            # Auto-enable consistency mode for multi-agent training
            args.consistency = True
            args.multi_roster = roster

            print(f"\n🎯 MULTI-AGENT MODE")
            print(f"  Committee members: {len(roster['members'])}")
            print(f"  Population size: {Config.POPULATION_SIZE} (standard)")
            print(f"  Training mode: Sequential (one member at a time)")
            print(f"  Target turnovers: {Config.MULTI_TARGET_TURNOVERS}")
            print(f"  Consistency mode: AUTO-ENABLED")

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

        # Run cleanup if requested (must have a resume_run_name)
        if args.cleanup:
            if resume_run_name:
                print("\n" + "="*60)
                print("Phase 1.5: Cleanup Orphaned Buffer Files")
                print("="*60)
                try:
                    cleanup_result = cleanup_orphans(
                        run_name=resume_run_name,
                        dry_run=False,
                        verbose=True
                    )
                    if not cleanup_result['success']:
                        print(f"⚠ Cleanup failed: {cleanup_result.get('error', 'unknown error')}")
                        print("Continuing with training anyway...")
                except Exception as e:
                    print(f"⚠ Cleanup error: {e}")
                    print("Continuing with training anyway...")
            else:
                print("\n⚠ WARNING: --cleanup requires --resume or --resume-run")
                print("Ignoring --cleanup flag.\n")

        # Create trainer (pass resume_run_name and leverage flag if resuming)
        # Pass original stdout/stderr so wandb can properly capture console output
        trainer = ERLTrainer(
            loader,
            resume_run_name=resume_run_name,
            enable_leverage=args.leverage,
            consistency_mode=args.consistency,
            heroes_hof_dir=args.heroes,
            single_agent_path=getattr(args, 'single_agent_path', None),
            buffer_storage_path=args.buffer,
            reset_limit=args.reset_limit,
            original_stdout=tee_logger.terminal,
            original_stderr=tee_logger.terminal,
            multi_mode=args.multi,
            multi_roster=getattr(args, 'multi_roster', None)
        )

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