"""
Download W&B Metrics to CSV

This script reads the last run info from last_run.json and downloads all metrics
from Weights & Biases to a CSV file in the workspace.

Usage:
    python download_metrics.py                              # Download from last run
    python download_metrics.py <run-name>                   # Download from specific run by name
    python download_metrics.py <entity>/<project>/<run-id> # Download using full W&B path
    python download_metrics.py --all                        # Download from all runs in project

Requirements:
    - Must be logged in to W&B: wandb login
    - Or set W&B_ENTITY environment variable: export W&B_ENTITY=your-username

Examples:
    python download_metrics.py                              # Uses last_run.json
    python download_metrics.py breezy-puddle-62             # By run name
    python download_metrics.py username/eigen2-self/abc123  # Full path
    python download_metrics.py --all                        # All runs
"""

import os
import sys
import json
import wandb
import pandas as pd
from pathlib import Path
from typing import Optional


def load_last_run_info() -> dict:
    """
    Load run information from last_run.json.

    Returns:
        Dictionary with run_name, run_id, and wandb_run_id
    """
    last_run_file = Path("last_run.json")

    if not last_run_file.exists():
        print("❌ Error: last_run.json not found.")
        print("Please run a training first or specify a run name.")
        sys.exit(1)

    with open(last_run_file, 'r') as f:
        run_info = json.load(f)

    return run_info


def download_run_metrics(run_path: str, output_dir: Path = Path(".")) -> str:
    """
    Download metrics from a W&B run and save to CSV.

    Args:
        run_path: Full W&B run path (e.g., "username/project/run_id")
        output_dir: Directory to save CSV file

    Returns:
        Path to saved CSV file
    """
    try:
        # Initialize wandb API
        api = wandb.Api()

        print(f"\n--- Downloading Metrics ---")
        print(f"Run: {run_path}")

        # Fetch run
        run = api.run(run_path)

        # Get run name for filename
        run_name = run.name

        print(f"Run Name: {run_name}")
        print(f"Created: {run.created_at}")
        print(f"State: {run.state}")

        # Download history (all logged metrics)
        print("\nFetching metrics history...")
        history = run.history()

        if history.empty:
            print("⚠ Warning: No metrics found for this run.")
            return None

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save to CSV
        output_file = output_dir / f"{run_name}_metrics.csv"
        history.to_csv(output_file, index=False)

        print(f"\n✓ Metrics saved to: {output_file}")
        print(f"  Rows: {len(history)}")
        print(f"  Columns: {len(history.columns)}")
        print(f"\nColumns available:")
        for col in history.columns:
            print(f"  - {col}")

        return str(output_file)

    except wandb.errors.CommError as e:
        print(f"❌ Error connecting to W&B: {e}")
        print("Make sure you're logged in: wandb login")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error downloading metrics: {e}")
        sys.exit(1)


def download_all_runs_metrics(entity: str, project: str, output_dir: Path = Path(".")):
    """
    Download metrics from all runs in a project.

    Args:
        entity: W&B username/entity
        project: W&B project name
        output_dir: Directory to save CSV files
    """
    try:
        api = wandb.Api()

        print(f"\n--- Downloading Metrics from All Runs ---")
        print(f"Project: {entity}/{project}")

        # Get all runs
        runs = api.runs(f"{entity}/{project}")

        print(f"Found {len(runs)} runs")

        for i, run in enumerate(runs, 1):
            print(f"\n[{i}/{len(runs)}] Processing: {run.name}")
            run_path = f"{entity}/{project}/{run.id}"
            download_run_metrics(run_path, output_dir)

        print(f"\n✓ Downloaded metrics from {len(runs)} runs to {output_dir}")

    except Exception as e:
        print(f"❌ Error downloading all runs: {e}")
        sys.exit(1)


def get_wandb_entity() -> str:
    """
    Get W&B entity (username) from current login or environment.

    Returns:
        W&B username/entity or None if not found
    """
    # First check environment variable
    entity = os.environ.get('W&B_ENTITY') or os.environ.get('WANDB_ENTITY')
    if entity:
        return entity

    # Try to get from W&B API
    try:
        api = wandb.Api()
        # Try to get the default entity - this is the most reliable way
        return api.default_entity
    except AttributeError:
        # Fallback for older wandb versions
        try:
            api = wandb.Api()
            # Get entity from any existing run in the default project
            runs = list(api.runs("eigen2-self", per_page=1))
            if runs:
                return runs[0].entity
        except:
            pass
    except:
        pass

    # Final fallback: try to get from viewer
    try:
        api = wandb.Api()
        viewer = api.viewer
        if isinstance(viewer, dict):
            return viewer.get('entity') or viewer.get('username')
        elif hasattr(viewer, 'entity'):
            return viewer.entity
        elif hasattr(viewer, 'username'):
            return viewer.username
    except:
        pass

    return None


def main():
    """Main entry point."""
    print("="*70)
    print("W&B Metrics Downloader".center(70))
    print("="*70)

    # Parse command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "--all":
            # Download all runs
            entity = get_wandb_entity()
            project = "eigen2-self"

            if not entity:
                print("❌ Error: Could not determine W&B entity.")
                print("\nPlease ensure you're logged in to W&B:")
                print("  1. Run: wandb login")
                print("  2. Or set W&B_ENTITY environment variable")
                print("\nAlternatively, you can manually specify the project:")
                print("  python download_metrics.py <entity>/<project>/<run-id>")
                sys.exit(1)

            download_all_runs_metrics(entity, project, Path("metrics"))
            return

        elif arg == "--help" or arg == "-h":
            print(__doc__)
            return

        else:
            # Check if it's a full run path (entity/project/run-id)
            if '/' in arg and arg.count('/') >= 2:
                # Full run path provided
                run_path = arg
                print(f"Using full run path: {run_path}")
                download_run_metrics(run_path, Path("."))
                return

            # Specific run name provided
            run_name = arg
            entity = get_wandb_entity()
            project = "eigen2-self"

            if not entity:
                print("❌ Error: Could not determine W&B entity.")
                print("\nPlease ensure you're logged in to W&B:")
                print("  1. Run: wandb login")
                print("  2. Or set W&B_ENTITY environment variable")
                print("\nAlternatively, you can manually specify the full run path:")
                print("  python download_metrics.py <entity>/<project>/<run-id>")
                print("\nExample:")
                print("  python download_metrics.py username/eigen2-self/abc123xyz")
                sys.exit(1)

            # Try to find the run by name
            api = wandb.Api()
            runs = api.runs(f"{entity}/{project}")

            matching_run = None
            for run in runs:
                if run.name == run_name or run.id == run_name:
                    matching_run = run
                    break

            if not matching_run:
                print(f"❌ Error: Run '{run_name}' not found in {entity}/{project}")
                print("\nAvailable runs:")
                for run in runs[:10]:  # Show first 10
                    print(f"  - {run.name} (ID: {run.id})")
                sys.exit(1)

            run_path = f"{entity}/{project}/{matching_run.id}"
            download_run_metrics(run_path, Path("."))
            return

    # Default: Use last_run.json
    run_info = load_last_run_info()

    print(f"\nLast run info:")
    print(f"  Run Name: {run_info.get('run_name', 'unknown')}")
    print(f"  Run ID: {run_info.get('run_id', 'unknown')}")
    print(f"  W&B Run ID: {run_info.get('wandb_run_id', 'unknown')}")

    # Get W&B entity
    entity = get_wandb_entity()

    if not entity:
        print("\n❌ Error: Could not determine W&B entity.")
        print("\nPlease ensure you're logged in to W&B:")
        print("  1. Run: wandb login")
        print("  2. Enter your API key from: https://wandb.ai/authorize")
        print("\nOr set the W&B_ENTITY environment variable:")
        print("  export W&B_ENTITY=your-username")
        print("\nYou can find your username at: https://wandb.ai/settings")
        sys.exit(1)

    # Construct run path
    project = "eigen2-self"
    # Get W&B run ID from last_run.json
    wandb_run_id = run_info.get('run_id')

    if not wandb_run_id or wandb_run_id == 'unknown':
        print("\n⚠️  No W&B run ID found in last_run.json")
        print("\nYou can find the run ID in your W&B dashboard URL.")
        print("For example, if your run URL is:")
        print("  https://wandb.ai/username/eigen2-self/runs/hf8skifg")
        print("Then the run ID is: hf8skifg")
        print(f"\nFor run '{run_info.get('run_name', 'unknown')}', please enter the W&B run ID:")

        try:
            wandb_run_id = input("Run ID: ").strip()
            if not wandb_run_id:
                print("❌ Error: No run ID provided")
                sys.exit(1)
        except (KeyboardInterrupt, EOFError):
            print("\n\n❌ Cancelled by user")
            sys.exit(1)

    run_path = f"{entity}/{project}/{wandb_run_id}"

    # Download metrics
    download_run_metrics(run_path, Path("."))

    print("\n" + "="*70)
    print("Download Complete!".center(70))
    print("="*70)


if __name__ == "__main__":
    main()
