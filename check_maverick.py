"""
Check if a run (by name) is a Maverick.

Checks in order:
  1. Weights & Biases (wandb) — looks for 'maverick' tag or maverick_mode=True in config
  2. Global50 JSON — checks is_maverick field on matching entries
  3. Archive JSON scoresheets (local) — checks is_maverick in archive metadata
  4. Long-term-archive JSON scoresheets (local) — checks is_maverick in long-term metadata
  5. Evaluation fallback — loads agents from checkpoints and runs a gauntlet eval.
     If total_trades > 5000 across all slices, the agent is classified as a Maverick.

Usage:
    python check_maverick.py <run_name>
    python check_maverick.py <run_name> --cw 504          # Specify context window
    python check_maverick.py <run_name> --eval             # Force evaluation even if found in metadata
    python check_maverick.py <run_name> --threshold 5000   # Custom trade threshold (default: 5000)
    python check_maverick.py <run_name> --checkpoint-dir checkpoints  # Custom checkpoint directory

Examples:
    python check_maverick.py azure-thunder-313
    python check_maverick.py twilight-haze-42 --cw 504
    python check_maverick.py crimson-wave-7 --eval --threshold 3000
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Optional, List, Tuple


WANDB_PROJECT = "eigen2-self"
MAVERICK_TRADE_THRESHOLD = 5000


def check_wandb(run_name: str) -> Optional[bool]:
    """
    Check wandb for a run matching run_name.
    Returns True/False if found, None if wandb unavailable or no match.
    """
    try:
        import wandb
        api = wandb.Api()
    except ImportError:
        print("  wandb not installed. Skipping wandb check.")
        return None
    except Exception as e:
        print(f"  wandb API init failed: {e}")
        return None

    print(f"  Searching wandb project '{WANDB_PROJECT}' for run name '{run_name}'...")

    try:
        # Search by display name
        runs = api.runs(WANDB_PROJECT, filters={"display_name": run_name})
        matched = list(runs)
    except Exception as e:
        print(f"  wandb query failed: {e}")
        return None

    if not matched:
        print(f"  No matching runs found in wandb.")
        return None

    for run in matched:
        tags = run.tags or []
        config = run.config or {}
        maverick_tag = "maverick" in tags
        maverick_config = config.get("maverick_mode", False)
        is_maverick = maverick_tag or maverick_config

        status_str = "MAVERICK" if is_maverick else "NOT Maverick"
        tag_str = f"tags={tags}" if tags else "no tags"
        config_str = f"maverick_mode={config.get('maverick_mode', 'N/A')}"

        print(f"  Found: {run.name} (id={run.id}) — {status_str}")
        print(f"    {tag_str}, {config_str}")

        return is_maverick

    return None


def check_global50_json(run_name: str, cw_days: int) -> Optional[bool]:
    """
    Check global50.json for entries matching run_name.
    Returns True/False if found, None if not found.
    """
    cw_id = f"cw{cw_days}"
    json_path = Path("global50") / cw_id / "global50.json"

    if not json_path.exists():
        print(f"  {json_path} not found.")
        return None

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error reading {json_path}: {e}")
        return None

    entries = data.get("entries", [])
    matches = [e for e in entries if e.get("run_name") == run_name]

    if not matches:
        print(f"  No entries for '{run_name}' in {json_path}.")
        return None

    any_maverick = False
    for e in matches:
        is_mav = e.get("is_maverick", False)
        if is_mav:
            any_maverick = True
        tag = " [M]" if is_mav else ""
        score = e.get("gauntlet_score", 0)
        trades = e.get("total_trades", 0)
        print(f"  Global50 entry: {run_name}_{e.get('agent_id', '?')}{tag} "
              f"score={score:.2f}, trades={trades}")

    return any_maverick


def check_archive_json(run_name: str, cw_days: int) -> Optional[bool]:
    """
    Check archive/ JSON scoresheets for matching run_name.
    Returns True/False if found, None if not found.
    """
    cw_id = f"cw{cw_days}"
    archive_dir = Path("global50") / cw_id / "archive"

    if not archive_dir.exists():
        print(f"  Archive directory {archive_dir} not found.")
        return None

    matches = []
    for json_file in archive_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            if data.get("run_name") == run_name:
                matches.append(data)
        except Exception:
            continue

    if not matches:
        print(f"  No entries for '{run_name}' in archive.")
        return None

    any_maverick = False
    for e in matches:
        is_mav = e.get("is_maverick", False)
        if is_mav:
            any_maverick = True
        tag = " [M]" if is_mav else ""
        score = e.get("gauntlet_score", 0)
        trades = e.get("total_trades", 0)
        print(f"  Archive entry: {run_name}_{e.get('agent_id', '?')}{tag} "
              f"score={score:.2f}, trades={trades}")

    return any_maverick


def check_long_term_archive_json(run_name: str, cw_days: int) -> Optional[bool]:
    """
    Check long-term-archive/ JSON scoresheets for matching run_name.
    Returns True/False if found, None if not found.
    """
    cw_id = f"cw{cw_days}"
    lt_archive_dir = Path("global50") / cw_id / "long-term-archive"

    if not lt_archive_dir.exists():
        print(f"  Long-term-archive directory {lt_archive_dir} not found.")
        return None

    matches = []
    for json_file in lt_archive_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            if data.get("run_name") == run_name:
                matches.append(data)
        except Exception:
            continue

    if not matches:
        print(f"  No entries for '{run_name}' in long-term-archive.")
        return None

    any_maverick = False
    for e in matches:
        is_mav = e.get("is_maverick", False)
        if is_mav:
            any_maverick = True
        tag = " [M]" if is_mav else ""
        score = e.get("gauntlet_score", 0)
        trades = e.get("total_trades", 0)
        print(f"  Long-term-archive entry: {run_name}_{e.get('agent_id', '?')}{tag} "
              f"score={score:.2f}, trades={trades}")

    return any_maverick


def find_checkpoint_agents(run_name: str, checkpoint_dir: str) -> List[Path]:
    """
    Find agent .pth files associated with a run_name.
    
    Search strategy:
      1. checkpoints/hall_of_fame/ — best agents saved during training
      2. checkpoints/population/ — current population snapshot
      3. checkpoints/best_agent.pth — single best agent
      4. Any directory matching the run_name pattern
    """
    base = Path(checkpoint_dir)
    agents = []

    # Check if a trainer_state.json exists to confirm the run_name
    state_path = base / "trainer_state.json"
    if state_path.exists():
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
            saved_name = state.get("wandb_run_name", "")
            if saved_name and saved_name != run_name:
                print(f"  Warning: checkpoint trainer_state.json has run_name='{saved_name}', "
                      f"not '{run_name}'.")
        except Exception:
            pass

    # Search for agents in order of preference
    # 1. Hall of fame agents
    hof_dir = base / "hall_of_fame"
    if hof_dir.exists():
        hof_agents = list(hof_dir.glob("hof_agent_*.pth"))
        if hof_agents:
            print(f"  Found {len(hof_agents)} hall-of-fame agents in {hof_dir}")
            agents.extend(hof_agents)

    # 2. Population agents
    pop_dir = base / "population"
    if pop_dir.exists():
        pop_agents = list(pop_dir.glob("agent_*.pth"))
        if pop_agents:
            print(f"  Found {len(pop_agents)} population agents in {pop_dir}")
            agents.extend(pop_agents)

    # 3. Best agent
    best_path = base / "best_agent.pth"
    if best_path.exists():
        print(f"  Found best_agent.pth")
        agents.append(best_path)

    # 4. Gauntlet snapshot agents
    gauntlet_pop = base / "gauntlet_snapshot" / "population"
    if gauntlet_pop.exists():
        gauntlet_agents = list(gauntlet_pop.glob("agent_*.pth"))
        if gauntlet_agents:
            print(f"  Found {len(gauntlet_agents)} gauntlet snapshot agents in {gauntlet_pop}")
            agents.extend(gauntlet_agents)

    # 5. Global50 agents directory (for runs already promoted)
    global_agents_dir = Path("global50") / f"cw{151}" / "agents"
    if global_agents_dir.exists():
        for pth_file in global_agents_dir.glob(f"{run_name}_*.pth"):
            print(f"  Found global50 agent: {pth_file.name}")
            agents.append(pth_file)

    return agents


def evaluate_agents_for_maverick(agent_paths: List[Path], threshold: int) -> Tuple[bool, dict]:
    """
    Load and evaluate agents using the same gauntlet as global50.py --eval.
    An agent is classified as Maverick if total_trades > threshold.
    
    Returns:
        (is_maverick, summary_dict)
    """
    # Heavy imports deferred to here so metadata-only checks stay fast
    from data.loader import StockDataLoader
    from environment.trading_env import TradingEnvironment
    from models.ddpg_agent import DDPGAgent
    from utils.config import Config
    from training.breakthrough import BreakthroughState
    from training.fitness import calculate_triad_fitness, calculate_expectancy
    from training.episode import run_episode_batched
    from training.validation_slices import generate_gauntlet_slices

    print(f"\n  Loading market data...")
    data_loader = StockDataLoader()
    data_array, stats = data_loader.load_and_prepare()
    val_start_idx = data_loader.val_start_idx
    val_end_idx = data_loader.val_end_idx

    full_end_idx = len(data_loader.data_array_full)
    eval_env = TradingEnvironment(
        data_array=data_loader.data_array,
        dates=data_loader.dates,
        normalization_stats=stats,
        start_idx=Config.CONTEXT_WINDOW_DAYS,
        end_idx=full_end_idx,
        trading_end_idx=Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS,
        data_array_full=data_loader.data_array_full,
        consistency_mode=False,
    )

    train_start_idx = Config.CONTEXT_WINDOW_DAYS
    train_end_idx = val_start_idx

    gauntlet_slices = generate_gauntlet_slices(
        train_start_idx, train_end_idx, val_start_idx, val_end_idx
    )
    print(f"  Gauntlet: {len(gauntlet_slices)} slices")

    results = []

    for agent_path in agent_paths:
        agent_name = agent_path.stem
        print(f"\n  Evaluating {agent_name}...")

        try:
            agent = DDPGAgent(agent_id=0)
            agent.load(str(agent_path))
            agent.actor.eval()
            agent.critic.eval()
        except Exception as e:
            print(f"    Failed to load: {e}")
            continue

        eval_env.set_consistency_mode(False)
        eval_env.set_gauntlet_mode(True)

        total_wins = 0
        total_losses = 0
        all_closed_trades = []

        for i, (start_idx, end_idx, _) in enumerate(gauntlet_slices):
            raw_fitness, episode_info = run_episode_batched(
                agent=agent,
                env=eval_env,
                start_idx=start_idx,
                end_idx=end_idx,
                training=False,
                batch_size=16,
            )
            total_wins += episode_info.get('num_wins', 0)
            total_losses += episode_info.get('num_losses', 0)
            if 'closed_trades' in episode_info and episode_info['closed_trades']:
                all_closed_trades.extend(episode_info['closed_trades'])

        total_trades = int(total_wins + total_losses)
        is_maverick = total_trades > threshold

        tag = " [M] MAVERICK" if is_maverick else " (Standard)"
        print(f"    Total trades: {total_trades} (threshold: {threshold}){tag}")

        results.append({
            'agent_name': agent_name,
            'total_trades': total_trades,
            'is_maverick': is_maverick,
        })

        eval_env.set_gauntlet_mode(False)

    if not results:
        return False, {'error': 'No agents could be evaluated'}

    any_maverick = any(r['is_maverick'] for r in results)
    return any_maverick, {
        'agents_evaluated': len(results),
        'maverick_count': sum(1 for r in results if r['is_maverick']),
        'details': results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check if a run is a Maverick.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("run_name", help="Run name to check (e.g., azure-thunder-313)")
    parser.add_argument("--cw", type=int, default=151,
                        help="Context window size in days (default: 151)")
    parser.add_argument("--eval", action="store_true", dest="force_eval",
                        help="Force gauntlet evaluation even if found in metadata")
    parser.add_argument("--threshold", type=int, default=MAVERICK_TRADE_THRESHOLD,
                        help=f"Trade count threshold for Maverick classification (default: {MAVERICK_TRADE_THRESHOLD})")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                        help="Checkpoint directory to search for agents (default: checkpoints)")

    args = parser.parse_args()
    run_name = args.run_name

    print(f"\n{'='*60}")
    print(f"MAVERICK CHECK: {run_name}")
    print(f"{'='*60}")
    print(f"  Context window: cw{args.cw}")
    print(f"  Trade threshold: {args.threshold}")

    # ── Step 1: wandb ────────────────────────────────────────────────────
    print(f"\n--- Step 1: Checking wandb ---")
    result = check_wandb(run_name)
    if result is not None and not args.force_eval:
        _print_verdict(run_name, result, "wandb")
        return

    # ── Step 2: Global50 JSON ────────────────────────────────────────────
    print(f"\n--- Step 2: Checking Global50 JSON ---")
    result = check_global50_json(run_name, args.cw)
    if result is not None and not args.force_eval:
        _print_verdict(run_name, result, "Global50")
        return

    # ── Step 3: Archive ──────────────────────────────────────────────────
    print(f"\n--- Step 3: Checking Archive ---")
    result = check_archive_json(run_name, args.cw)
    if result is not None and not args.force_eval:
        _print_verdict(run_name, result, "Archive")
        return

    # ── Step 4: Long-term Archive ────────────────────────────────────────
    print(f"\n--- Step 4: Checking Long-term Archive ---")
    result = check_long_term_archive_json(run_name, args.cw)
    if result is not None and not args.force_eval:
        _print_verdict(run_name, result, "Long-term Archive")
        return

    # ── Step 5: Evaluation fallback ──────────────────────────────────────
    print(f"\n--- Step 5: Evaluation Fallback ---")
    print(f"  No metadata match found. Searching for agents to evaluate...")
    print(f"  Maverick threshold: >{args.threshold} trades across gauntlet slices")

    agent_paths = find_checkpoint_agents(run_name, args.checkpoint_dir)

    if not agent_paths:
        print(f"\n  No agent .pth files found for '{run_name}'.")
        print(f"  Searched: {args.checkpoint_dir}/ and global50/cw{args.cw}/agents/")
        print(f"\n  Cannot determine Maverick status. Provide agent files or mark manually:")
        print(f"    python global50.py --mark-maverick {run_name}")
        sys.exit(1)

    print(f"\n  Found {len(agent_paths)} agent(s) to evaluate.")
    is_maverick, summary = evaluate_agents_for_maverick(agent_paths, args.threshold)

    if 'error' in summary:
        print(f"\n  Evaluation failed: {summary['error']}")
        sys.exit(1)

    _print_verdict(run_name, is_maverick, "Evaluation", summary)


def _print_verdict(run_name: str, is_maverick: bool, source: str, summary: dict = None):
    """Print the final verdict."""
    print(f"\n{'='*60}")
    if is_maverick:
        print(f"  VERDICT: {run_name} is a MAVERICK  [M]")
    else:
        print(f"  VERDICT: {run_name} is NOT a Maverick (Standard)")
    print(f"  Source:  {source}")
    if summary and 'details' in summary:
        print(f"  Agents evaluated: {summary['agents_evaluated']}")
        print(f"  Mavericks found:  {summary['maverick_count']}")
        for d in summary['details']:
            tag = " [M]" if d['is_maverick'] else ""
            print(f"    {d['agent_name']}: {d['total_trades']} trades{tag}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
