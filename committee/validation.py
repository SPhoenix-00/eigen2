"""
Committee validation: slice evaluation, validation runner, and sweep functions.

Handles evaluating individual agents and committees on holdout/validation slices,
running full validation passes, and parameter sweep (quorum, conviction, combined).
"""

import numpy as np
import torch
import gc
import csv
from pathlib import Path
from tqdm import tqdm

from utils.config import Config
from committee.utils import (
    load_agent_actor_only, get_agent_filepath, sanitize_date_for_filename,
    parse_date_flexible, calculate_expectancy, format_quality_ratio,
    generate_validation_slices, aggregate_slice_metrics, aggregate_consensus_stats,
    build_metrics_result, enrich_closed_trades, get_validation_data,
)
from committee.agent import CommitteeAgent, recalculate_conviction_thresholds


# --- Evaluation ---

def evaluate_agent_on_slice(agent_path: Path, loader, stats,
                            slice_start_idx: int, slice_end_idx: int) -> dict:
    """
    Evaluate a single agent on a holdout slice using TradingEnvironment.

    Args:
        agent_path: Path to agent weights file
        loader: StockDataLoader instance
        stats: Normalization statistics
        slice_start_idx: Starting day index for slice
        slice_end_idx: Ending day index for slice (exclusive)

    Returns:
        Dict with comprehensive fitness metrics matching environment rules
    """
    from environment.trading_env import TradingEnvironment

    agent = load_agent_actor_only(agent_path, 0)
    if agent is None:
        return {'error': 'Failed to load agent'}

    env = TradingEnvironment(
        data_array=loader.data_array,
        dates=loader.dates,
        normalization_stats=stats,
        start_idx=slice_start_idx,
        end_idx=slice_end_idx,
        data_array_full=loader.data_array_full,
        is_training=False,
        gauntlet_mode=True
    )

    obs, info = env.reset()
    terminated = False

    with torch.no_grad():
        while not terminated:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(Config.DEVICE)
            action = agent.actor(obs_tensor).cpu().numpy()[0]
            obs, reward, terminated, truncated, info = env.step(action)

    summary = env.get_episode_summary()

    del agent
    del env
    torch.cuda.empty_cache()

    return build_metrics_result(summary)


def evaluate_committee_on_slice(members: list, loader, stats,
                                 context_window_days: int,
                                 slice_start_idx: int,
                                 slice_end_idx: int) -> dict:
    """
    Evaluate committee consensus on a holdout slice using TradingEnvironment.

    Args:
        members: List of committee member dicts with agent metadata and stats
        loader: StockDataLoader instance
        stats: Normalization statistics
        context_window_days: Context window for loading agent files
        slice_start_idx: Starting day index for slice
        slice_end_idx: Ending day index for slice (exclusive)

    Returns:
        Dict with comprehensive metrics matching environment rules
    """
    from environment.trading_env import TradingEnvironment

    committee = CommitteeAgent(members, context_window_days, track_consensus=True)

    env = TradingEnvironment(
        data_array=loader.data_array,
        dates=loader.dates,
        normalization_stats=stats,
        start_idx=slice_start_idx,
        end_idx=slice_end_idx,
        data_array_full=loader.data_array_full,
        is_training=False,
        gauntlet_mode=True
    )

    obs, info = env.reset()
    terminated = False
    committee.reset_episode()

    while not terminated:
        held_ids = list(env.open_positions.keys())
        action = committee.predict_action(obs, held_stock_ids=held_ids)
        obs, reward, terminated, truncated, info = env.step(action)

    summary = env.get_episode_summary()
    consensus_stats = committee.get_consensus_summary()

    # Enrich closed trades
    closed_trades = summary.get('closed_trades', [])
    enrich_closed_trades(closed_trades, loader)

    # Cleanup
    committee.cleanup()
    del committee
    del env
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return build_metrics_result(summary, consensus_stats=consensus_stats)


# --- Validation Runner ---

def run_validation(manager, loader, stats, holdout_info,
                   context_window_days: int, members_override: list = None,
                   conviction_percentile: int = None, skip_exports: bool = False,
                   quiet: bool = False) -> dict:
    """
    Run 5-slice validation on the committee: 3 slices on validation data, 2 slices on holdout data.

    Args:
        manager: CommitteeManager instance
        loader: StockDataLoader instance
        stats: Normalization stats dict
        holdout_info: Holdout period info dict
        context_window_days: Context window size
        members_override: Optional pre-computed members list (for A/B testing)
        conviction_percentile: Optional percentile override (recalculates thresholds if provided)
        skip_exports: If True, skip CSV/XLSX export and cloud upload
        quiet: If True, skip header and detailed results printing (for batch/swap use)

    Returns validation results dict with comprehensive metrics.
    """
    if not quiet:
        print("\n" + "="*60)
        print("PHASE 2: VALIDATION (3 Val Slices + 2 Holdout Slices)")
        print("="*60)

    roster = manager.load_roster()
    if roster is None:
        print(f"❌ No roster found. Run --draft first.")
        return None

    # Use override members if provided, otherwise use roster members
    if members_override is not None:
        members = members_override
        print(f"  Using custom members (A/B test mode)")
    elif conviction_percentile is not None:
        print(f"  Recalculating conviction thresholds at P{conviction_percentile}...")
        members = recalculate_conviction_thresholds(
            roster['members'], loader, stats, holdout_info,
            context_window_days, conviction_percentile
        )
    else:
        members = roster['members']

    num_slices = Config.COMMITTEE_VALIDATION_SLICES
    episode_length = Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

    # Generate slices
    slices = generate_validation_slices(holdout_info, episode_length)

    val_start_idx = holdout_info['val_start']
    val_end_idx = holdout_info['val_end']
    holdout_start_idx = holdout_info['holdout_start']
    holdout_end_idx = holdout_info['holdout_end']
    val_days = val_end_idx - val_start_idx + 1
    holdout_days = holdout_end_idx - holdout_start_idx + 1

    if not quiet:
        print(f"\n  Episode length: {episode_length} days ({Config.TRADING_PERIOD_DAYS} trading + {Config.SETTLEMENT_PERIOD_DAYS} settlement)")
        print(f"  Validation range: indices {val_start_idx} to {val_end_idx} ({val_days} days)")
        print(f"    → 3 episodes of {episode_length} days each")
        print(f"  Holdout range: indices {holdout_start_idx} to {holdout_end_idx} ({holdout_days} days)")
        print(f"    → 2 episodes of {episode_length} days each")
        print(f"  Total slices: {num_slices}")

    results = {
        'individual_agents': [],
        'committee_slices': [],
        'committee_aggregate': {
            'mean_fitness': 0.0,
            'mean_win_rate': 0.0,
            'mean_quality_ratio': 0.0,
            'mean_expectancy': 0.0,
            'mean_roi': 0.0,
            'total_trades': 0,
        },
        'consensus_summary': {
            'avg_unanimity_pct': 0.0,
            'avg_consensus_votes': 0.0,
        }
    }

    # Validate individual agents on each slice
    if not quiet:
        print(f"\nValidating {len(members)} individual agents...")

    for member in tqdm(members, desc="Agents", disable=quiet):
        entry = member
        filepath = get_agent_filepath(entry, context_window_days)

        agent_results = {
            'agent_id': entry['agent_id'],
            'run_name': entry['run_name'],
            'stored_gauntlet': entry['gauntlet_score'],
            'slice_metrics': [],
        }

        for sl in slices:
            start_date = loader.dates[sl['start']]
            end_date = loader.dates[sl['end'] - 1]

            metrics = evaluate_agent_on_slice(filepath, loader, stats, sl['start'], sl['end'])
            closed_trades_agent = metrics.get('closed_trades', [])
            agent_results['slice_metrics'].append({
                'slice': sl['index'],
                'slice_type': sl['type'],
                'start_idx': sl['start'],
                'end_idx': sl['end'] - 1,
                'start_date': start_date,
                'end_date': end_date,
                'num_days': sl['end'] - sl['start'],
                'fitness': metrics.get('fitness', 0.0),
                'win_rate': metrics.get('win_rate', 0.0),
                'quality_ratio': metrics.get('quality_ratio', 0.0),
                'expectancy': metrics.get('expectancy', 0.0),
                'roi': metrics.get('roi', 0.0),
                'num_trades': metrics.get('num_trades', 0),
                'num_wins': metrics.get('num_wins', 0),
                'num_losses': metrics.get('num_losses', 0),
                'raw_pnl': metrics.get('raw_pnl', 0.0),
                'peak_capital_employed': metrics.get('peak_capital_employed', 0.0),
                'closed_trades': closed_trades_agent,
            })

        # Calculate aggregates for this agent
        all_agent_closed_trades = []
        for m in agent_results['slice_metrics']:
            if m.get('closed_trades'):
                all_agent_closed_trades.extend(m['closed_trades'])

        agent_agg = aggregate_slice_metrics(agent_results['slice_metrics'], all_agent_closed_trades)
        agent_results['fresh_mean_fitness'] = agent_agg['mean_fitness']
        agent_results['fresh_mean_win_rate'] = agent_agg['mean_win_rate']
        agent_results['fresh_mean_quality_ratio'] = agent_agg['mean_quality_ratio']
        agent_results['fresh_mean_expectancy'] = agent_agg['mean_expectancy']
        agent_results['fresh_mean_roi'] = agent_agg['mean_roi']

        results['individual_agents'].append(agent_results)

    # Validate committee consensus on each slice
    if not quiet:
        print(f"\nValidating committee consensus...")

    all_committee_closed_trades = []

    for sl in slices:
        start_date = loader.dates[sl['start']]
        end_date = loader.dates[sl['end'] - 1]

        metrics = evaluate_committee_on_slice(
            members, loader, stats, context_window_days, sl['start'], sl['end']
        )

        closed_trades = metrics.get('closed_trades', [])
        if closed_trades:
            all_committee_closed_trades.extend(closed_trades)

        csv_filename = None

        # Export CSV/XLSX if not skipping
        if closed_trades and not skip_exports:
            csv_filename = _export_slice_trades(
                closed_trades, sl, start_date, end_date, manager, loader
            )

        slice_result = {
            'slice': sl['index'],
            'slice_type': sl['type'],
            'start_idx': sl['start'],
            'end_idx': sl['end'] - 1,
            'start_date': start_date,
            'end_date': end_date,
            'num_days': sl['end'] - sl['start'],
            'fitness': metrics.get('fitness', 0.0),
            'win_rate': metrics.get('win_rate', 0.0),
            'quality_ratio': metrics.get('quality_ratio', 0.0),
            'expectancy': metrics.get('expectancy', 0.0),
            'roi': metrics.get('roi', 0.0),
            'num_trades': metrics.get('num_trades', 0),
            'num_wins': metrics.get('num_wins', 0),
            'num_losses': metrics.get('num_losses', 0),
            'consensus_stats': metrics.get('consensus_stats', {'note': 'Consensus applied per-step'}),
            'raw_pnl': metrics.get('raw_pnl', 0.0),
            'peak_capital_employed': metrics.get('peak_capital_employed', 0.0),
            'csv_filename': csv_filename,
        }

        results['committee_slices'].append(slice_result)

    # Calculate committee aggregates using DRY helper
    results['committee_aggregate'] = aggregate_slice_metrics(
        results['committee_slices'], all_committee_closed_trades
    )

    # Aggregate consensus stats
    all_consensus_stats = [s['consensus_stats'] for s in results['committee_slices']]
    results['consensus_summary'] = aggregate_consensus_stats(all_consensus_stats)

    # Print comprehensive results
    if not quiet:
        _print_validation_results(results)

    return results


def run_validation_sweep(manager, loader, stats, holdout_info,
                          context_window_days: int, members_override: list = None) -> dict:
    """
    Lightweight validation wrapper for sweeps - only computes committee metrics.

    Skips individual agent validation, exports, and verbose printing.

    Args:
        manager: CommitteeManager instance
        loader: StockDataLoader instance
        stats: Normalization stats dict
        holdout_info: Holdout period info dict
        context_window_days: Context window size
        members_override: Optional pre-computed members list

    Returns:
        Dict with only committee_aggregate and consensus_summary
    """
    roster = manager.load_roster()
    if roster is None:
        print(f"❌ No roster found. Run --draft first.")
        return None

    members = members_override if members_override is not None else roster['members']

    episode_length = Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
    slices_list = generate_validation_slices(holdout_info, episode_length)

    committee_slices = []
    all_committee_closed_trades = []

    for sl in slices_list:
        metrics = evaluate_committee_on_slice(
            members, loader, stats, context_window_days, sl['start'], sl['end']
        )

        closed_trades = metrics.get('closed_trades', [])
        if closed_trades:
            all_committee_closed_trades.extend(closed_trades)

        committee_slices.append({
            'slice': sl['index'],
            'slice_type': sl['type'],
            'fitness': metrics.get('fitness', 0.0),
            'win_rate': metrics.get('win_rate', 0.0),
            'quality_ratio': metrics.get('quality_ratio', 0.0),
            'expectancy': metrics.get('expectancy', 0.0),
            'roi': metrics.get('roi', 0.0),
            'num_trades': metrics.get('num_trades', 0),
            'num_wins': metrics.get('num_wins', 0),
            'num_losses': metrics.get('num_losses', 0),
            'consensus_stats': metrics.get('consensus_stats', {'note': 'Consensus applied per-step'}),
            'raw_pnl': metrics.get('raw_pnl', 0.0),
            'peak_capital_employed': metrics.get('peak_capital_employed', 0.0),
        })

        del closed_trades
        del metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Use DRY helpers
    committee_aggregate = aggregate_slice_metrics(committee_slices, all_committee_closed_trades)
    all_consensus_stats = [s['consensus_stats'] for s in committee_slices]
    consensus_summary = aggregate_consensus_stats(all_consensus_stats)

    # Cleanup (keep committee_slices for sweep per-slice view)
    del all_committee_closed_trades
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    return {
        'committee_aggregate': committee_aggregate,
        'consensus_summary': consensus_summary,
        'committee_slices': committee_slices,
    }


# --- Sweep Functions ---

def run_quorum_sweep(manager, loader, stats, holdout_info,
                     context_window_days: int, quorum_values: list) -> dict:
    """
    Run validation with multiple quorum values for A/B testing.
    """
    print("\n" + "="*60)
    print("QUORUM SWEEP: A/B Testing Multiple Quorum Values")
    print("="*60)
    print(f"  Quorum values to test: {quorum_values}")
    print(f"  Committee size: {Config.COMMITTEE_SIZE}")

    original_quorum = Config.COMMITTEE_QUORUM
    all_results = {}

    for quorum in quorum_values:
        print(f"\n{'='*60}")
        print(f"TESTING QUORUM = {quorum}")
        print(f"{'='*60}")

        Config.COMMITTEE_QUORUM = quorum
        results = run_validation_sweep(manager, loader, stats, holdout_info, context_window_days)

        if results:
            all_results[quorum] = {
                'aggregate': results['committee_aggregate'],
                'consensus': results['consensus_summary'],
                'committee_slices': results.get('committee_slices', []),
            }

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    Config.COMMITTEE_QUORUM = original_quorum

    _print_sweep_summary("QUORUM", all_results, key_name="Quorum",
                         format_key=lambda k: f"{k}")
    _print_sweep_per_slice("QUORUM", all_results, "Quorum", lambda k: f"{k}")

    return all_results


def run_conviction_sweep(manager, loader, stats, holdout_info,
                         context_window_days: int, percentile_values: list) -> dict:
    """
    Run validation with multiple conviction percentile values for A/B testing.
    """
    print("\n" + "="*60)
    print("CONVICTION SWEEP: A/B Testing Multiple Percentiles")
    print("="*60)
    print(f"  Percentile values to test: {percentile_values}")
    print(f"  Committee size: {Config.COMMITTEE_SIZE}")
    print(f"  Current quorum: {Config.COMMITTEE_QUORUM}")

    roster = manager.load_roster()
    if roster is None:
        print(f"❌ No roster found. Run --draft first.")
        return None

    all_results = {}

    for percentile in percentile_values:
        print(f"\n{'='*60}")
        print(f"TESTING CONVICTION PERCENTILE = P{percentile}")
        print(f"{'='*60}")

        print(f"  Recalculating thresholds for {len(roster['members'])} members...")
        members_with_new_thresholds = recalculate_conviction_thresholds(
            roster['members'], loader, stats, holdout_info,
            context_window_days, percentile
        )

        results = run_validation_sweep(
            manager, loader, stats, holdout_info, context_window_days,
            members_override=members_with_new_thresholds
        )

        if results:
            all_results[percentile] = {
                'aggregate': results['committee_aggregate'],
                'consensus': results['consensus_summary'],
                'committee_slices': results.get('committee_slices', []),
            }

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _print_sweep_summary("CONVICTION", all_results, key_name="Percentile",
                         format_key=lambda k: f"P{k}")
    _print_sweep_per_slice("CONVICTION", all_results, "Percentile", lambda k: f"P{k}")

    return all_results


def run_combined_sweep(manager, loader, stats, holdout_info,
                       context_window_days: int, quorum_values: list, percentile_values: list) -> dict:
    """
    Run validation with multiple quorum values AND conviction percentiles in a grid search.
    """
    print("\n" + "="*60)
    print("COMBINED SWEEP: Grid Search Over Quorum & Conviction Percentile")
    print("="*60)
    print(f"  Quorum values to test: {quorum_values}")
    print(f"  Percentile values to test: {percentile_values}")
    print(f"  Total combinations: {len(quorum_values) * len(percentile_values)}")
    print(f"  Committee size: {Config.COMMITTEE_SIZE}")

    roster = manager.load_roster()
    if roster is None:
        print(f"❌ No roster found. Run --draft first.")
        return None

    original_quorum = Config.COMMITTEE_QUORUM
    all_results = {}
    total_combinations = len(quorum_values) * len(percentile_values)
    current_combination = 0

    # Load validation tensor ONCE before the sweep
    print(f"\n  Pre-loading validation tensor for threshold recalculation...")
    val_tensor, valid_indices = get_validation_data(loader, stats, holdout_info)
    val_tensor_cpu = val_tensor.cpu()
    del val_tensor
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Outer loop: each conviction percentile (threshold calculated once per percentile).
    # Inner loop: each quorum (no recalculation; just run validation).
    for percentile in percentile_values:
        print(f"\n{'='*60}")
        print(f"CONVICTION P{percentile} — calculating thresholds once, then sweeping quorums")
        print(f"{'='*60}")

        print(f"  Recalculating thresholds for {len(roster['members'])} members at P{percentile}...")
        members_with_new_thresholds = recalculate_conviction_thresholds(
            roster['members'], loader, stats, holdout_info,
            context_window_days, percentile, val_tensor_cpu=val_tensor_cpu
        )

        for quorum in quorum_values:
            current_combination += 1
            print(f"\n{'='*60}")
            print(f"COMBINATION {current_combination}/{total_combinations}: Quorum={quorum}, Conviction=P{percentile}")
            print(f"{'='*60}")

            Config.COMMITTEE_QUORUM = quorum

            results = run_validation_sweep(
                manager, loader, stats, holdout_info, context_window_days,
                members_override=members_with_new_thresholds
            )

            if results:
                all_results[(quorum, percentile)] = {
                    'aggregate': results['committee_aggregate'],
                    'consensus': results['consensus_summary'],
                    'committee_slices': results.get('committee_slices', []),
                }

            del results
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

        del members_with_new_thresholds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    # Cleanup pre-loaded tensor
    del val_tensor_cpu
    torch.cuda.empty_cache()
    gc.collect()

    Config.COMMITTEE_QUORUM = original_quorum

    # Print combined summary and per-slice view
    _print_combined_sweep_summary(all_results)
    _print_combined_sweep_per_slice(all_results)

    return all_results


# --- Private Helpers ---

def _export_slice_trades(closed_trades, sl, start_date, end_date, manager, loader):
    """Export slice trades to CSV/XLSX and upload to cloud. Returns csv_filename or None."""
    start_date_safe = sanitize_date_for_filename(start_date)
    end_date_safe = sanitize_date_for_filename(end_date)
    csv_filename = f"committee_slice_{sl['index']}_{sl['type']}_{start_date_safe}_to_{end_date_safe}.csv"
    csv_path = manager.local_committee_dir / csv_filename

    # Write CSV
    preferred_columns = [
        'stock_id', 'stock_name', 'entry_date', 'exit_date', 'days_held',
        'entry_price', 'exit_price', 'gain_pct', 'coefficient', 'reason',
        'base_reward', 'forced_exit_penalty', 'reward', 'day', 'action'
    ]
    actual_columns = list(closed_trades[0].keys())
    fieldnames = [c for c in preferred_columns if c in actual_columns]
    fieldnames += [c for c in actual_columns if c not in fieldnames]

    with open(csv_path, 'w', newline='') as f:
        if len(closed_trades) > 0:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(closed_trades)

    # Generate xlsx
    xlsx_path = csv_path.parent / f"{csv_path.stem}_positions.xlsx"
    try:
        from transform_trades_to_positions import transform_trades_to_positions
        slice_start_dt = parse_date_flexible(start_date)
        slice_end_dt = parse_date_flexible(end_date)
        transform_trades_to_positions(
            str(csv_path), str(xlsx_path),
            start_date=slice_start_dt, end_date=slice_end_dt
        )
        print(f"  ✓ Generated xlsx: {xlsx_path.name}")
    except ImportError:
        print(f"  ⚠ openpyxl not installed - skipping xlsx generation")
        xlsx_path = None
    except Exception as e:
        print(f"  ⚠ Failed to generate xlsx: {e}")
        xlsx_path = None

    # Upload to cloud
    if manager.cloud_sync.provider != "local":
        cloud_csv_path = f"{manager.cloud_committee_base}/{csv_filename}"
        if manager.cloud_sync.upload_file_verified(str(csv_path), cloud_csv_path):
            print(f"  ✓ Uploaded CSV: {csv_filename}")
        else:
            print(f"  ✗ Failed to upload CSV: {csv_filename}")

        if xlsx_path and xlsx_path.exists():
            xlsx_filename = xlsx_path.name
            cloud_xlsx_path = f"{manager.cloud_committee_base}/{xlsx_filename}"
            if manager.cloud_sync.upload_file_verified(str(xlsx_path), cloud_xlsx_path):
                print(f"  ✓ Uploaded xlsx: {xlsx_filename}")
            else:
                print(f"  ✗ Failed to upload xlsx: {xlsx_filename}")

    return csv_filename


def _print_validation_results(results):
    """Print comprehensive validation results."""
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")

    print("\n" + "-"*60)
    print("INDIVIDUAL AGENTS (Slice Performance)")
    print("-"*60)
    for agent in results['individual_agents']:
        print(f"\n  {agent['run_name']}_{agent['agent_id']}:")
        print(f"    Stored Gauntlet: {agent['stored_gauntlet']:.2f}")
        qr_str = format_quality_ratio(agent['fresh_mean_quality_ratio'])
        print(f"    Fresh Averages: fitness={agent['fresh_mean_fitness']:.2f}, "
              f"win_rate={agent['fresh_mean_win_rate']:.2%}, "
              f"quality_ratio={qr_str}, "
              f"expectancy={agent['fresh_mean_expectancy']:.6f}, "
              f"roi={agent['fresh_mean_roi']:.2f}%")
        for m in agent['slice_metrics']:
            print(f"      Slice {m['slice']} ({m['slice_type']}) [{m['start_date']} to {m['end_date']}, {m['num_days']} days]:")
            print(f"        fitness={m['fitness']:.2f}, win_rate={m['win_rate']:.2%}, "
                  f"quality_ratio={m['quality_ratio']:.3f}, expectancy={m['expectancy']:.6f}, "
                  f"roi={m['roi']:.2f}%, trades={m['num_trades']}")

    print("\n" + "-"*60)
    print("COMMITTEE PERFORMANCE (Per Slice)")
    print("-"*60)
    for s in results['committee_slices']:
        print(f"\n  Slice {s['slice']} ({s['slice_type'].upper()})")
        print(f"    Period: {s['start_date']} to {s['end_date']} ({s['num_days']} days)")
        print(f"    Fitness: {s['fitness']:.2f}")
        print(f"    Win Rate: {s['win_rate']:.2%}")
        print(f"    Quality Ratio: {s['quality_ratio']:.3f}")
        print(f"    Expectancy: {s['expectancy']:.6f}")
        print(f"    ROI: {s['roi']:.2f}% (Raw P&L: ${s['raw_pnl']:.2f}, Peak Capital: ${s['peak_capital_employed']:.2f})")
        print(f"    Trades: {s['num_trades']}")
        if s.get('csv_filename'):
            print(f"    CSV: {s['csv_filename']}")
        print(f"")

        cs = s['consensus_stats']
        if 'note' in cs:
            print(f"    Consensus: {cs['note']}")
        elif 'avg_consensus_votes' in cs:
            print(f"    Consensus:")
            print(f"      Unanimity: {cs['unanimity_pct']:.1f}%, "
                  f"Avg Votes: {cs['avg_consensus_votes']:.2f}")
            print(f"      By Quorum: {cs['trades_by_quorum']}, "
                  f"By Conviction Only: {cs.get('trades_by_conviction_only', 0)}")

    print("\n" + "-"*60)
    print("COMMITTEE AGGREGATE PERFORMANCE")
    print("-"*60)
    ca = results['committee_aggregate']
    print(f"  Mean Fitness: {ca['mean_fitness']:.2f}")
    print(f"  Mean Win Rate: {ca['mean_win_rate']:.2%}")
    qr = ca['mean_quality_ratio']
    print(f"  Mean Quality Ratio: {format_quality_ratio(qr)}")
    print(f"  Mean Expectancy: {ca['mean_expectancy']:.6f}")
    print(f"  Mean ROI: {ca['mean_roi']:.2f}%")
    print(f"  Total Trades (All Slices): {ca['total_trades']}")

    print("\n" + "-"*60)
    print("CONSENSUS SUMMARY")
    print("-"*60)
    cs_sum = results['consensus_summary']
    if 'note' in cs_sum:
        print(f"  {cs_sum['note']}")
    else:
        print(f"  Unanimity Rate: {cs_sum['avg_unanimity_pct']:.1f}%")
        print(f"  Avg Votes per Trade: {cs_sum['avg_consensus_votes']:.2f}")
        print(f"  Trades by Quorum: {cs_sum['total_trades_by_quorum']}")
        print(f"  Trades by Conviction Only: {cs_sum.get('total_trades_by_conviction_only', 0)}")

    print("\n" + "-"*60)
    print("COMMITTEE VS INDIVIDUAL COMPARISON")
    print("-"*60)
    committee_mean = ca['mean_fitness']
    individual_means = [agent['fresh_mean_fitness'] for agent in results['individual_agents']]
    best_individual = max(individual_means)
    worst_individual = min(individual_means)
    avg_individual = np.mean(individual_means)

    print(f"  Committee Fitness: {committee_mean:.2f}")
    print(f"  Best Individual: {best_individual:.2f}")
    print(f"  Worst Individual: {worst_individual:.2f}")
    print(f"  Avg Individual: {avg_individual:.2f}")
    print(f"  Committee vs Best: {committee_mean - best_individual:+.2f} ({((committee_mean / best_individual - 1) * 100):+.1f}%)")
    print(f"  Committee vs Avg: {committee_mean - avg_individual:+.2f} ({((committee_mean / avg_individual - 1) * 100):+.1f}%)")


def _print_sweep_summary(sweep_type, all_results, key_name, format_key):
    """Print sweep comparison summary (used by quorum and conviction sweeps)."""
    print(f"\n{'='*60}")
    print(f"{sweep_type} SWEEP COMPARISON SUMMARY")
    print(f"{'='*60}")

    print(f"\n{key_name:<12} {'Fitness':<10} {'Win Rate':<10} {'Quality':<10} {'Expectancy':<12} {'ROI':<10} {'Trades':<8}")
    print("-" * 82)

    for key in sorted(all_results.keys()):
        agg = all_results[key]['aggregate']
        qr_str = format_quality_ratio(agg['mean_quality_ratio'])
        print(f"{format_key(key):<12} {agg['mean_fitness']:<10.2f} {agg['mean_win_rate']*100:<10.1f}% "
              f"{qr_str:<10} {agg['mean_expectancy']:<12.6f} "
              f"{agg['mean_roi']:<10.2f}% {agg['total_trades']:<8}")

    print(f"\n{key_name:<12} {'By Quorum':<12} {'By Conviction':<16} {'Unanimity':<12} {'Avg Votes':<10}")
    print("-" * 72)

    for key in sorted(all_results.keys()):
        cs = all_results[key]['consensus']
        if 'note' not in cs:
            print(f"{format_key(key):<12} {cs.get('total_trades_by_quorum', 0):<12} "
                  f"{cs.get('total_trades_by_conviction_only', cs.get('total_trades_by_conviction', 0)):<16} "
                  f"{cs.get('avg_unanimity_pct', 0):<12.1f}% "
                  f"{cs.get('avg_consensus_votes', 0):<10.2f}")
        else:
            print(f"{format_key(key):<12} {cs['note']}")

    best_key = max(all_results.keys(), key=lambda k: all_results[k]['aggregate']['mean_fitness'])
    best_fitness = all_results[best_key]['aggregate']['mean_fitness']

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: {format_key(best_key)} achieved highest mean fitness ({best_fitness:.2f})")
    print(f"{'='*60}")


def _print_sweep_per_slice(sweep_type, all_results, key_name, format_key):
    """Print ROI and PnL (and fitness) per slice for each sweep configuration."""
    first = next(iter(all_results.values()), None)
    slices_list = first.get('committee_slices', []) if first else []
    if not slices_list:
        return
    print(f"\n{'='*60}")
    print(f"{sweep_type} SWEEP — VIEW SLICE PER SLICE (ROI & PnL)")
    print(f"{'='*60}")
    # Table header: key_name, then for each slice we show configs in rows, or one table per slice with configs as rows
    # Per-slice view: for each slice, one table with columns Config | Fitness | ROI % | Raw PnL
    for s in slices_list:
        sl_idx = s['slice']
        sl_type = s['slice_type']
        print(f"\n  Slice {sl_idx} ({sl_type})")
        print(f"  {key_name:<12} {'Fitness':<10} {'ROI %':<10} {'Raw PnL':<14}")
        print("  " + "-" * 50)
        for key in sorted(all_results.keys()):
            sl_data = all_results[key].get('committee_slices', [])
            row = next((x for x in sl_data if x.get('slice') == sl_idx), None)
            if row is None:
                continue
            pnl = row.get('raw_pnl', 0.0)
            print(f"  {format_key(key):<12} {row.get('fitness', 0):<10.2f} {row.get('roi', 0):<10.2f}% ${pnl:<12.2f}")
    print()


def _print_combined_sweep_summary(all_results):
    """Print combined sweep comparison summary."""
    print(f"\n{'='*60}")
    print("COMBINED SWEEP COMPARISON SUMMARY")
    print(f"{'='*60}")

    print(f"\n{'Quorum':<8} {'Percentile':<12} {'Fitness':<10} {'Win Rate':<10} {'Quality':<10} {'Expectancy':<12} {'ROI':<10} {'Trades':<8}")
    print("-" * 90)

    sorted_keys = sorted(all_results.keys(), key=lambda x: (x[0], x[1]))

    for key in sorted_keys:
        quorum, percentile = key
        agg = all_results[key]['aggregate']
        qr_str = format_quality_ratio(agg['mean_quality_ratio'])
        print(f"{quorum:<8} P{percentile:<11} {agg['mean_fitness']:<10.2f} {agg['mean_win_rate']*100:<10.1f}% "
              f"{qr_str:<10} {agg['mean_expectancy']:<12.6f} "
              f"{agg['mean_roi']:<10.2f}% {agg['total_trades']:<8}")

    print(f"\n{'Quorum':<8} {'Percentile':<12} {'By Quorum':<12} {'By Conviction':<16} {'Unanimity':<12} {'Avg Votes':<10}")
    print("-" * 80)

    for key in sorted_keys:
        quorum, percentile = key
        cs = all_results[key]['consensus']
        if 'note' not in cs:
            print(f"{quorum:<8} P{percentile:<11} {cs.get('total_trades_by_quorum', 0):<12} "
                  f"{cs.get('total_trades_by_conviction_only', cs.get('total_trades_by_conviction', 0)):<16} "
                  f"{cs.get('avg_unanimity_pct', 0):<12.1f}% "
                  f"{cs.get('avg_consensus_votes', 0):<10.2f}")
        else:
            print(f"{quorum:<8} P{percentile:<11} {cs['note']}")

    best_key = max(all_results.keys(), key=lambda k: all_results[k]['aggregate']['mean_fitness'])
    best_quorum, best_percentile = best_key
    best_fitness = all_results[best_key]['aggregate']['mean_fitness']
    best_consensus = all_results[best_key]['consensus']

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: Quorum={best_quorum}, Conviction=P{best_percentile}")
    print(f"  Achieved highest mean fitness: {best_fitness:.2f}")
    print(f"  Win Rate: {all_results[best_key]['aggregate']['mean_win_rate']*100:.2f}%")
    print(f"  ROI: {all_results[best_key]['aggregate']['mean_roi']:.2f}%")
    print(f"  Trades by Quorum: {best_consensus.get('total_trades_by_quorum', 0)}")
    print(f"  Trades by Conviction Only: {best_consensus.get('total_trades_by_conviction_only', 0)}")
    print(f"{'='*60}")


def _print_combined_sweep_per_slice(all_results):
    """Print ROI and PnL per slice for each combined (quorum, conviction) configuration."""
    first = next(iter(all_results.values()), None)
    slices_list = first.get('committee_slices', []) if first else []
    if not slices_list:
        return
    print(f"\n{'='*60}")
    print("COMBINED SWEEP — VIEW SLICE PER SLICE (ROI & PnL)")
    print(f"{'='*60}")
    sorted_keys = sorted(all_results.keys(), key=lambda x: (x[0], x[1]))
    for s in slices_list:
        sl_idx = s['slice']
        sl_type = s['slice_type']
        print(f"\n  Slice {sl_idx} ({sl_type})")
        print(f"  {'Quorum':<8} {'Pct':<8} {'Fitness':<10} {'ROI %':<10} {'Raw PnL':<14} {'Trades':<8}")
        print("  " + "-" * 62)
        for key in sorted_keys:
            quorum, percentile = key
            sl_data = all_results[key].get('committee_slices', [])
            row = next((x for x in sl_data if x.get('slice') == sl_idx), None)
            if row is None:
                continue
            pnl = row.get('raw_pnl', 0.0)
            trades = row.get('num_wins', 0) + row.get('num_losses', 0)
            print(f"  {quorum:<8} P{percentile:<7} {row.get('fitness', 0):<10.2f} {row.get('roi', 0):<10.2f}% ${pnl:<12.2f}  {trades}")
    print()

