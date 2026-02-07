"""
Committee CLI entry point.

Usage: python -m committee [--draft] [--validate] [--mirror] ...

Contains the CLI argument parser and high-level orchestration commands
(draft, simulate, swap, update_conviction) that tie together the lower-level modules.
"""

import argparse
import csv
import gc
import numpy as np
import torch
import pandas as pd
from datetime import datetime
from pathlib import Path

from utils.config import Config
from committee.utils import (
    load_normalization_stats, verify_data_split, load_global50_candidates,
    get_agent_filepath, get_validation_data, sanitize_date_for_filename,
    parse_date_input, parse_date_flexible, find_date_index,
    calculate_expectancy, format_quality_ratio,
    build_member_data, build_metrics_result, enrich_closed_trades,
)
from committee.manager import CommitteeManager
from committee.agent import (
    CommitteeAgent, calculate_agent_stats_vectorized, recalculate_conviction_thresholds,
)
from committee.optimization import (
    calculate_coefficient_correlations, committee_objective,
    optimize_committee, interactive_correlation_refinement,
    automatic_correlation_refinement,
)
from committee.validation import (
    run_validation, run_validation_sweep,
    run_quorum_sweep, run_conviction_sweep, run_combined_sweep,
    evaluate_committee_on_slice,
)


# --- Phase 1: Draft Day ---

def run_draft(manager, loader, stats, holdout_info, deep=False, require_maverick=False):
    """
    Phase 1: Select committee from Global50 using coefficient correlation optimization.

    IMPORTANT: Correlation is calculated on VALIDATION data (not holdout) to prevent
    data leakage. The holdout period remains unseen until Phase 2 validation.
    """
    print("\n" + "="*60)
    print("PHASE 1: DRAFT DAY (Global50 Selection)")
    print("="*60)

    context_window_days = manager.context_window_days

    # 1. Load Global50 candidates
    entries = load_global50_candidates(context_window_days)
    if not entries:
        print("❌ No agents in Global50")
        return None

    print(f"\n  Top 5 agents by gauntlet score:")
    for i, e in enumerate(entries[:5]):
        maverick_tag = " [M]" if e.get('is_maverick', False) else ""
        print(f"    {i+1}. {e['run_name']}_{e['agent_id']}{maverick_tag}: "
              f"score={e['gauntlet_score']:.2f}, roi={e.get('roi', 0):.2f}%")

    # Check maverick availability
    maverick_entries = [i for i, e in enumerate(entries) if e.get('is_maverick', False)]

    should_enforce_maverick = False
    if require_maverick:
        if not maverick_entries:
            print(f"\n❌ No maverick agents found in Global50!")
            print(f"   Use fix_global50.py --set-maverick to mark agents as mavericks first.")
            return None
        should_enforce_maverick = True
        print(f"\n  ✓ Maverick requirement enabled: Exactly 1 maverick required ({len(maverick_entries)} available in Global50)")
    elif maverick_entries:
        should_enforce_maverick = True
        print(f"\n  ✓ Default maverick constraint: Exactly 1 maverick required ({len(maverick_entries)} available in Global50)")
    else:
        print(f"\n  ⚠ No maverick agents found in Global50 (edge case: allowing 0 mavericks)")

    # 2. Prepare VALIDATION data for correlation calculation
    val_tensor, valid_indices = get_validation_data(loader, stats, holdout_info)
    print(f"\n  Validation tensor shape: {val_tensor.shape}")
    print(f"  (Using validation period for correlation - holdout remains unseen)")

    # 3. Calculate coefficient correlations
    corr_matrix, coefficients, _ = calculate_coefficient_correlations(
        entries, val_tensor, context_window_days
    )

    # 4. Optimize committee selection
    result = optimize_committee(entries, corr_matrix, require_maverick=should_enforce_maverick)

    if result is None:
        print("❌ Optimization failed")
        return None

    # 4b. Correlation refinement pass
    if deep:
        refined_indices = automatic_correlation_refinement(
            result['committee_indices'], entries, corr_matrix,
            require_maverick=should_enforce_maverick
        )
    else:
        refined_indices = interactive_correlation_refinement(
            result['committee_indices'], entries, corr_matrix,
            require_maverick=should_enforce_maverick
        )

    # Verify maverick requirement after refinement
    if should_enforce_maverick:
        maverick_count = sum(1 for i in refined_indices if entries[i].get('is_maverick', False))
        if maverick_count != 1:
            print(f"\n⚠ WARNING: Refinement changed maverick count to {maverick_count} (required: exactly 1)!")
            print(f"   Reverting to pre-refinement committee to preserve maverick requirement.")
            refined_indices = result['committee_indices']

    # Recalculate metrics after refinement
    final_obj, final_score_sum, final_avg_corr, final_max_corr = committee_objective(
        refined_indices, entries, corr_matrix
    )

    # 5. Build roster
    committee_indices = refined_indices
    committee_members = [entries[i] for i in committee_indices]

    # Build correlation matrix for committee
    n = len(committee_indices)
    committee_corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            committee_corr[i, j] = corr_matrix[committee_indices[i], committee_indices[j]]

    # Calculate conviction thresholds
    print(f"\nCalculating conviction thresholds for committee members...")
    members_with_stats = []
    for idx, e in enumerate(committee_members):
        original_idx = committee_indices[idx]
        agent_coeffs_1d = coefficients[original_idx]

        if agent_coeffs_1d is not None:
            num_days = len(valid_indices)
            num_stocks = Config.NUM_INVESTABLE_STOCKS
            agent_coeffs_2d = agent_coeffs_1d.reshape(num_days, num_stocks)

            conviction_threshold_vector = calculate_agent_stats_vectorized(agent_coeffs_2d)
            member_data = build_member_data(e, conviction_threshold_vector)
            members_with_stats.append(member_data)
            print(f"  ✓ {e['run_name']}_{e['agent_id']}: "
                  f"p95_mean={np.mean(conviction_threshold_vector):.3f}")
        else:
            print(f"  ⚠ {e['run_name']}_{e['agent_id']}: No coefficient data available")

    roster_data = {
        'committee_size': len(members_with_stats),
        'members': members_with_stats,
        'aggregate_score': final_score_sum,
        'objective_value': final_obj,
        'correlation': {
            'average': final_avg_corr,
            'max_pair': final_max_corr,
            'matrix': committee_corr.tolist(),
        },
        'context_window_days': context_window_days,
        'holdout_period': {
            'start_idx': int(valid_indices[0]),
            'end_idx': int(valid_indices[-1]),
            'num_days': len(valid_indices),
        },
        'generated_at': str(datetime.now()),
    }

    manager.save_roster(roster_data, committee_corr)

    # Print committee
    print(f"\n{'='*60}")
    print("FINAL COMMITTEE")
    print(f"{'='*60}")

    maverick_count = 0
    for i, m in enumerate(roster_data['members']):
        maverick_tag = " [M]" if m.get('is_maverick', False) else ""
        if m.get('is_maverick', False):
            maverick_count += 1
        print(f"  {i+1}. {m['run_name']}_{m['agent_id']}{maverick_tag}: "
              f"score={m['gauntlet_score']:.2f}, roi={m['roi']:.2f}%")

    print(f"\n  Committee Summary:")
    print(f"    Total members: {len(roster_data['members'])}")
    print(f"    Maverick members: {maverick_count}")
    print(f"    Non-maverick members: {len(roster_data['members']) - maverick_count}")

    if should_enforce_maverick and maverick_count != 1:
        print(f"\n  ⚠ WARNING: Maverick count is {maverick_count} (required: exactly 1)!")
    elif not should_enforce_maverick and maverick_count == 0:
        print(f"\n  ℹ Note: No maverick agents in committee (none available in Global50)")
        print(f"    --multi mode requires at least one maverick agent.")
    elif not should_enforce_maverick and maverick_count > 0:
        print(f"\n  ⚠ WARNING: Unexpected state: {maverick_count} mavericks selected but constraint was not enforced")
    elif maverick_count > Config.MAVERICK_CAP:
        print(f"\n  ⚠ WARNING: {maverick_count} mavericks selected (exceeds cap of {Config.MAVERICK_CAP})")

    print(f"\n  Aggregate Score: {final_score_sum:.2f}")
    print(f"  Objective Value: {final_obj:.2f}")
    print(f"  Avg Correlation: {final_avg_corr:.3f}")
    print(f"  Max Pair Correlation: {final_max_corr:.3f}")

    return roster_data


# --- Simulation ---

def simulate_committee_continuous(members, loader, stats, context_window_days,
                                   start_idx, trading_end_idx, settlement_end_idx):
    """Run committee simulation continuously over an arbitrary time period."""
    from environment.trading_env import TradingEnvironment

    committee = CommitteeAgent(members, context_window_days, track_consensus=True)

    env = TradingEnvironment(
        data_array=loader.data_array,
        dates=loader.dates,
        normalization_stats=stats,
        start_idx=start_idx,
        end_idx=settlement_end_idx,
        trading_end_idx=trading_end_idx + 1,
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

    closed_trades = summary.get('closed_trades', [])
    enrich_closed_trades(closed_trades, loader)

    committee.cleanup()
    del env
    torch.cuda.empty_cache()

    return build_metrics_result(summary, consensus_stats=consensus_stats, extra_fields={
        'trading_days': trading_end_idx - start_idx + 1,
        'settlement_days': settlement_end_idx - trading_end_idx - 1,
    })


def run_simulation(manager, loader, stats, context_window_days):
    """Interactive simulation mode: Run committee over a user-specified date range."""
    print("\n" + "="*60)
    print("COMMITTEE SIMULATION MODE")
    print("="*60)

    roster = manager.load_roster()
    if roster is None:
        print("❌ No committee roster found. Run --draft first.")
        return None

    members = roster['members']
    print(f"\n✓ Loaded committee with {len(members)} members")

    # Show available date range
    first_date = loader.dates[0]
    last_date = loader.dates[-1]

    def _display_date(d):
        if isinstance(d, str):
            return d
        return pd.Timestamp(d).strftime('%d-%m-%y')

    print(f"\n  Dataset date range: {_display_date(first_date)} to {_display_date(last_date)}")
    print(f"  Total days in dataset: {len(loader.dates)}")
    print(f"\n  Simulation structure:")
    print(f"    Context window: {Config.CONTEXT_WINDOW_DAYS} days (required before first trade)")
    print(f"    Settlement period: {Config.SETTLEMENT_PERIOD_DAYS} days (auto-added after last trading day)")
    print(f"\n  NOTE: Unlike validation mode, simulation allows CONTINUOUS trading")
    print(f"        for any period length (not limited to {Config.TRADING_PERIOD_DAYS}-day episodes).")

    # Get first trading day
    print(f"\n" + "-"*60)
    print("Enter simulation date range (format: DD-MM-YY)")
    print("-"*60)

    earliest_first_idx = Config.CONTEXT_WINDOW_DAYS
    earliest_display = _display_date(loader.dates[earliest_first_idx])
    print(f"\n  Earliest valid first trading day: {earliest_display}")

    while True:
        first_day_input = input("\n  First trading day: ").strip()
        try:
            first_day_dt = parse_date_input(first_day_input)
            first_day_idx = find_date_index(loader, first_day_dt)
            if first_day_idx < Config.CONTEXT_WINDOW_DAYS:
                print(f"  ❌ Need at least {Config.CONTEXT_WINDOW_DAYS} days of context before first trading day.")
                continue
            break
        except ValueError as e:
            print(f"  ❌ {e}")

    latest_last_idx = len(loader.dates) - Config.SETTLEMENT_PERIOD_DAYS - 1
    if latest_last_idx <= first_day_idx:
        print(f"\n❌ Not enough data after {first_day_input} for trading + settlement.")
        return None

    latest_last_display = _display_date(loader.dates[latest_last_idx])
    print(f"\n  Latest valid last trading day: {latest_last_display}")

    while True:
        last_day_input = input("\n  Last trading day: ").strip()
        try:
            last_day_dt = parse_date_input(last_day_input)
            last_day_idx = find_date_index(loader, last_day_dt)
            if last_day_idx <= first_day_idx:
                print(f"  ❌ Last trading day must be after first trading day.")
                continue
            settlement_end_idx = last_day_idx + Config.SETTLEMENT_PERIOD_DAYS + 1
            if settlement_end_idx > len(loader.dates):
                print(f"  ❌ Not enough data for settlement period after {last_day_input}.")
                continue
            break
        except ValueError as e:
            print(f"  ❌ {e}")

    # Display simulation parameters
    actual_first_display = _display_date(loader.dates[first_day_idx])
    actual_last_trading_display = _display_date(loader.dates[last_day_idx])
    actual_settlement_end_display = _display_date(loader.dates[settlement_end_idx - 1])

    trading_days = last_day_idx - first_day_idx + 1
    settlement_days = Config.SETTLEMENT_PERIOD_DAYS
    total_days = trading_days + settlement_days

    print(f"\n" + "="*60)
    print("SIMULATION PARAMETERS")
    print("="*60)
    print(f"  First trading day:  {actual_first_display} (index {first_day_idx})")
    print(f"  Last trading day:   {actual_last_trading_display} (index {last_day_idx})")
    print(f"  Settlement ends:    {actual_settlement_end_display} (index {settlement_end_idx - 1})")
    print(f"  Trading period:     {trading_days} days")
    print(f"  Settlement period:  {settlement_days} days")
    print(f"  Total simulation:   {total_days} days")
    print(f"  Committee size:     {len(members)} members")

    confirm = input("\n  Proceed with simulation? [Y/n]: ").strip().lower()
    if confirm in ('n', 'no'):
        print("  Simulation cancelled.")
        return None

    print(f"\n" + "="*60)
    print("RUNNING SIMULATION")
    print("="*60)

    metrics = simulate_committee_continuous(
        members, loader, stats, context_window_days,
        start_idx=first_day_idx,
        trading_end_idx=last_day_idx,
        settlement_end_idx=settlement_end_idx
    )

    # Display results
    print(f"\n" + "="*60)
    print("SIMULATION RESULTS")
    print("="*60)
    print(f"\n  Period: {actual_first_display} to {actual_settlement_end_display}")
    print(f"    Trading: {actual_first_display} to {actual_last_trading_display} ({metrics.get('trading_days', trading_days)} days)")
    print(f"    Settlement: {metrics.get('settlement_days', settlement_days)} days")
    print(f"\n  Performance Metrics:")
    print(f"    Fitness:       {metrics.get('fitness', 0.0):.2f}")
    print(f"    Win Rate:      {metrics.get('win_rate', 0.0):.2%}")
    print(f"    Quality Ratio: {metrics.get('quality_ratio', 0.0):.3f}")
    print(f"    Expectancy:    {metrics.get('expectancy', 0.0):.6f}")
    print(f"    ROI:           {metrics.get('roi', 0.0):.2f}%")
    print(f"    Raw P&L:       ${metrics.get('raw_pnl', 0.0):.2f}")
    print(f"    Peak Capital:  ${metrics.get('peak_capital_employed', 0.0):.2f}")
    print(f"    Total Trades:  {metrics.get('num_trades', 0)}")

    cs = metrics.get('consensus_stats', {})
    if 'avg_consensus_votes' in cs:
        print(f"\n  Consensus Statistics:")
        print(f"    Unanimity Rate:    {cs.get('unanimity_pct', 0):.1f}%")
        print(f"    Min Consensus:     {cs.get('min_consensus_pct', 0):.1f}%")
        print(f"    Avg Votes/Trade:   {cs.get('avg_consensus_votes', 0):.2f}")
        print(f"    Trades by Quorum:  {cs.get('trades_by_quorum', 0)}")
        print(f"    Trades by Conviction: {cs.get('trades_by_conviction', 0)}")
        print(f"    Trades Vetoed:     {cs.get('trades_vetoed', 0)}")

    # Save trades to CSV
    closed_trades = metrics.get('closed_trades', [])
    if closed_trades:
        first_date_safe = sanitize_date_for_filename(actual_first_display).replace('-', '')
        last_date_safe = sanitize_date_for_filename(actual_last_trading_display).replace('-', '')
        csv_filename = f"simulation_{first_date_safe}_to_{last_date_safe}.csv"
        csv_path = manager.local_committee_dir / csv_filename

        preferred_columns = [
            'stock_id', 'stock_name', 'entry_date', 'exit_date', 'days_held',
            'entry_price', 'exit_price', 'gain_pct', 'coefficient', 'reason',
            'base_reward', 'forced_exit_penalty', 'reward', 'day', 'action'
        ]
        actual_columns = list(closed_trades[0].keys())
        fieldnames = [c for c in preferred_columns if c in actual_columns]
        fieldnames += [c for c in actual_columns if c not in fieldnames]

        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(closed_trades)

        print(f"\n  Trades CSV saved: {csv_path}")

        # Generate xlsx
        xlsx_path = csv_path.parent / f"{csv_path.stem}_positions.xlsx"
        print(f"\n  Generating position tracking Excel file...")
        try:
            from transform_trades_to_positions import transform_trades_to_positions
            sim_start = datetime.strptime(actual_first_display, "%d-%m-%y")
            sim_end = datetime.strptime(actual_settlement_end_display, "%d-%m-%y")
            transform_trades_to_positions(str(csv_path), str(xlsx_path), start_date=sim_start, end_date=sim_end)
            print(f"  ✓ Position tracking saved: {xlsx_path}")
        except ImportError:
            print(f"  ⚠ openpyxl not installed. Install with: pip install openpyxl")
        except Exception as e:
            print(f"  ⚠ Failed to generate Excel file: {e}")

        # Upload to cloud
        if manager.cloud_sync.provider != "local":
            print(f"\n  Syncing to cloud...")
            cloud_csv_path = f"{manager.cloud_committee_base}/{csv_filename}"
            if manager.cloud_sync.upload_file_verified(str(csv_path), cloud_csv_path):
                print(f"    ✓ Uploaded: {csv_filename}")
            else:
                print(f"    ✗ Failed to upload: {csv_filename}")

            xlsx_filename = f"{csv_path.stem}_positions.xlsx"
            cloud_xlsx_path = f"{manager.cloud_committee_base}/{xlsx_filename}"
            if xlsx_path.exists():
                if manager.cloud_sync.upload_file_verified(str(xlsx_path), cloud_xlsx_path):
                    print(f"    ✓ Uploaded: {xlsx_filename}")
                else:
                    print(f"    ✗ Failed to upload: {xlsx_filename}")
    else:
        print(f"\n  ⚠ No trades were executed during this simulation period.")

    print(f"\n" + "="*60)
    print("SIMULATION COMPLETE")
    print("="*60)

    return metrics


# --- Update & Swap ---

def update_conviction_percentile(manager, loader, stats, holdout_info,
                                  context_window_days, percentile):
    """Update conviction percentile in existing committee roster."""
    print("\n" + "="*60)
    print(f"UPDATING CONVICTION PERCENTILE TO P{percentile}")
    print("="*60)

    roster = manager.load_roster()
    if not roster:
        print("❌ No committee roster found. Run --draft first.")
        return False

    print(f"  Current roster: {len(roster['members'])} members")
    current_percentile = roster.get('conviction_percentile', 95)
    print(f"  Current percentile: P{current_percentile}")
    print(f"  Target percentile: P{percentile}")

    if current_percentile == percentile:
        print(f"\n✓ Roster already uses P{percentile}. No update needed.")
        return True

    print(f"\nRecalculating conviction thresholds at P{percentile}...")
    updated_members = recalculate_conviction_thresholds(
        roster['members'], loader, stats, holdout_info, context_window_days, percentile
    )

    roster['members'] = updated_members
    roster['conviction_percentile'] = percentile
    roster['last_updated'] = datetime.now().isoformat() + 'Z'

    print(f"\nSaving updated roster...")
    if manager.save_roster(roster):
        print(f"✓ Roster updated with P{percentile} conviction thresholds")
        print(f"✓ Synced to cloud")
        return True
    else:
        print(f"✗ Failed to save roster")
        return False


def run_swap_agent(manager, loader, stats, holdout_info, agent_to_swap):
    """Find best replacement candidates for a specific committee member."""
    print("\n" + "="*60)
    print(f"COMMITTEE AGENT SWAP EVALUATION")
    print(f"Target: {agent_to_swap}")
    print("="*60)

    roster = manager.load_roster()
    if not roster:
        print("❌ No committee roster found. Run --draft first.")
        return

    current_ids = [f"{m['run_name']}_{m['agent_id']}" for m in roster['members']]
    if agent_to_swap not in current_ids:
        print(f"❌ Agent {agent_to_swap} not found in current committee.")
        print(f"Current members: {', '.join(current_ids)}")
        return

    entries = load_global50_candidates(manager.context_window_days)
    if not entries:
        return

    entry_map = {f"{e['run_name']}_{e['agent_id']}": i for i, e in enumerate(entries)}

    current_indices = []
    target_idx = -1
    target_is_maverick = False

    for mid in current_ids:
        if mid in entry_map:
            idx = entry_map[mid]
            current_indices.append(idx)
            if mid == agent_to_swap:
                target_idx = idx
                target_is_maverick = entries[idx].get('is_maverick', False)
        else:
            print(f"⚠ Warning: Member {mid} not found in Global50 (skipping)")

    if target_idx == -1:
        print(f"❌ Target {agent_to_swap} not found in Global50.")
        return

    base_indices = [i for i in current_indices if i != target_idx]

    if target_is_maverick:
        print(f"  Target {agent_to_swap} is a Maverick [M]. Only Maverick candidates will be considered.")
    else:
        print(f"  Target {agent_to_swap} is NOT a Maverick. Only non-Maverick candidates will be considered.")

    # Calculate Correlations
    print("\nCalculating coefficient correlations (this may take a moment)...")
    val_tensor, valid_indices = get_validation_data(loader, stats, holdout_info)
    corr_matrix, coefficients, _ = calculate_coefficient_correlations(
        entries, val_tensor, manager.context_window_days
    )

    # Evaluate Current Committee
    curr_obj, curr_score, curr_avg, curr_max = committee_objective(tuple(current_indices), entries, corr_matrix)
    print(f"\nCurrent Committee Status:")
    print(f"  Objective: {curr_obj:.2f}")
    print(f"  Score Sum: {curr_score:.2f}")
    print(f"  Avg Corr:  {curr_avg:.4f}")

    # Find Candidates
    candidates = []
    for i in range(len(entries)):
        if i in current_indices:
            continue
        if corr_matrix[i, i] != 1.0:
            continue
        candidate_is_maverick = entries[i].get('is_maverick', False)
        if target_is_maverick != candidate_is_maverick:
            continue

        trial_indices = tuple(sorted(base_indices + [i]))
        obj, score, avg, mx = committee_objective(trial_indices, entries, corr_matrix)
        candidates.append({
            'index': i, 'entry': entries[i],
            'objective': obj, 'score': score, 'avg_corr': avg, 'max_corr': mx
        })

    candidates.sort(key=lambda x: x['objective'], reverse=True)

    # Display Top 10
    print(f"\n{'='*60}")
    print("TOP 10 CANDIDATES BY OBJECTIVE FUNCTION")
    print(f"{'='*60}")
    print(f"{'#':<3} {'Agent':<25} {'Obj':<10} {'Diff':<8} {'Score':<10} {'AvgCorr':<8} {'ROI':<7} {'M'}")
    print("-" * 90)

    for rank, c in enumerate(candidates[:10]):
        e = c['entry']
        eid = f"{e['run_name']}_{e['agent_id']}"
        diff = c['objective'] - curr_obj
        mav = "✓" if e.get('is_maverick') else ""
        print(f"{rank+1:<3} {eid:<25} {c['objective']:.2f}      {diff:+.2f}    {c['score']:.2f}      {c['avg_corr']:.4f}   {e.get('roi',0):.1f}%   {mav}")

    # Baseline validation
    print(f"\n{'='*60}")
    print("RUNNING BASELINE VALIDATION (Current Committee)")
    print(f"{'='*60}")
    baseline_results = run_validation(
        manager, loader, stats, holdout_info, manager.context_window_days,
        members_override=None, conviction_percentile=None
    )

    if not baseline_results:
        print("❌ Failed to run baseline validation")
        return

    baseline_agg = baseline_results['committee_aggregate']

    # Validate top candidates
    top_n = min(10, len(candidates))
    print(f"\n{'='*60}")
    print(f"RUNNING VALIDATION FOR TOP {top_n} CANDIDATES")
    print(f"{'='*60}")

    candidate_validation_results = []

    for rank, c in enumerate(candidates[:top_n]):
        e = c['entry']
        eid = f"{e['run_name']}_{e['agent_id']}"
        print(f"\n[{rank+1}/{top_n}] Validating candidate: {eid}")

        chosen_coeffs = coefficients[c['index']]
        num_days = len(valid_indices)
        num_stocks = Config.NUM_INVESTABLE_STOCKS
        chosen_coeffs_2d = chosen_coeffs.reshape(num_days, num_stocks)
        conviction_vec = calculate_agent_stats_vectorized(chosen_coeffs_2d)

        new_member = build_member_data(e, conviction_vec)

        modified_members = []
        for m in roster['members']:
            mid = f"{m['run_name']}_{m['agent_id']}"
            if mid == agent_to_swap:
                modified_members.append(new_member)
            else:
                modified_members.append(m)

        validation_results = run_validation(
            manager, loader, stats, holdout_info, manager.context_window_days,
            members_override=modified_members, conviction_percentile=None
        )

        if validation_results:
            candidate_validation_results.append({
                'rank': rank + 1, 'candidate': c, 'entry': e,
                'agent_id': eid, 'results': validation_results['committee_aggregate']
            })

    # Display comparison
    print(f"\n{'='*60}")
    print("CANDIDATE IMPACT ANALYSIS")
    print(f"{'='*60}")

    baseline_avg_corr = roster.get('correlation', {}).get('average', 0.0)
    baseline_max_corr = roster.get('correlation', {}).get('max_pair', 0.0)

    print(f"\n{'Rank':<5} {'Agent':<25} {'Fitness':<12} {'ΔFitness':<12} {'WinRate':<10} {'ROI':<8} {'AvgCorr':<10} {'Trades':<8}")
    print("-" * 100)

    print(f"{'BASE':<5} {'(Current)':<25} {baseline_agg['mean_fitness']:<12.2f} {'--':<12} "
          f"{baseline_agg['mean_win_rate']*100:<9.2f}% {baseline_agg['mean_roi']:<7.2f}% "
          f"{baseline_avg_corr:<10.4f} {baseline_agg['total_trades']:<8}")

    for cvr in candidate_validation_results:
        agg = cvr['results']
        c = cvr['candidate']
        delta_fitness = agg['mean_fitness'] - baseline_agg['mean_fitness']
        print(f"{cvr['rank']:<5} {cvr['agent_id']:<25} {agg['mean_fitness']:<12.2f} {delta_fitness:<+12.2f} "
              f"{agg['mean_win_rate']*100:<9.2f}% {agg['mean_roi']:<7.2f}% "
              f"{c['avg_corr']:<10.4f} {agg['total_trades']:<8}")

    # Interactive Swap
    print(f"\n{'='*60}")
    print("SWAP SELECTION")
    print(f"{'='*60}")

    choice = input(f"\nEnter candidate rank (1-{top_n}) to swap, or Enter to cancel: ")
    if not choice.isdigit():
        print("Cancelled.")
        return

    rank = int(choice)
    if 1 <= rank <= top_n:
        cvr = next((c for c in candidate_validation_results if c['rank'] == rank), None)
        if cvr:
            chosen = cvr['candidate']
            chosen_entry = cvr['entry']
        else:
            chosen = candidates[rank-1]
            chosen_entry = chosen['entry']

        print(f"\nSwapping {agent_to_swap} ➔ {chosen_entry['run_name']}_{chosen_entry['agent_id']}...")

        chosen_coeffs = coefficients[chosen['index']]
        num_days = len(valid_indices)
        num_stocks = Config.NUM_INVESTABLE_STOCKS
        chosen_coeffs_2d = chosen_coeffs.reshape(num_days, num_stocks)
        conviction_vec = calculate_agent_stats_vectorized(chosen_coeffs_2d)

        new_member = build_member_data(chosen_entry, conviction_vec)

        new_roster_members = []
        for m in roster['members']:
            mid = f"{m['run_name']}_{m['agent_id']}"
            if mid == agent_to_swap:
                new_roster_members.append(new_member)
            else:
                new_roster_members.append(m)

        roster['members'] = new_roster_members
        roster['aggregate_score'] = chosen['score']
        roster['objective_value'] = chosen['objective']
        roster['correlation']['average'] = chosen['avg_corr']
        roster['correlation']['max_pair'] = chosen['max_corr']
        roster['last_updated'] = str(datetime.now())

        new_indices = tuple(sorted(base_indices + [chosen['index']]))
        n_new = len(new_indices)
        sub_matrix = np.zeros((n_new, n_new))
        for r in range(n_new):
            for c_col in range(n_new):
                sub_matrix[r, c_col] = corr_matrix[new_indices[r], new_indices[c_col]]

        roster['correlation']['matrix'] = sub_matrix.tolist()
        manager.save_roster(roster, sub_matrix)
        print("✓ Swap complete!")
    else:
        print(f"❌ Invalid rank. Please enter a number between 1 and {top_n}.")


# --- CLI ---

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Committee Selection from Global50")
    parser.add_argument('--draft', action='store_true',
                        help='Run Phase 1: Draft committee from Global50')
    parser.add_argument('--draft-deep', action='store_true',
                        help='Run Phase 1 with automatic deep refinement (no manual swaps)')
    parser.add_argument('--maverick', action='store_true',
                        help='Force exactly one maverick agent in committee (use with --draft or --draft-deep)')
    parser.add_argument('--validate', action='store_true',
                        help='Run Phase 2: Validate on holdout slices')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify data split')
    parser.add_argument('--mirror', action='store_true',
                        help='Check cloud sync status, download missing files')
    parser.add_argument('--update-maverick-flags', action='store_true',
                        help='Update is_maverick flags in committee roster from Global50')
    parser.add_argument('--update-conviction', type=float, default=None,
                        help='Update conviction percentile in roster (e.g., 90, 95, 99, 99.9)')
    parser.add_argument('--simulate', action='store_true',
                        help='Simulate committee deployment over a custom date range')
    parser.add_argument('--quorum', type=int, default=None,
                        help=f'Override quorum threshold (default: {Config.COMMITTEE_QUORUM})')
    parser.add_argument('--sweep-quorum', type=str, default=None,
                        help='Sweep multiple quorum values, comma-separated (e.g., "2,3,4,5")')
    parser.add_argument('--conviction-percentile', type=int, default=None,
                        help='Override conviction percentile threshold (default: 95)')
    parser.add_argument('--sweep-conviction', type=str, default=None,
                        help='Sweep multiple conviction percentiles, comma-separated (e.g., "90,95,99,99.9")')
    parser.add_argument('--sweep-both', type=str, default=None,
                        help='Sweep both quorum and conviction (grid search). Format: "quorums:percentiles"')
    parser.add_argument('--swap-agent', type=str, default=None,
                        help='Evaluate and swap a specific agent (e.g. "run-name_id")')
    args = parser.parse_args()

    has_action = any([
        args.draft, args.draft_deep, args.validate, args.verify_only,
        args.mirror, args.update_maverick_flags, args.update_conviction is not None,
        args.simulate, args.sweep_quorum, args.sweep_conviction,
        args.sweep_both, args.swap_agent,
    ])

    if not has_action:
        parser.print_help()
        exit(0)

    print("Initializing Committee Engine...")
    Config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {Config.DEVICE}")

    context_window_days = Config.CONTEXT_WINDOW_DAYS
    manager = CommitteeManager(context_window_days)

    print(f"  Context Window: {context_window_days} days")
    print(f"  Cloud Provider: {manager.cloud_sync.provider}")
    if manager.cloud_sync.provider != "local":
        print(f"  Bucket: {manager.cloud_sync.bucket_name}")
        print(f"  Cloud Path: {manager.cloud_committee_base}/")

    # Handle --mirror mode (no data loading needed)
    if args.mirror:
        manager.check_mirror_status()
        exit(0)

    # Handle --update-maverick-flags mode (no data loading needed)
    if args.update_maverick_flags:
        manager.update_maverick_flags()
        exit(0)

    # Load data for other operations
    loader, stats = load_normalization_stats()

    # Verify data split
    print("\n" + "="*60)
    print("DATA INTEGRITY CHECK")
    print("="*60)

    is_valid, error_msg, holdout_info = verify_data_split(loader)

    if not is_valid:
        print(f"\n❌ Data split verification FAILED!")
        print(f"   Error: {error_msg}")
        exit(1)

    if args.verify_only:
        print("\n✓ Verification complete.")
        exit(0)

    # Handle --update-conviction
    if args.update_conviction is not None:
        if not (1.0 <= args.update_conviction <= 100.0):
            print("❌ Percentile must be between 1 and 100")
            exit(1)
        update_conviction_percentile(manager, loader, stats, holdout_info, context_window_days, args.update_conviction)
        exit(0)

    # Handle --swap-agent
    if args.swap_agent:
        run_swap_agent(manager, loader, stats, holdout_info, args.swap_agent)
        exit(0)

    # Validate --maverick is used with draft
    if args.maverick and not (args.draft or args.draft_deep):
        print("❌ --maverick must be used with --draft or --draft-deep")
        exit(1)

    if args.draft or args.draft_deep:
        roster = run_draft(manager, loader, stats, holdout_info,
                          deep=args.draft_deep, require_maverick=args.maverick)
        if roster and args.validate:
            gc.collect()
            torch.cuda.empty_cache()

    # Handle sweeps
    if args.sweep_quorum:
        quorum_values = [int(q.strip()) for q in args.sweep_quorum.split(',')]
        run_quorum_sweep(manager, loader, stats, holdout_info, context_window_days, quorum_values)
        exit(0)

    if args.sweep_conviction:
        percentile_values = [float(p.strip()) for p in args.sweep_conviction.split(',')]
        run_conviction_sweep(manager, loader, stats, holdout_info, context_window_days, percentile_values)
        exit(0)

    if args.sweep_both:
        if ':' not in args.sweep_both:
            print("❌ --sweep-both format error. Expected: 'quorums:percentiles' (e.g., '2,3,4:90,95,99')")
            exit(1)
        parts = args.sweep_both.split(':', 1)
        quorum_values = [int(q.strip()) for q in parts[0].split(',')]
        percentile_values = [float(p.strip()) for p in parts[1].split(',')]
        run_combined_sweep(manager, loader, stats, holdout_info, context_window_days, quorum_values, percentile_values)
        exit(0)

    # Handle --quorum override
    if args.quorum is not None:
        original_quorum = Config.COMMITTEE_QUORUM
        Config.COMMITTEE_QUORUM = args.quorum
        print(f"\n⚙ Quorum override: {original_quorum} → {args.quorum}")

    # Handle --conviction-percentile override
    conviction_percentile = args.conviction_percentile
    if conviction_percentile is not None:
        print(f"\n⚙ Conviction percentile override: 95 → P{conviction_percentile}")

    if args.validate:
        validation_results = run_validation(
            manager, loader, stats, holdout_info, context_window_days,
            conviction_percentile=conviction_percentile
        )

        if validation_results:
            if args.quorum is None and conviction_percentile is None:
                roster = manager.load_roster()
                if roster:
                    roster['validation'] = validation_results
                    manager.save_roster(roster)
                    print(f"\n✓ Validation results added and synced to cloud")
            else:
                overrides = []
                if args.quorum is not None:
                    overrides.append(f"quorum={args.quorum}")
                if conviction_percentile is not None:
                    overrides.append(f"conviction=P{conviction_percentile}")
                print(f"\n⚠ Skipping roster update ({', '.join(overrides)} used for A/B testing)")

    if args.simulate:
        run_simulation(manager, loader, stats, context_window_days)


if __name__ == "__main__":
    main()

