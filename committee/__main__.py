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
    generate_validation_slices,
)
from committee.manager import CommitteeManager
from committee.agent import (
    CommitteeAgent, calculate_conviction_threshold, recalculate_conviction_thresholds,
)
from committee.optimization import (
    calculate_coefficient_correlations, committee_objective,
    optimize_committee, interactive_correlation_refinement,
    automatic_correlation_refinement,
)
from committee.validation import (
    run_validation, run_validation_sweep,
    run_quorum_sweep, run_conviction_sweep, run_combined_sweep,
    evaluate_committee_on_slice, evaluate_agent_on_slice,
)


# --- Phase 1: Draft Day ---

def run_draft(manager, loader, stats, holdout_info, deep=False, exhaustive=False, require_maverick=False):
    """
    Phase 1: Select committee from Global50 using coefficient correlation optimization.

    IMPORTANT: Correlation is calculated on VALIDATION data (not holdout) to prevent
    data leakage. The holdout period remains unseen until Phase 2 validation.

    Args:
        deep: If True, use automatic correlation refinement + slice improvement (top N candidates).
        exhaustive: If True, slice improvement tests ALL Global50 candidates (--draft-deep2).
        require_maverick: Force exactly one maverick in committee.
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
    result = optimize_committee(entries, corr_matrix, require_maverick=should_enforce_maverick,
                                exhaustive=exhaustive)

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

            conviction_threshold = calculate_conviction_threshold(agent_coeffs_2d)
            member_data = build_member_data(e, conviction_threshold)
            members_with_stats.append(member_data)
            print(f"  ✓ {e['run_name']}_{e['agent_id']}: "
                  f"conviction_threshold={conviction_threshold:.3f}")
        else:
            print(f"  ⚠ {e['run_name']}_{e['agent_id']}: No coefficient data available")

    # Phase 1c: Slice-based improvement (deep mode only)
    if deep and len(members_with_stats) == len(committee_indices):
        slice_top_n = None if exhaustive else 5
        improved_members, improved_indices = run_slice_improvement_pass(
            members_with_stats, entries, coefficients, valid_indices,
            corr_matrix, list(committee_indices), loader, stats,
            holdout_info, context_window_days,
            require_maverick=should_enforce_maverick,
            top_n_candidates=slice_top_n,
        )

        if improved_indices != list(committee_indices):
            members_with_stats = improved_members
            committee_indices = tuple(improved_indices)
            committee_members = [entries[i] for i in committee_indices]

            final_obj, final_score_sum, final_avg_corr, final_max_corr = committee_objective(
                committee_indices, entries, corr_matrix
            )

            n = len(committee_indices)
            committee_corr = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    committee_corr[i, j] = corr_matrix[committee_indices[i], committee_indices[j]]

    # Phase 1d: Maverick rotation (deep mode only, when maverick constraint active)
    if deep and should_enforce_maverick:
        rotated_members, rotated_indices = run_maverick_rotation(
            members_with_stats, entries, coefficients, valid_indices,
            corr_matrix, list(committee_indices), loader, stats,
            holdout_info, context_window_days,
        )

        if rotated_indices != list(committee_indices):
            members_with_stats = rotated_members
            committee_indices = tuple(rotated_indices)
            committee_members = [entries[i] for i in committee_indices]

            final_obj, final_score_sum, final_avg_corr, final_max_corr = committee_objective(
                committee_indices, entries, corr_matrix
            )

            n = len(committee_indices)
            committee_corr = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    committee_corr[i, j] = corr_matrix[committee_indices[i], committee_indices[j]]

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


def run_slice_improvement_pass(members_with_stats, entries, coefficients, valid_indices,
                                corr_matrix, committee_indices, loader, stats,
                                holdout_info, context_window_days,
                                require_maverick=False, max_rounds=5, top_n_candidates=5):
    """
    Phase 1c: Slice-based committee improvement.

    After correlation refinement, evaluates the committee on validation slices,
    finds the worst-performing slice, identifies the weakest member on that slice,
    and tries replacing them with candidates from the Global50 pool that improve
    overall mean fitness across all slices.

    Args:
        members_with_stats: List of member dicts with conviction thresholds
        entries: Full list of Global50 entries
        coefficients: Pre-computed coefficient dict {index: 1D array} from correlation step
        valid_indices: Validation data indices (for reshaping coefficients)
        corr_matrix: Full NxN correlation matrix
        committee_indices: Current committee indices (list of ints into entries)
        loader: StockDataLoader instance
        stats: Normalization stats
        holdout_info: Holdout period info dict
        context_window_days: Context window size
        require_maverick: If True, preserve maverick type on swaps
        max_rounds: Maximum improvement rounds
        top_n_candidates: Number of top candidates to evaluate per round, or None for all

    Returns:
        (improved_members, improved_indices) — possibly unchanged if no improvement
    """
    episode_length = Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
    slices = generate_validation_slices(holdout_info, episode_length)
    num_stocks = Config.NUM_INVESTABLE_STOCKS
    num_days = len(valid_indices)

    current_members = list(members_with_stats)
    current_indices = list(committee_indices)

    exhaustive = top_n_candidates is None
    mode_label = "Exhaustive" if exhaustive else "Deep"

    print(f"\n{'='*60}")
    print(f"PHASE 1c: SLICE-BASED IMPROVEMENT ({mode_label})")
    print(f"{'='*60}")
    print(f"  Max rounds: {max_rounds}")
    print(f"  Candidates per round: {'ALL' if exhaustive else top_n_candidates}")
    print(f"  Validation slices: {len(slices)}")

    swap_count = 0

    for round_num in range(1, max_rounds + 1):
        print(f"\n  {'─'*56}")
        print(f"  Round {round_num}/{max_rounds}")
        print(f"  {'─'*56}")

        # 1. Evaluate committee on all slices
        print(f"    Evaluating committee on {len(slices)} slices...")
        slice_results = []
        for sl in slices:
            metrics = evaluate_committee_on_slice(
                current_members, loader, stats, context_window_days,
                sl['start'], sl['end']
            )
            slice_results.append({
                'index': sl['index'],
                'type': sl['type'],
                'start': sl['start'],
                'end': sl['end'],
                'fitness': metrics.get('fitness', 0.0),
                'roi': metrics.get('roi', 0.0),
                'num_trades': metrics.get('num_trades', 0),
            })
            gc.collect()
            torch.cuda.empty_cache()

        mean_fitness = np.mean([sr['fitness'] for sr in slice_results])

        for sr in slice_results:
            print(f"      Slice {sr['index']} ({sr['type']}): "
                  f"fitness={sr['fitness']:.2f}, roi={sr['roi']:.2f}%")
        print(f"      Mean fitness: {mean_fitness:.2f}")

        # 2. Find worst slice
        worst_slice = min(slice_results, key=lambda x: x['fitness'])
        worst_sl = slices[worst_slice['index']]
        print(f"\n    Worst slice: {worst_slice['index']} ({worst_slice['type']}) "
              f"fitness={worst_slice['fitness']:.2f}")

        # 3. Evaluate each member individually on worst slice to find weakest
        print(f"    Evaluating {len(current_members)} agents individually on worst slice...")
        agent_performances = []
        for i, member in enumerate(current_members):
            filepath = get_agent_filepath(member, context_window_days)
            agent_metrics = evaluate_agent_on_slice(
                filepath, loader, stats, worst_sl['start'], worst_sl['end']
            )
            maverick_tag = " [M]" if member.get('is_maverick', False) else ""
            agent_performances.append({
                'member_idx': i,
                'name': f"{member['run_name']}_{member['agent_id']}{maverick_tag}",
                'fitness': agent_metrics.get('fitness', 0.0),
                'is_maverick': member.get('is_maverick', False),
            })
            gc.collect()
            torch.cuda.empty_cache()

        agent_performances.sort(key=lambda x: x['fitness'])

        print(f"    Agent ranking on slice {worst_slice['index']}:")
        for ap in agent_performances:
            marker = " <<< weakest" if ap is agent_performances[0] else ""
            print(f"      {ap['name']}: fitness={ap['fitness']:.2f}{marker}")

        weakest = agent_performances[0]

        # 4. Find candidate replacements ranked by objective function
        weak_idx = weakest['member_idx']
        weak_is_maverick = weakest['is_maverick']

        remaining_global = [idx for i, idx in enumerate(current_indices) if i != weak_idx]

        candidates = []
        for cand_idx in range(len(entries)):
            if cand_idx in current_indices:
                continue
            if corr_matrix[cand_idx, cand_idx] != 1.0:
                continue
            if coefficients.get(cand_idx) is None:
                continue

            cand_is_maverick = entries[cand_idx].get('is_maverick', False)
            if require_maverick and cand_is_maverick != weak_is_maverick:
                continue

            trial_indices = tuple(sorted(remaining_global + [cand_idx]))
            obj, _, _, _ = committee_objective(trial_indices, entries, corr_matrix)
            candidates.append({
                'idx': cand_idx,
                'objective': obj,
                'entry': entries[cand_idx],
            })

        candidates.sort(key=lambda x: x['objective'], reverse=True)
        if top_n_candidates is not None:
            candidates = candidates[:top_n_candidates]

        if not candidates:
            print(f"    No valid candidates found. Stopping.")
            break

        scope = "ALL" if top_n_candidates is None else f"top {len(candidates)}"
        print(f"\n    Testing {scope} ({len(candidates)}) candidates to replace {weakest['name']}...")

        # 5. Evaluate each candidate via full committee simulation on all slices
        best_candidate = None
        best_mean_fitness = mean_fitness
        best_new_members = None
        best_new_indices = None

        for c in candidates:
            cand_coeffs_1d = coefficients[c['idx']]
            cand_coeffs_2d = cand_coeffs_1d.reshape(num_days, num_stocks)
            conviction_threshold = calculate_conviction_threshold(cand_coeffs_2d)
            new_member = build_member_data(c['entry'], conviction_threshold)

            modified_members = list(current_members)
            modified_members[weak_idx] = new_member

            modified_indices = list(current_indices)
            modified_indices[weak_idx] = c['idx']

            cand_fitnesses = []
            for sl in slices:
                metrics = evaluate_committee_on_slice(
                    modified_members, loader, stats, context_window_days,
                    sl['start'], sl['end']
                )
                cand_fitnesses.append(metrics.get('fitness', 0.0))
                gc.collect()
                torch.cuda.empty_cache()

            cand_mean = np.mean(cand_fitnesses)
            cand_name = f"{c['entry']['run_name']}_{c['entry']['agent_id']}"
            maverick_tag = " [M]" if c['entry'].get('is_maverick', False) else ""
            delta = cand_mean - mean_fitness
            print(f"      {cand_name}{maverick_tag}: mean_fitness={cand_mean:.2f} ({delta:+.2f})")

            if cand_mean > best_mean_fitness:
                best_mean_fitness = cand_mean
                best_candidate = c
                best_new_members = modified_members
                best_new_indices = modified_indices

        # 6. Accept or stop
        if best_candidate is not None:
            improvement = best_mean_fitness - mean_fitness
            cand_name = f"{best_candidate['entry']['run_name']}_{best_candidate['entry']['agent_id']}"
            swap_count += 1
            print(f"\n    ✓ Swap #{swap_count}: {weakest['name']} → {cand_name}")
            print(f"      Mean fitness: {mean_fitness:.2f} → {best_mean_fitness:.2f} "
                  f"({improvement:+.2f})")
            current_members = best_new_members
            current_indices = best_new_indices
        else:
            print(f"\n    No candidate improves overall mean fitness. Stopping.")
            break

    print(f"\n{'='*60}")
    print(f"SLICE IMPROVEMENT COMPLETE: {swap_count} swap(s)")
    print(f"{'='*60}")

    return current_members, current_indices


def run_maverick_rotation(members_with_stats, entries, coefficients, valid_indices,
                           corr_matrix, committee_indices, loader, stats,
                           holdout_info, context_window_days):
    """
    Phase 1d: Maverick rotation.

    With only a handful of mavericks in the Global50 pool (MAVERICK_CAP=5),
    exhaustively test each one in the maverick slot to find which maverick
    maximizes overall mean fitness across validation slices.

    Args:
        members_with_stats: Current member dicts with conviction thresholds
        entries: Full list of Global50 entries
        coefficients: Pre-computed coefficient dict {index: 1D array}
        valid_indices: Validation data indices (for reshaping coefficients)
        corr_matrix: Full NxN correlation matrix
        committee_indices: Current committee indices (list of ints into entries)
        loader: StockDataLoader instance
        stats: Normalization stats
        holdout_info: Holdout period info dict
        context_window_days: Context window size

    Returns:
        (updated_members, updated_indices) — possibly unchanged if current maverick is optimal
    """
    episode_length = Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
    slices = generate_validation_slices(holdout_info, episode_length)
    num_stocks = Config.NUM_INVESTABLE_STOCKS
    num_days = len(valid_indices)

    current_members = list(members_with_stats)
    current_indices = list(committee_indices)

    # Find current maverick in committee
    current_mav_member_idx = None
    current_mav_global_idx = None
    for i, (member, global_idx) in enumerate(zip(current_members, current_indices)):
        if member.get('is_maverick', False):
            current_mav_member_idx = i
            current_mav_global_idx = global_idx
            break

    if current_mav_member_idx is None:
        print("\n  No maverick in current committee. Skipping maverick rotation.")
        return current_members, current_indices

    current_mav_entry = entries[current_mav_global_idx]
    current_mav_name = f"{current_mav_entry['run_name']}_{current_mav_entry['agent_id']}"

    # Find all other valid mavericks in Global50
    other_mavericks = []
    for idx, entry in enumerate(entries):
        if (entry.get('is_maverick', False) and
                idx != current_mav_global_idx and
                corr_matrix[idx, idx] == 1.0 and
                coefficients.get(idx) is not None):
            other_mavericks.append(idx)

    print(f"\n{'='*60}")
    print("PHASE 1d: MAVERICK ROTATION")
    print(f"{'='*60}")
    print(f"  Current maverick: {current_mav_name} [M]")
    print(f"  Alternative mavericks to test: {len(other_mavericks)}")

    if not other_mavericks:
        print("  No other valid mavericks available. Current maverick retained.")
        return current_members, current_indices

    # Evaluate baseline (current committee) on all slices
    print(f"\n  Evaluating current committee (baseline)...")
    baseline_fitnesses = []
    for sl in slices:
        metrics = evaluate_committee_on_slice(
            current_members, loader, stats, context_window_days,
            sl['start'], sl['end']
        )
        baseline_fitnesses.append(metrics.get('fitness', 0.0))
        gc.collect()
        torch.cuda.empty_cache()

    baseline_mean = np.mean(baseline_fitnesses)

    per_slice_baseline = ", ".join(f"{f:.1f}" for f in baseline_fitnesses)
    print(f"    {current_mav_name} [M] (current): "
          f"mean={baseline_mean:.2f}  [{per_slice_baseline}]")

    # Test each alternative maverick
    print(f"\n  Testing {len(other_mavericks)} alternative mavericks...")

    best_mean = baseline_mean
    best_mav_idx = current_mav_global_idx
    best_members = None
    best_indices = None

    for mav_idx in other_mavericks:
        mav_entry = entries[mav_idx]
        mav_name = f"{mav_entry['run_name']}_{mav_entry['agent_id']}"

        # Build conviction thresholds for this maverick
        coeffs_1d = coefficients[mav_idx]
        coeffs_2d = coeffs_1d.reshape(num_days, num_stocks)
        conviction_threshold = calculate_conviction_threshold(coeffs_2d)
        new_member = build_member_data(mav_entry, conviction_threshold)

        # Build modified committee
        modified_members = list(current_members)
        modified_members[current_mav_member_idx] = new_member
        modified_indices = list(current_indices)
        modified_indices[current_mav_member_idx] = mav_idx

        # Evaluate on all slices
        mav_fitnesses = []
        for sl in slices:
            metrics = evaluate_committee_on_slice(
                modified_members, loader, stats, context_window_days,
                sl['start'], sl['end']
            )
            mav_fitnesses.append(metrics.get('fitness', 0.0))
            gc.collect()
            torch.cuda.empty_cache()

        mav_mean = np.mean(mav_fitnesses)
        delta = mav_mean - baseline_mean

        per_slice = ", ".join(f"{f:.1f}" for f in mav_fitnesses)
        print(f"    {mav_name} [M]: mean={mav_mean:.2f} ({delta:+.2f})  [{per_slice}]")

        if mav_mean > best_mean:
            best_mean = mav_mean
            best_mav_idx = mav_idx
            best_members = modified_members
            best_indices = modified_indices

    # Report results
    print(f"\n  {'─'*56}")
    if best_mav_idx != current_mav_global_idx:
        best_entry = entries[best_mav_idx]
        best_name = f"{best_entry['run_name']}_{best_entry['agent_id']}"
        improvement = best_mean - baseline_mean
        print(f"  ✓ Maverick swap: {current_mav_name} → {best_name}")
        print(f"    Mean fitness: {baseline_mean:.2f} → {best_mean:.2f} ({improvement:+.2f})")
        return best_members, best_indices
    else:
        print(f"  Current maverick ({current_mav_name}) is already optimal.")
        return current_members, current_indices


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
    print("Enter simulation date range (format: DD-MM-YY or DD-MM-YYYY)")
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
    print(f"  Quorum:             {Config.COMMITTEE_QUORUM}")
    print(f"  Conviction (P):     P{roster.get('conviction_percentile', 95)}")

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
    print(f"    Quorum: {Config.COMMITTEE_QUORUM}  |  Conviction: P{roster.get('conviction_percentile', 95)}")
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
        print(f"    Avg Votes/Trade:   {cs.get('avg_consensus_votes', 0):.2f}")
        print(f"    Trades by Quorum:  {cs.get('trades_by_quorum', 0)}")
        print(f"    Trades by Conviction Only: {cs.get('trades_by_conviction_only', 0)}")

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


def run_swap_agent(manager, loader, stats, holdout_info, agent_to_swap, focus_slices=None):
    """Find best replacement candidates for a specific committee member.

    Args:
        focus_slices: Optional list of slice indices (e.g. [0, 3]) to focus improvement on.
                      When set, candidates are ranked by mean fitness improvement on those slices.
    """
    print("\n" + "="*60)
    print(f"COMMITTEE AGENT SWAP EVALUATION")
    print(f"Target: {agent_to_swap}")
    if focus_slices is not None:
        print(f"Focus Slices: {focus_slices}")
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

    # Validate focus_slices against actual slice count
    if focus_slices is not None:
        ep_len = Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
        all_slices_check = generate_validation_slices(holdout_info, ep_len)
        known_indices = set(s['index'] for s in all_slices_check)
        invalid_slices = [s for s in focus_slices if s not in known_indices]
        if invalid_slices:
            print(f"❌ Invalid slice indices: {invalid_slices}. Valid: {sorted(known_indices)}")
            return

        print(f"\n🎯 FOCUS MODE: optimising for slice(s) {focus_slices}")
        for s in all_slices_check:
            if s['index'] in focus_slices:
                bs = baseline_results['committee_slices'][s['index']]
                print(f"  Slice {s['index']} ({s['type']}): baseline fitness={bs['fitness']:.2f}, roi={bs['roi']:.2f}%")

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
        conviction_threshold = calculate_conviction_threshold(chosen_coeffs_2d)

        new_member = build_member_data(e, conviction_threshold)

        modified_members = []
        for m in roster['members']:
            mid = f"{m['run_name']}_{m['agent_id']}"
            if mid == agent_to_swap:
                modified_members.append(new_member)
            else:
                modified_members.append(m)

        validation_results = run_validation(
            manager, loader, stats, holdout_info, manager.context_window_days,
            members_override=modified_members, conviction_percentile=None,
            quiet=True, skip_exports=True
        )

        if validation_results:
            candidate_validation_results.append({
                'rank': rank + 1, 'candidate': c, 'entry': e,
                'agent_id': eid, 'results': validation_results['committee_aggregate'],
                'committee_slices': validation_results['committee_slices'],
            })

    # Re-rank by focused-slice improvement when focus_slices is active
    if focus_slices is not None and candidate_validation_results:
        baseline_slices_for_focus = baseline_results['committee_slices']
        focused_baseline_fit = np.mean([baseline_slices_for_focus[i]['fitness'] for i in focus_slices])
        focused_baseline_roi = np.mean([baseline_slices_for_focus[i]['roi'] for i in focus_slices])

        for cvr in candidate_validation_results:
            cand_slices = cvr['committee_slices']
            focused_new_fit = np.mean([cand_slices[i]['fitness'] for i in focus_slices])
            focused_new_roi = np.mean([cand_slices[i]['roi'] for i in focus_slices])
            cvr['focused_delta_fitness'] = focused_new_fit - focused_baseline_fit
            cvr['focused_delta_roi'] = focused_new_roi - focused_baseline_roi
            cvr['focused_fitness'] = focused_new_fit
            cvr['focused_roi'] = focused_new_roi

        candidate_validation_results.sort(key=lambda x: x['focused_delta_fitness'], reverse=True)
        for i, cvr in enumerate(candidate_validation_results):
            cvr['rank'] = i + 1

        print(f"\n{'='*60}")
        print(f"FOCUSED SLICE IMPACT (slices {focus_slices})")
        print(f"{'='*60}")
        print(f"  Baseline focused fitness: {focused_baseline_fit:.2f}  |  focused ROI: {focused_baseline_roi:.2f}%")
        print(f"\n{'Rank':<5} {'Agent':<25} {'FocFit':<10} {'ΔFocFit':<10} {'FocROI':<10} {'ΔFocROI':<10} {'AllFit':<10} {'ΔAllFit':<10}")
        print("-" * 100)
        for cvr in candidate_validation_results:
            agg = cvr['results']
            delta_all = agg['mean_fitness'] - baseline_agg['mean_fitness']
            print(f"{cvr['rank']:<5} {cvr['agent_id']:<25} "
                  f"{cvr['focused_fitness']:<10.2f} {cvr['focused_delta_fitness']:<+10.2f} "
                  f"{cvr['focused_roi']:<10.2f}% {cvr['focused_delta_roi']:<+10.2f} "
                  f"{agg['mean_fitness']:<10.2f} {delta_all:<+10.2f}")

    # Display comparison (overall)
    print(f"\n{'='*60}")
    print("CANDIDATE IMPACT ANALYSIS (all slices)")
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

    # Outgoing agent index (roster order = baseline individual_agents order)
    outgoing_index = next((i for i, mid in enumerate(current_ids) if mid == agent_to_swap), None)
    if outgoing_index is None:
        print("❌ Outgoing agent not found in roster.")
        return
    outgoing_slice_metrics = baseline_results['individual_agents'][outgoing_index]['slice_metrics']

    episode_length = Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
    slices = generate_validation_slices(holdout_info, episode_length)
    baseline_slices = baseline_results['committee_slices']

    def _print_per_slice_for_candidate(cvr):
        """Print per-slice agent (out vs cand) and committee (curr vs new) for one candidate."""
        chosen = cvr['candidate']
        chosen_entry = cvr['entry']
        cand_id = cvr['agent_id']
        new_committee_slices = cvr.get('committee_slices', [])

        # Candidate agent per-slice (evaluate now)
        print(f"\n  Evaluating candidate agent on each slice...")
        cand_filepath = get_agent_filepath(chosen_entry, manager.context_window_days)
        cand_slice_metrics = []
        for sl in slices:
            metrics = evaluate_agent_on_slice(cand_filepath, loader, stats, sl['start'], sl['end'])
            cand_slice_metrics.append({
                'slice': sl['index'], 'slice_type': sl['type'],
                'fitness': metrics.get('fitness', 0.0), 'roi': metrics.get('roi', 0.0),
                'num_trades': metrics.get('num_trades', 0),
            })

        focus_set = set(focus_slices) if focus_slices else set()

        print(f"\n  PER-SLICE: OUTGOING AGENT vs CANDIDATE AGENT")
        print(f"  (Out: {agent_to_swap}  →  Cand: {cand_id})")
        print("-" * 80)
        print(f"  {'':3} {'Slice':<8} {'Type':<12} {'Out Fit':<10} {'Cand Fit':<10} {'Out ROI':<10} {'Cand ROI':<10} {'Out Tr':<8} {'Cand Tr':<8}")
        print("-" * 80)
        for sl, out_m, cand_m in zip(slices, outgoing_slice_metrics, cand_slice_metrics):
            marker = ">>>" if sl['index'] in focus_set else "   "
            print(f"  {marker} {sl['index']:<8} {sl['type']:<12} {out_m['fitness']:<10.2f} {cand_m['fitness']:<10.2f} "
                  f"{out_m['roi']:<10.2f}% {cand_m['roi']:<10.2f}% {out_m.get('num_trades', 0):<8} {cand_m['num_trades']:<8}")
        print("-" * 80)

        print(f"\n  PER-SLICE: CURRENT COMMITTEE vs NEW COMMITTEE")
        print("-" * 80)
        print(f"  {'':3} {'Slice':<8} {'Type':<12} {'Curr Fit':<10} {'New Fit':<10} {'Curr ROI':<10} {'New ROI':<10} {'Curr Tr':<8} {'New Tr':<8}")
        print("-" * 80)
        for bs, ns in zip(baseline_slices, new_committee_slices):
            marker = ">>>" if bs['slice'] in focus_set else "   "
            print(f"  {marker} {bs['slice']:<8} {bs['slice_type']:<12} {bs['fitness']:<10.2f} {ns['fitness']:<10.2f} "
                  f"{bs['roi']:<10.2f}% {ns['roi']:<10.2f}% {bs.get('num_trades', 0):<8} {ns.get('num_trades', 0):<8}")
        print("-" * 80)

    # Interactive Swap: pick rank → see per-slice → confirm or pick another
    print(f"\n{'='*60}")
    print("SWAP SELECTION")
    print(f"{'='*60}")
    print("Enter a candidate rank to see per-slice comparison (agent vs agent, committee vs committee),")
    print("then confirm swap or pick another rank.")

    while True:
        choice = input(f"\nEnter candidate rank (1-{top_n}) to see details and swap, or Enter to cancel: ").strip()
        if not choice:
            print("Cancelled.")
            return
        if not choice.isdigit():
            print(f"Invalid input. Enter a number 1-{top_n} or Enter to cancel.")
            continue
        rank = int(choice)
        if rank < 1 or rank > top_n:
            print(f"Invalid rank. Enter 1-{top_n} or Enter to cancel.")
            continue

        cvr = next((c for c in candidate_validation_results if c['rank'] == rank), None)
        if not cvr:
            cvr = {
                'rank': rank, 'candidate': candidates[rank - 1], 'entry': candidates[rank - 1]['entry'],
                'agent_id': f"{candidates[rank - 1]['entry']['run_name']}_{candidates[rank - 1]['entry']['agent_id']}",
                'results': None, 'committee_slices': [],
            }
            # We need committee_slices for this candidate; we only have them for top_n. So cvr should always be in candidate_validation_results.
            print(f"❌ No validation data for rank {rank}. Choose 1-{top_n}.")
            continue

        print(f"\n{'='*60}")
        print(f"CANDIDATE RANK {rank}: {cvr['agent_id']}")
        print(f"{'='*60}")
        _print_per_slice_for_candidate(cvr)

        confirm = input("\nConfirm swap with this candidate? (y/n): ").strip().lower()
        if confirm == 'y':
            chosen = cvr['candidate']
            chosen_entry = cvr['entry']

            print(f"\nSwapping {agent_to_swap} ➔ {chosen_entry['run_name']}_{chosen_entry['agent_id']}...")

            chosen_coeffs = coefficients[chosen['index']]
            num_days = len(valid_indices)
            num_stocks = Config.NUM_INVESTABLE_STOCKS
            chosen_coeffs_2d = chosen_coeffs.reshape(num_days, num_stocks)
            conviction_threshold = calculate_conviction_threshold(chosen_coeffs_2d)

            new_member = build_member_data(chosen_entry, conviction_threshold)

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
            return
        # n or anything else: pick another rank (loop continues)
        print("Pick another candidate rank to compare, or Enter to cancel.")


def run_diagnose_slice(manager, loader, stats, holdout_info, focus_slices):
    """
    Diagnose problematic validation slices.

    Runs baseline validation then analyses the focused slices:
    1. Committee performance on the slice(s).
    2. Individual agent breakdown (sorted worst-to-best).
    3. Consensus stats for the slice(s).
    4. Focused quorum/conviction mini-sweep on just those slices.
    5. Automated recommendation (regime / agent / parameter problem).
    """
    context_window_days = manager.context_window_days

    episode_length = Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
    slices = generate_validation_slices(holdout_info, episode_length)
    known_indices = set(s['index'] for s in slices)
    invalid = [s for s in focus_slices if s not in known_indices]
    if invalid:
        print(f"❌ Invalid slice indices: {invalid}. Valid: {sorted(known_indices)}")
        return

    print("\n" + "="*70)
    print("SLICE DIAGNOSIS")
    print("="*70)
    print(f"  Context Window: {context_window_days} days")
    print(f"  Target Slices:  {focus_slices}")
    for sl in slices:
        if sl['index'] in focus_slices:
            start_date = loader.dates[sl['start']]
            end_date = loader.dates[sl['end'] - 1]
            print(f"    Slice {sl['index']} ({sl['type']}): indices {sl['start']}-{sl['end']-1}  "
                  f"[{start_date} to {end_date}]")

    # --- Run baseline validation to get per-agent and per-slice data ---
    print(f"\n{'='*70}")
    print("RUNNING BASELINE VALIDATION")
    print(f"{'='*70}")
    baseline_results = run_validation(
        manager, loader, stats, holdout_info, context_window_days,
        members_override=None, conviction_percentile=None,
        quiet=True, skip_exports=True,
    )
    if not baseline_results:
        print("❌ Failed to run baseline validation")
        return

    # === Section 1: Committee performance on focused slices ===
    print(f"\n{'='*70}")
    print("1. COMMITTEE PERFORMANCE ON TARGET SLICES")
    print(f"{'='*70}")
    print(f"\n  {'Slice':<8} {'Type':<12} {'Fitness':<10} {'ROI':<10} {'WinRate':<10} {'Trades':<8} {'Wins':<6} {'Losses':<8}")
    print("-" * 80)
    for sl_idx in focus_slices:
        cs = baseline_results['committee_slices'][sl_idx]
        print(f"  {cs['slice']:<8} {cs['slice_type']:<12} {cs['fitness']:<10.2f} "
              f"{cs['roi']:<10.2f}% {cs['win_rate']*100:<9.1f}% {cs['num_trades']:<8} "
              f"{cs.get('num_wins', 0):<6} {cs.get('num_losses', 0):<8}")
    print("-" * 80)

    # === Section 2: Individual agent breakdown per focused slice ===
    print(f"\n{'='*70}")
    print("2. INDIVIDUAL AGENT BREAKDOWN (sorted worst-to-best)")
    print(f"{'='*70}")

    roster = manager.load_roster()
    members = roster['members']
    individual_agents = baseline_results['individual_agents']

    for sl_idx in focus_slices:
        sl_info = slices[sl_idx]
        print(f"\n  --- Slice {sl_idx} ({sl_info['type']}) ---")
        print(f"  {'Agent':<30} {'Fitness':<10} {'ROI':<10} {'Trades':<8} {'WinRate':<10}")
        print("  " + "-" * 68)

        agent_performances = []
        for i, agent in enumerate(individual_agents):
            sm = agent['slice_metrics'][sl_idx]
            m = members[i]
            maverick_tag = " [M]" if m.get('is_maverick', False) else ""
            name = f"{m['run_name']}_{m['agent_id']}{maverick_tag}"
            agent_performances.append({
                'name': name, 'fitness': sm['fitness'],
                'roi': sm['roi'], 'num_trades': sm['num_trades'],
                'win_rate': sm['win_rate'],
            })

        agent_performances.sort(key=lambda x: x['fitness'])
        for ap in agent_performances:
            worst_marker = " <<< worst" if ap is agent_performances[0] else ""
            print(f"  {ap['name']:<30} {ap['fitness']:<10.2f} {ap['roi']:<10.2f}% "
                  f"{ap['num_trades']:<8} {ap['win_rate']*100:<9.1f}%{worst_marker}")
        print("  " + "-" * 68)

    # === Section 3: Consensus stats for focused slices ===
    print(f"\n{'='*70}")
    print("3. CONSENSUS STATS FOR TARGET SLICES")
    print(f"{'='*70}")

    for sl_idx in focus_slices:
        cs = baseline_results['committee_slices'][sl_idx]
        con = cs.get('consensus_stats', {})
        print(f"\n  --- Slice {sl_idx} ({cs['slice_type']}) ---")
        if 'note' in con:
            print(f"  {con['note']}")
        else:
            print(f"  Unanimity Rate:      {con.get('unanimity_pct', 0):.1f}%")
            print(f"  Avg Votes/Trade:     {con.get('avg_consensus_votes', 0):.2f}")
            print(f"  Trades by Quorum:    {con.get('trades_by_quorum', 0)}")
            print(f"  Trades by Conviction Only: {con.get('trades_by_conviction_only', 0)}")

    # === Section 4: Focused quorum/conviction mini-sweep ===
    print(f"\n{'='*70}")
    print("4. QUORUM / CONVICTION MINI-SWEEP (on target slices only)")
    print(f"{'='*70}")

    quorum_values = [2, 3, 4, 5]
    conviction_percentiles = [90, 95, 99]

    original_quorum = Config.COMMITTEE_QUORUM
    focused_slices_list = [slices[i] for i in focus_slices]

    sweep_results = []
    current_combo = (original_quorum, 95)

    print(f"\n  Testing {len(quorum_values)} quorum x {len(conviction_percentiles)} conviction combos "
          f"on {len(focus_slices)} slice(s)...")

    for q in quorum_values:
        for pct in conviction_percentiles:
            Config.COMMITTEE_QUORUM = q

            if pct != 95:
                sweep_members = recalculate_conviction_thresholds(
                    members, loader, stats, holdout_info, context_window_days, pct
                )
            else:
                sweep_members = members

            slice_fitnesses = []
            slice_rois = []
            slice_trades = []

            for sl in focused_slices_list:
                metrics = evaluate_committee_on_slice(
                    sweep_members, loader, stats, context_window_days,
                    sl['start'], sl['end']
                )
                slice_fitnesses.append(metrics.get('fitness', 0.0))
                slice_rois.append(metrics.get('roi', 0.0))
                slice_trades.append(metrics.get('num_trades', 0))

            mean_fit = np.mean(slice_fitnesses)
            mean_roi = np.mean(slice_rois)
            total_trades = sum(slice_trades)

            is_current = (q == current_combo[0] and pct == current_combo[1])
            sweep_results.append({
                'quorum': q, 'pct': pct,
                'fitness': mean_fit, 'roi': mean_roi,
                'trades': total_trades, 'is_current': is_current,
            })

    Config.COMMITTEE_QUORUM = original_quorum

    sweep_results.sort(key=lambda x: x['fitness'], reverse=True)
    best_sweep = sweep_results[0]

    print(f"\n  {'Quorum':<8} {'ConvP':<8} {'Fitness':<10} {'ROI':<10} {'Trades':<8} {'Note'}")
    print("-" * 60)
    for sr in sweep_results:
        note = ""
        if sr['is_current']:
            note = "<<< current"
        if sr is best_sweep and not sr['is_current']:
            note = "<<< best for slice"
        elif sr is best_sweep and sr['is_current']:
            note = "<<< current (best)"
        print(f"  {sr['quorum']:<8} P{sr['pct']:<7} {sr['fitness']:<10.2f} {sr['roi']:<10.2f}% {sr['trades']:<8} {note}")
    print("-" * 60)

    # === Section 5: Recommendation ===
    print(f"\n{'='*70}")
    print("5. RECOMMENDATION")
    print(f"{'='*70}")

    all_agent_fitnesses = []
    for sl_idx in focus_slices:
        for agent in individual_agents:
            sm = agent['slice_metrics'][sl_idx]
            all_agent_fitnesses.append(sm['fitness'])

    per_agent_mean_fit = []
    for i, agent in enumerate(individual_agents):
        mean_f = np.mean([agent['slice_metrics'][sl_idx]['fitness'] for sl_idx in focus_slices])
        per_agent_mean_fit.append((i, mean_f))

    per_agent_mean_fit.sort(key=lambda x: x[1])
    mean_all = np.mean([f for _, f in per_agent_mean_fit])
    std_all = np.std([f for _, f in per_agent_mean_fit]) if len(per_agent_mean_fit) > 1 else 0.0

    worst_idx, worst_fit = per_agent_mean_fit[0]
    worst_member = members[worst_idx]
    worst_name = f"{worst_member['run_name']}_{worst_member['agent_id']}"
    worst_maverick = " [M]" if worst_member.get('is_maverick', False) else ""

    all_negative = all(f < 0 for _, f in per_agent_mean_fit)
    worst_is_outlier = (std_all > 0) and (worst_fit < (mean_all - 2 * std_all))

    current_result = next(sr for sr in sweep_results if sr['is_current'])
    best_improvement = best_sweep['fitness'] - current_result['fitness']

    slices_str = ','.join(str(s) for s in focus_slices)

    if all_negative:
        print(f"\n  REGIME PROBLEM")
        print(f"  All {len(per_agent_mean_fit)} agents have negative mean fitness on the target slice(s).")
        print(f"  Mean individual fitness: {mean_all:.2f}")
        print(f"  This is likely a hostile market regime. No committee change will fix it.")
        if best_improvement > 0:
            print(f"\n  However, parameter tuning can reduce the damage:")
            print(f"  Best combo: quorum={best_sweep['quorum']}, conviction=P{best_sweep['pct']} "
                  f"(fitness improvement: {best_improvement:+.2f})")
            print(f"  Run: python -m committee --sweep-both \"{','.join(str(q) for q in quorum_values)}:"
                  f"{','.join(str(p) for p in conviction_percentiles)}\" to check impact on all slices.")
    elif worst_is_outlier:
        print(f"\n  AGENT PROBLEM")
        print(f"  {worst_name}{worst_maverick} is a significant outlier (fitness: {worst_fit:.2f}, "
              f"mean: {mean_all:.2f}, std: {std_all:.2f}).")
        print(f"  This agent is dragging down the committee on the target slice(s).")
        print(f"\n  Suggested fix:")
        print(f"  python -m committee --swap-agent \"{worst_name}\" --focus-slices \"{slices_str}\"")
    elif best_improvement > 2.0:
        print(f"\n  PARAMETER PROBLEM")
        print(f"  Tuning quorum/conviction significantly improves the target slice(s).")
        print(f"  Best combo: quorum={best_sweep['quorum']}, conviction=P{best_sweep['pct']} "
              f"(fitness improvement: {best_improvement:+.2f})")
        print(f"\n  Suggested fix:")
        print(f"  Run: python -m committee --sweep-both \"{','.join(str(q) for q in quorum_values)}:"
              f"{','.join(str(p) for p in conviction_percentiles)}\" to check impact on all slices.")
    else:
        print(f"\n  MIXED / NO CLEAR FIX")
        print(f"  No single agent is a clear outlier (worst: {worst_name}{worst_maverick} at {worst_fit:.2f}, "
              f"mean: {mean_all:.2f}).")
        print(f"  Parameter tuning improvement is modest ({best_improvement:+.2f}).")
        print(f"\n  Consider:")
        print(f"  1. Multi-agent swap: try replacing the bottom 2-3 performers one at a time.")
        print(f"     Weakest: {worst_name}{worst_maverick} ({worst_fit:.2f})")
        second_worst_idx, second_worst_fit = per_agent_mean_fit[1]
        second_worst_m = members[second_worst_idx]
        second_worst_name = f"{second_worst_m['run_name']}_{second_worst_m['agent_id']}"
        print(f"     2nd weakest: {second_worst_name} ({second_worst_fit:.2f})")
        print(f"  2. Accept partial loss on this regime if overall committee performance is healthy.")

    print("\n" + "="*70)


def print_committee_stats(manager):
    """
    Load committee_roster.json and display comprehensive statistics.
    Same pattern as global50.py --stats.
    """
    roster = manager.load_roster()
    print("\n" + "="*70)
    print("COMMITTEE STATISTICS")
    print("="*70)
    print(f"Context Window: {manager.context_window_days} days ({manager.context_window_id})")

    if roster is None:
        print("\n❌ Committee roster not found.")
        print("   Run with --draft first, or --mirror to download from cloud.")
        print("="*70)
        return

    members = roster.get('members', [])
    if not members:
        print("\n⚠ No members in committee (empty roster)")
        print("="*70)
        return

    def calc_stats(values):
        if not values:
            return {'min': 0, 'max': 0, 'mean': 0}
        return {
            'min': min(values),
            'max': max(values),
            'mean': np.mean(values),
        }

    num_mavericks = sum(1 for m in members if m.get('is_maverick', False))
    gauntlet_scores = [m.get('gauntlet_score', 0) for m in members]
    rois = [m.get('roi', 0) for m in members]
    expectancies = [m.get('expectancy', 0) for m in members]
    quality_ratios = [m.get('quality_ratio', 0) for m in members]
    win_ratios = [m.get('win_ratio', 0) for m in members]

    g_stats = calc_stats(gauntlet_scores)
    roi_stats = calc_stats(rois)
    exp_stats = calc_stats(expectancies)
    q_stats = calc_stats(quality_ratios)
    w_stats = calc_stats(win_ratios)

    corr = roster.get('correlation', {})
    avg_corr = corr.get('average', 0)
    max_corr = corr.get('max_pair', 0)
    obj = roster.get('objective_value', 0)
    agg = roster.get('aggregate_score', 0)
    conv_pct = roster.get('conviction_percentile', 95)
    generated = roster.get('generated_at', 'N/A')
    last_updated = roster.get('last_updated', 'N/A')

    print(f"\n📊 OVERVIEW")
    print("-" * 70)
    print(f"  Roster:          {manager.local_roster_path}")
    print(f"  Total Members:   {len(members):>4}")
    print(f"  Mavericks [M]:    {num_mavericks:>4} ({100*num_mavericks/len(members):.1f}%)")
    print(f"  Conviction Pct:  {conv_pct:>4}")
    print(f"  Generated:       {generated}")
    print(f"  Last Updated:    {last_updated}")

    print(f"\n📋 AGENTS IN ROSTER")
    print("-" * 70)
    for i, m in enumerate(members, 1):
        maverick_tag = " [M]" if m.get('is_maverick', False) else ""
        name = f"{m['run_name']}_{m['agent_id']}{maverick_tag}"
        print(f"  {i:>2}. {name}")

    print(f"\n🎯 CORRELATION & OBJECTIVE")
    print("-" * 70)
    print(f"  Avg Correlation: {avg_corr:>10.3f}")
    print(f"  Max Pair Corr:   {max_corr:>10.3f}")
    print(f"  Objective Value: {obj:>10.2f}")
    print(f"  Aggregate Score: {agg:>10.2f}")

    print(f"\n🎯 GAUNTLET SCORE (per member)")
    print("-" * 70)
    print(f"  Min:               {g_stats['min']:>10.2f}")
    print(f"  Max:               {g_stats['max']:>10.2f}")
    print(f"  Mean:              {g_stats['mean']:>10.2f}")

    print(f"\n📈 ROI % (per member)")
    print("-" * 70)
    print(f"  Min:               {roi_stats['min']:>10.2f}")
    print(f"  Max:               {roi_stats['max']:>10.2f}")
    print(f"  Mean:              {roi_stats['mean']:>10.2f}")

    print(f"\n📈 EXPECTANCY (per member)")
    print("-" * 70)
    print(f"  Min:               {exp_stats['min']:>10.2f}")
    print(f"  Max:               {exp_stats['max']:>10.2f}")
    print(f"  Mean:              {exp_stats['mean']:>10.2f}")

    print(f"\n📊 QUALITY RATIO (per member)")
    print("-" * 70)
    print(f"  Min:               {q_stats['min']:>10.3f}")
    print(f"  Max:               {q_stats['max']:>10.3f}")
    print(f"  Mean:              {q_stats['mean']:>10.3f}")

    print(f"\n📊 WIN RATIO (per member)")
    print("-" * 70)
    print(f"  Min:               {w_stats['min']:>10.3f}")
    print(f"  Max:               {w_stats['max']:>10.3f}")
    print(f"  Mean:              {w_stats['mean']:>10.3f}")

    print("\n" + "="*70)


# --- CLI ---

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Committee Selection from Global50")
    parser.add_argument('--draft', action='store_true',
                        help='Run Phase 1: Draft committee from Global50')
    parser.add_argument('--draft-deep', action='store_true',
                        help='Run Phase 1 with automatic deep refinement (no manual swaps)')
    parser.add_argument('--draft-deep2', action='store_true',
                        help='Like --draft-deep but slice improvement tests ALL Global50 candidates (exhaustive)')
    parser.add_argument('--maverick', action='store_true',
                        help='Force exactly one maverick agent in committee (use with --draft, --draft-deep, or --draft-deep2)')
    parser.add_argument('--validate', action='store_true',
                        help='Run Phase 2: Validate on holdout slices')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify data split')
    parser.add_argument('--stats', action='store_true',
                        help='Display comprehensive statistics about the current committee roster.')
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
    parser.add_argument('--conviction-percentile', type=float, default=None,
                        help='Override conviction percentile threshold (default: 95)')
    parser.add_argument('--sweep-conviction', type=str, default=None,
                        help='Sweep multiple conviction percentiles, comma-separated (e.g., "90,95,99,99.9")')
    parser.add_argument('--sweep-both', type=str, default=None,
                        help='Sweep both quorum and conviction (grid search). Format: "quorums:percentiles"')
    parser.add_argument('--swap-agent', type=str, default=None,
                        help='Evaluate and swap a specific agent (e.g. "run-name_id")')
    parser.add_argument('--focus-slices', type=str, default=None,
                        help='Focus swap search on specific slices, comma-separated (e.g., "0,3"). '
                             'Candidates are ranked by improvement on these slices. Use with --swap-agent.')
    parser.add_argument('--diagnose-slice', type=str, default=None,
                        help='Diagnose problematic slices: agent breakdown, consensus stats, '
                             'quorum/conviction mini-sweep, and recommendation. '
                             'Comma-separated slice indices (e.g., "0,3").')
    args = parser.parse_args()

    has_action = any([
        args.draft, args.draft_deep, args.draft_deep2, args.validate, args.verify_only,
        args.stats, args.mirror, args.update_maverick_flags,
        args.update_conviction is not None, args.simulate,
        args.sweep_quorum, args.sweep_conviction,
        args.sweep_both, args.swap_agent, args.diagnose_slice,
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

    # Handle --stats mode (no data loading needed)
    if args.stats:
        print_committee_stats(manager)
        exit(0)

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

    # Handle --diagnose-slice
    if args.diagnose_slice:
        diag_slices = [int(s.strip()) for s in args.diagnose_slice.split(',')]
        run_diagnose_slice(manager, loader, stats, holdout_info, diag_slices)
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
        focus_slices = None
        if args.focus_slices:
            focus_slices = [int(s.strip()) for s in args.focus_slices.split(',')]
        run_swap_agent(manager, loader, stats, holdout_info, args.swap_agent, focus_slices=focus_slices)
        exit(0)

    # Validate --maverick is used with draft
    if args.maverick and not (args.draft or args.draft_deep or args.draft_deep2):
        print("❌ --maverick must be used with --draft, --draft-deep, or --draft-deep2")
        exit(1)

    if args.draft or args.draft_deep or args.draft_deep2:
        roster = run_draft(manager, loader, stats, holdout_info,
                          deep=(args.draft_deep or args.draft_deep2),
                          exhaustive=args.draft_deep2,
                          require_maverick=args.maverick)
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

