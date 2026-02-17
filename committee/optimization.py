"""
Committee optimization: objective function, exhaustive search, and refinement algorithms.

Includes correlation calculation, committee objective function, incremental exhaustive search,
and both interactive and automatic correlation refinement passes.
"""

import numpy as np
import torch
from itertools import combinations
from tqdm import tqdm
from math import comb

from utils.config import Config
from committee.utils import get_agent_filepath, load_agent_actor_only


# --- Correlation Calculation ---

def calculate_coefficient_correlations(entries: list, data_tensor: torch.Tensor,
                                        context_window_days: int) -> tuple:
    """
    Calculate pairwise coefficient correlations between agents.

    Args:
        entries: List of Global50 entry dicts
        data_tensor: Prepared data tensor [Days, Context, Stocks, Features]
                     (typically validation data to avoid holdout leakage)
        context_window_days: Context window for filepath lookup

    Returns:
        (correlation_matrix, coefficients_dict, loaded_agents_list)
    """
    n = len(entries)
    coefficients = {}  # entry_index -> coefficient array
    loaded_agents = []

    print(f"\nCalculating coefficient correlations for {n} agents...")

    for idx, entry in enumerate(tqdm(entries, desc="Loading agents")):
        filepath = get_agent_filepath(entry, context_window_days)

        if not filepath.exists():
            print(f"  ⚠ Agent file not found: {filepath}")
            coefficients[idx] = None
            loaded_agents.append(None)
            continue

        agent = load_agent_actor_only(filepath, entry['agent_id'])
        if agent is None:
            coefficients[idx] = None
            loaded_agents.append(None)
            continue

        with torch.no_grad():
            # Process in batches to avoid OOM on large validation sets
            batch_size = 32
            num_samples = data_tensor.shape[0]
            all_coefs = []

            for start_idx in range(0, num_samples, batch_size):
                end_idx = min(start_idx + batch_size, num_samples)
                batch = data_tensor[start_idx:end_idx]
                batch_actions = agent.actor(batch).cpu().numpy()
                # Extract coefficients (first output dimension)
                all_coefs.append(batch_actions[:, :, 0])

            # Concatenate and flatten to 1D for correlation: [Days * Stocks]
            coefs = np.concatenate(all_coefs, axis=0).flatten()

        coefficients[idx] = coefs
        loaded_agents.append(agent)

        del agent
        torch.cuda.empty_cache()

    # Build correlation matrix
    corr_matrix = np.zeros((n, n))

    valid_indices = [i for i in range(n) if coefficients[i] is not None]

    for i in valid_indices:
        for j in valid_indices:
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr = np.corrcoef(coefficients[i], coefficients[j])[0, 1]
                corr_matrix[i, j] = corr if not np.isnan(corr) else 0.0

    return corr_matrix, coefficients, loaded_agents


# --- Objective Function ---

def committee_objective(indices: tuple, entries: list, corr_matrix: np.ndarray) -> tuple:
    """
    Calculate committee objective value.

    Objective = Σ(gauntlet_scores) * (1 - avg_correlation)

    The correlation multiplier rewards anti-correlation (hedging) and penalizes
    positive correlation (redundancy):
      - avg_corr = -0.5 → multiplier = 1.5 (50% bonus for hedging)
      - avg_corr =  0.0 → multiplier = 1.0 (neutral)
      - avg_corr = +0.5 → multiplier = 0.5 (50% penalty for redundancy)

    Args:
        indices: Tuple of agent indices in the committee
        entries: Full list of Global50 entries
        corr_matrix: Full NxN correlation matrix

    Returns:
        (objective_value, score_sum, avg_correlation, max_correlation)
    """
    # Sum of gauntlet scores
    score_sum = sum(entries[i]['gauntlet_score'] for i in indices)

    # Extract submatrix for committee members
    n = len(indices)
    pairs = []
    max_corr = -1.0  # Track actual max (can be negative)

    for i_idx in range(n):
        for j_idx in range(i_idx + 1, n):
            i = indices[i_idx]
            j = indices[j_idx]
            corr = corr_matrix[i, j]
            pairs.append(corr)
            if corr > max_corr:
                max_corr = corr

    avg_corr = np.mean(pairs) if pairs else 0.0

    # Objective with directional correlation penalty
    if score_sum >= 0:
        multiplier = 1.0 - avg_corr
    else:
        multiplier = 1.0 + avg_corr

    objective = score_sum * multiplier

    return objective, score_sum, avg_corr, max_corr


def find_highest_correlation_pair(committee_indices: tuple, entries: list,
                                   corr_matrix: np.ndarray) -> tuple:
    """
    Find the pair of agents in the committee with the highest correlation.

    Args:
        committee_indices: Tuple of agent indices in the committee
        entries: Full list of Global50 entries
        corr_matrix: Full NxN correlation matrix

    Returns:
        (agent1_idx, agent2_idx, correlation_value, agent1_entry, agent2_entry)
        where agent1 has lower fitness than agent2 (agent1 is the swap candidate)
    """
    max_corr = -1.0
    max_pair = (None, None)

    n = len(committee_indices)
    for i_idx in range(n):
        for j_idx in range(i_idx + 1, n):
            i = committee_indices[i_idx]
            j = committee_indices[j_idx]
            corr = corr_matrix[i, j]

            if corr > max_corr:
                max_corr = corr
                max_pair = (i, j)

    if max_pair[0] is None:
        return None, None, 0.0, None, None

    # Determine which agent has lower fitness (swap candidate)
    i, j = max_pair
    fitness_i = entries[i]['gauntlet_score']
    fitness_j = entries[j]['gauntlet_score']

    if fitness_i <= fitness_j:
        return i, j, max_corr, entries[i], entries[j]
    else:
        return j, i, max_corr, entries[j], entries[i]


def find_best_swap_candidate(committee_indices: tuple, drop_idx: int, entries: list,
                              corr_matrix: np.ndarray,
                              optimize_for: str = 'objective',
                              preserve_maverick_type: bool = True) -> tuple:
    """
    Find the agent from the entire Global50 that best improves the committee
    when swapped in for the dropped agent.

    Args:
        committee_indices: Current committee indices
        drop_idx: Index of agent being dropped
        entries: Full list of Global50 entries
        corr_matrix: Full NxN correlation matrix
        optimize_for: What to optimize - 'objective' (default) or 'max_corr'
        preserve_maverick_type: If True, only swap mavericks with mavericks

    Returns:
        (best_candidate_idx, new_committee_indices, new_objective, new_score_sum,
         new_avg_corr, new_max_corr, candidate_entry)
    """
    remaining = [idx for idx in committee_indices if idx != drop_idx]
    dropped_is_maverick = entries[drop_idx].get('is_maverick', False)

    best_candidate = None
    best_objective = float('-inf')
    best_max_corr = float('inf')
    best_new_indices = None
    best_score_sum = 0
    best_avg_corr = 0
    best_max_corr_result = 0

    valid_entries = [i for i in range(len(entries)) if corr_matrix[i, i] == 1.0]

    for candidate_idx in valid_entries:
        if candidate_idx in committee_indices:
            continue

        if preserve_maverick_type:
            candidate_is_maverick = entries[candidate_idx].get('is_maverick', False)
            if candidate_is_maverick != dropped_is_maverick:
                continue

        new_indices = tuple(sorted(remaining + [candidate_idx]))
        obj, score_sum, avg_corr, max_corr = committee_objective(new_indices, entries, corr_matrix)

        if optimize_for == 'max_corr':
            if max_corr < best_max_corr:
                best_max_corr = max_corr
                best_candidate = candidate_idx
                best_new_indices = new_indices
                best_objective = obj
                best_score_sum = score_sum
                best_avg_corr = avg_corr
                best_max_corr_result = max_corr
        else:
            if obj > best_objective:
                best_objective = obj
                best_candidate = candidate_idx
                best_new_indices = new_indices
                best_score_sum = score_sum
                best_avg_corr = avg_corr
                best_max_corr_result = max_corr

    if best_candidate is None:
        return None, None, 0, 0, 0, 0, None

    return (best_candidate, best_new_indices, best_objective, best_score_sum,
            best_avg_corr, best_max_corr_result, entries[best_candidate])


# --- Optimization ---

def optimize_committee(entries: list, corr_matrix: np.ndarray,
                       require_maverick: bool = False, exhaustive: bool = False) -> dict:
    """
    Find optimal committee using incremental exhaustive search.

    Standard mode (exhaustive=False):
        1. Start with top K agents (COMMITTEE_TOP_K_INITIAL)
        2. Exhaustive search for best COMMITTEE_SIZE combination
        3. Expand pool by 1 agent, re-optimize
        4. Stop after COMMITTEE_EARLY_STOP consecutive non-improvements

    Exhaustive mode (exhaustive=True) with maverick constraint:
        Per-maverick optimization: each of the 5 mavericks is tested as the
        fixed maverick slot while the remaining non-maverick slots are optimized
        from a much larger pool with more patient early stopping.

    Exhaustive mode without maverick constraint:
        Same incremental approach but with a larger initial pool and more patient
        early stopping.

    Args:
        entries: List of Global50 entry dicts
        corr_matrix: Full NxN correlation matrix
        require_maverick: If True, only consider combinations with exactly one maverick
        exhaustive: If True, use broader search (--draft-deep2)

    Returns:
        Dict with best committee info
    """
    committee_size = Config.COMMITTEE_SIZE

    valid_entries = [i for i in range(len(entries)) if corr_matrix[i, i] == 1.0]

    if require_maverick:
        valid_mavericks = [i for i in valid_entries if entries[i].get('is_maverick', False)]
        valid_non_mavericks = [i for i in valid_entries if not entries[i].get('is_maverick', False)]
        if not valid_mavericks:
            print(f"❌ No valid maverick agents available for committee selection!")
            return None
        if len(valid_entries) < committee_size:
            print(f"❌ Not enough valid agents ({len(valid_entries)}) for committee of {committee_size}")
            return None
        if len(valid_non_mavericks) < committee_size - 1:
            print(f"❌ Not enough non-maverick agents to form committee with exactly one maverick!")
            print(f"   Need at least {committee_size - 1} non-mavericks, have {len(valid_non_mavericks)}")
            return None

    if len(valid_entries) < committee_size:
        print(f"❌ Not enough valid agents ({len(valid_entries)}) for committee of {committee_size}")
        return None

    # Per-maverick exhaustive search
    if exhaustive and require_maverick:
        return _optimize_per_maverick(entries, corr_matrix, valid_entries, committee_size)

    # Incremental search — expanded params for exhaustive mode
    if exhaustive:
        initial_pool = min(35, len(valid_entries))
        early_stop = 15
    else:
        initial_pool = Config.COMMITTEE_TOP_K_INITIAL
        early_stop = Config.COMMITTEE_EARLY_STOP

    mode_label = "Exhaustive" if exhaustive else "Incremental"

    print(f"\n{'='*60}")
    print(f"COMMITTEE OPTIMIZATION ({mode_label})")
    print(f"{'='*60}")
    print(f"  Committee Size: {committee_size}")
    print(f"  Initial Pool: top {initial_pool} agents")
    print(f"  Early Stop: {early_stop} consecutive non-improvements")
    print(f"  Valid Agents: {len(valid_entries)}")
    if require_maverick:
        valid_mavericks = [i for i in valid_entries if entries[i].get('is_maverick', False)]
        print(f"  Maverick Requirement: Exactly 1 maverick required ({len(valid_mavericks)} available)")

    best_committee = None
    best_objective = float('-inf')
    best_score_sum = 0
    best_avg_corr = 0
    best_max_corr = 0
    consecutive_failures = 0
    current_pool_size = min(initial_pool, len(valid_entries))

    iteration = 0

    while consecutive_failures < early_stop and current_pool_size <= len(valid_entries):
        iteration += 1

        if iteration == 1:
            pool = valid_entries[:current_pool_size]
            num_combos = comb(len(pool), committee_size)
            combo_gen = combinations(pool, committee_size)
            print(f"\n  Iteration {iteration}: Pool size {current_pool_size}, "
                  f"searching {num_combos:,} combinations...")
        else:
            new_agent = valid_entries[current_pool_size - 1]
            prev_pool = valid_entries[:current_pool_size - 1]
            num_combos = comb(len(prev_pool), committee_size - 1)
            combo_gen = (
                tuple(sorted(partial + (new_agent,)))
                for partial in combinations(prev_pool, committee_size - 1)
            )
            print(f"\n  Iteration {iteration}: +agent #{current_pool_size}, "
                  f"{num_combos:,} new combinations...")

        improved = False

        for combo in combo_gen:
            if require_maverick:
                maverick_count = sum(1 for i in combo if entries[i].get('is_maverick', False))
                if maverick_count != 1:
                    continue

            obj, score_sum, avg_corr, max_corr = committee_objective(combo, entries, corr_matrix)

            if obj > best_objective:
                best_objective = obj
                best_committee = combo
                best_score_sum = score_sum
                best_avg_corr = avg_corr
                best_max_corr = max_corr
                improved = True

        if improved:
            consecutive_failures = 0
            print(f"    ✓ New best: objective={best_objective:.2f}, "
                  f"score_sum={best_score_sum:.2f}, "
                  f"avg_corr={best_avg_corr:.3f}, max_corr={best_max_corr:.3f}")
        else:
            consecutive_failures += 1
            print(f"    No improvement ({consecutive_failures}/{early_stop})")

        current_pool_size += 1

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Final Objective: {best_objective:.2f}")
    print(f"  Aggregate Score: {best_score_sum:.2f}")
    print(f"  Avg Correlation: {best_avg_corr:.3f}")
    print(f"  Max Pair Correlation: {best_max_corr:.3f}")

    return {
        'committee_indices': best_committee,
        'objective': best_objective,
        'score_sum': best_score_sum,
        'avg_correlation': best_avg_corr,
        'max_correlation': best_max_corr,
    }


def _optimize_per_maverick(entries: list, corr_matrix: np.ndarray,
                            valid_entries: list, committee_size: int) -> dict:
    """
    Per-maverick exhaustive optimization for --draft-deep2.

    Fixes each maverick in the committee one at a time and runs an incremental
    exhaustive search for the best (committee_size - 1) non-maverick combination.
    This ensures every maverick gets a fair shot — not just whichever happened
    to land in the initial top-K pool.

    Uses a larger initial pool (top 30 non-mavericks) and more patient early
    stopping (10 consecutive non-improvements) than the standard search.

    Args:
        entries: Full Global50 entry list
        corr_matrix: Full NxN correlation matrix
        valid_entries: Pre-filtered list of valid agent indices
        committee_size: Target committee size

    Returns:
        Dict with best committee info, or None if optimization failed
    """
    valid_mavericks = [i for i in valid_entries if entries[i].get('is_maverick', False)]
    valid_non_mavericks = [i for i in valid_entries if not entries[i].get('is_maverick', False)]

    non_mav_slots = committee_size - 1
    initial_pool_nm = min(30, len(valid_non_mavericks))
    early_stop_nm = 10

    print(f"\n{'='*60}")
    print(f"COMMITTEE OPTIMIZATION (Per-Maverick Exhaustive)")
    print(f"{'='*60}")
    print(f"  Committee Size: {committee_size}")
    print(f"  Mavericks to test: {len(valid_mavericks)}")
    print(f"  Non-maverick pool: {len(valid_non_mavericks)}")
    print(f"  Non-maverick slots: {non_mav_slots}")
    print(f"  Per-maverick initial pool: top {initial_pool_nm} non-mavericks")
    print(f"  Per-maverick early stop: {early_stop_nm} consecutive non-improvements")

    total_combos_estimate = len(valid_mavericks) * comb(initial_pool_nm, non_mav_slots)
    print(f"  Estimated combinations (first pass): {total_combos_estimate:,}")

    best_overall = None
    best_obj_overall = float('-inf')

    for mav_num, mav_idx in enumerate(valid_mavericks, 1):
        mav_entry = entries[mav_idx]
        mav_name = f"{mav_entry['run_name']}_{mav_entry['agent_id']}"

        print(f"\n  {'─'*56}")
        print(f"  Maverick {mav_num}/{len(valid_mavericks)}: {mav_name} "
              f"(score={mav_entry['gauntlet_score']:.2f})")
        print(f"  {'─'*56}")

        best_committee_for_mav = None
        best_obj_for_mav = float('-inf')
        best_score_for_mav = 0
        best_avg_corr_for_mav = 0
        best_max_corr_for_mav = 0
        consecutive_failures = 0
        current_pool_size = min(initial_pool_nm, len(valid_non_mavericks))

        iteration = 0

        while consecutive_failures < early_stop_nm and current_pool_size <= len(valid_non_mavericks):
            iteration += 1

            if iteration == 1:
                pool = valid_non_mavericks[:current_pool_size]
                num_combos = comb(len(pool), non_mav_slots)
                nm_combo_gen = combinations(pool, non_mav_slots)
                print(f"    Iter {iteration}: pool {current_pool_size}, "
                      f"{num_combos:,} combinations...")
            else:
                new_agent = valid_non_mavericks[current_pool_size - 1]
                prev_pool = valid_non_mavericks[:current_pool_size - 1]
                num_combos = comb(len(prev_pool), non_mav_slots - 1)
                nm_combo_gen = (
                    partial + (new_agent,)
                    for partial in combinations(prev_pool, non_mav_slots - 1)
                )
                print(f"    Iter {iteration}: +agent #{current_pool_size}, "
                      f"{num_combos:,} new combinations...")

            improved = False

            for nm_combo in nm_combo_gen:
                combo = tuple(sorted(nm_combo + (mav_idx,)))
                obj, score_sum, avg_corr, max_corr = committee_objective(
                    combo, entries, corr_matrix
                )

                if obj > best_obj_for_mav:
                    best_obj_for_mav = obj
                    best_committee_for_mav = combo
                    best_score_for_mav = score_sum
                    best_avg_corr_for_mav = avg_corr
                    best_max_corr_for_mav = max_corr
                    improved = True

            if improved:
                consecutive_failures = 0
                print(f"      ✓ Best: obj={best_obj_for_mav:.2f}, "
                      f"score={best_score_for_mav:.2f}, "
                      f"avg_corr={best_avg_corr_for_mav:.3f}, "
                      f"max_corr={best_max_corr_for_mav:.3f}")
            else:
                consecutive_failures += 1
                print(f"      No improvement ({consecutive_failures}/{early_stop_nm})")

            current_pool_size += 1

        if best_committee_for_mav is not None:
            if best_obj_for_mav > best_obj_overall:
                best_obj_overall = best_obj_for_mav
                best_overall = {
                    'committee_indices': best_committee_for_mav,
                    'objective': best_obj_for_mav,
                    'score_sum': best_score_for_mav,
                    'avg_correlation': best_avg_corr_for_mav,
                    'max_correlation': best_max_corr_for_mav,
                }
                print(f"  ★ New overall best! obj={best_obj_overall:.2f}")
            else:
                print(f"    Best for this maverick: {best_obj_for_mav:.2f} "
                      f"(overall best: {best_obj_overall:.2f})")

    if best_overall is None:
        print("❌ Per-maverick optimization failed")
        return None

    # Report selected maverick
    selected_mav = [i for i in best_overall['committee_indices']
                     if entries[i].get('is_maverick', False)]
    selected_mav_entry = entries[selected_mav[0]] if selected_mav else None

    print(f"\n{'='*60}")
    print(f"OPTIMIZATION COMPLETE (Per-Maverick Exhaustive)")
    print(f"{'='*60}")
    print(f"  Final Objective: {best_overall['objective']:.2f}")
    print(f"  Aggregate Score: {best_overall['score_sum']:.2f}")
    print(f"  Avg Correlation: {best_overall['avg_correlation']:.3f}")
    print(f"  Max Pair Correlation: {best_overall['max_correlation']:.3f}")
    if selected_mav_entry:
        print(f"  Selected Maverick: {selected_mav_entry['run_name']}_"
              f"{selected_mav_entry['agent_id']} [M]")

    return best_overall


# --- Refinement ---

def interactive_correlation_refinement(committee_indices: tuple, entries: list,
                                         corr_matrix: np.ndarray,
                                         require_maverick: bool = False) -> tuple:
    """
    Interactive refinement pass: iteratively swap out high-correlation agents.

    IMPORTANT: Swaps preserve maverick type - mavericks only swap with mavericks,
    non-mavericks only swap with non-mavericks. This maintains the maverick count.

    Args:
        committee_indices: Initial committee indices from optimization
        entries: Full list of Global50 entries
        corr_matrix: Full NxN correlation matrix
        require_maverick: If True, ensure exactly one maverick remains after each swap

    Returns:
        Final committee indices after all swaps
    """
    current_indices = committee_indices
    swap_count = 0
    initial_maverick_count = sum(1 for i in current_indices if entries[i].get('is_maverick', False))

    print(f"\n{'='*60}")
    print("PHASE 1b: CORRELATION REFINEMENT (Interactive)")
    print(f"{'='*60}")
    print("This pass allows you to iteratively reduce correlation by swapping agents.")
    print("Press Enter for 'no' to skip, or type 'y' to swap.")
    if require_maverick:
        print(f"⚠ Maverick constraint: Exactly {initial_maverick_count} maverick(s) will be preserved.\n")
    else:
        print(f"ℹ Maverick type preservation: Swaps maintain maverick/non-maverick status.\n")

    while True:
        curr_obj, curr_score_sum, curr_avg_corr, curr_max_corr = committee_objective(
            current_indices, entries, corr_matrix
        )

        drop_idx, keep_idx, max_corr, drop_entry, keep_entry = find_highest_correlation_pair(
            current_indices, entries, corr_matrix
        )

        if drop_idx is None:
            print("  No valid pairs found in committee.")
            break

        (obj_candidate_idx, obj_new_indices, obj_new_obj, obj_new_score_sum,
         obj_new_avg_corr, obj_new_max_corr, obj_candidate_entry) = find_best_swap_candidate(
            current_indices, drop_idx, entries, corr_matrix, optimize_for='objective',
            preserve_maverick_type=True
        )

        (corr_candidate_idx, corr_new_indices, corr_new_obj, corr_new_score_sum,
         corr_new_avg_corr, corr_new_max_corr, corr_candidate_entry) = find_best_swap_candidate(
            current_indices, drop_idx, entries, corr_matrix, optimize_for='max_corr',
            preserve_maverick_type=True
        )

        if obj_candidate_idx is None and corr_candidate_idx is None:
            print("  ⚠ No valid replacement found for highest correlation pair.")
            break

        drop_maverick_tag = " [M]" if drop_entry.get('is_maverick', False) else ""
        keep_maverick_tag = " [M]" if keep_entry.get('is_maverick', False) else ""

        print(f"\n  {'─'*56}")
        print(f"  HIGHEST CORRELATION PAIR:")
        print(f"    Agent A: {drop_entry['run_name']}_{drop_entry['agent_id']}{drop_maverick_tag} "
              f"(fitness: {drop_entry['gauntlet_score']:.2f})")
        print(f"    Agent B: {keep_entry['run_name']}_{keep_entry['agent_id']}{keep_maverick_tag} "
              f"(fitness: {keep_entry['gauntlet_score']:.2f})")
        print(f"    Correlation: {max_corr:.4f}")

        same_candidate = (obj_candidate_idx == corr_candidate_idx)

        if same_candidate:
            candidate_idx = obj_candidate_idx
            new_indices = obj_new_indices
            new_obj = obj_new_obj
            new_score_sum = obj_new_score_sum
            new_avg_corr = obj_new_avg_corr
            new_max_corr = obj_new_max_corr
            candidate_entry = obj_candidate_entry

            candidate_maverick_tag = " [M]" if candidate_entry.get('is_maverick', False) else ""

            print(f"\n  PROPOSED SWAP:")
            print(f"    OUT: {drop_entry['run_name']}_{drop_entry['agent_id']}{drop_maverick_tag} "
                  f"(fitness: {drop_entry['gauntlet_score']:.2f})")
            print(f"    IN:  {candidate_entry['run_name']}_{candidate_entry['agent_id']}{candidate_maverick_tag} "
                  f"(fitness: {candidate_entry['gauntlet_score']:.2f})")

            new_agent_corr_with_kept = corr_matrix[candidate_idx, keep_idx]
            print(f"    New pair correlation: {new_agent_corr_with_kept:.4f} "
                  f"(was {max_corr:.4f}, Δ{new_agent_corr_with_kept - max_corr:+.4f})")

            print(f"\n  {'METRIC':<20} {'BEFORE':>12} {'AFTER':>12} {'CHANGE':>12}")
            print(f"  {'-'*56}")
            print(f"  {'Objective':<20} {curr_obj:>12.2f} {new_obj:>12.2f} "
                  f"{new_obj - curr_obj:>+12.2f}")
            print(f"  {'Aggregate Score':<20} {curr_score_sum:>12.2f} {new_score_sum:>12.2f} "
                  f"{new_score_sum - curr_score_sum:>+12.2f}")
            print(f"  {'Avg Correlation':<20} {curr_avg_corr:>12.4f} {new_avg_corr:>12.4f} "
                  f"{new_avg_corr - curr_avg_corr:>+12.4f}")
            print(f"  {'Max Correlation':<20} {curr_max_corr:>12.4f} {new_max_corr:>12.4f} "
                  f"{new_max_corr - curr_max_corr:>+12.4f}")

            print(f"\n  → Accept this swap?")
            confirm = input("    [y/N]: ").strip().lower()

            if confirm in ('y', 'yes'):
                new_maverick_count = sum(1 for i in new_indices if entries[i].get('is_maverick', False))
                if require_maverick and new_maverick_count != initial_maverick_count:
                    print(f"\n  ⚠ WARNING: Swap would change maverick count from {initial_maverick_count} to {new_maverick_count}!")
                    print(f"     This should not happen with maverick type preservation. Rejecting swap.")
                    continue

                current_indices = new_indices
                swap_count += 1
                print(f"\n  ✓ Swap #{swap_count} confirmed.")
                if require_maverick:
                    print(f"     Maverick count preserved: {new_maverick_count}")
            else:
                print(f"\n  ✗ Swap cancelled. Keeping current composition.")
                print(f"\n  ✓ Committee composition confirmed.")
                break
        else:
            # Two different options - show both
            print(f"\n  TWO SWAP OPTIONS AVAILABLE:")
            print(f"    OUT: {drop_entry['run_name']}_{drop_entry['agent_id']} "
                  f"(fitness: {drop_entry['gauntlet_score']:.2f})")

            obj_candidate_maverick_tag = " [M]" if obj_candidate_entry.get('is_maverick', False) else ""
            corr_candidate_maverick_tag = " [M]" if corr_candidate_entry.get('is_maverick', False) else ""

            print(f"\n  [1] BEST FOR OBJECTIVE:")
            print(f"      IN:  {obj_candidate_entry['run_name']}_{obj_candidate_entry['agent_id']}{obj_candidate_maverick_tag} "
                  f"(fitness: {obj_candidate_entry['gauntlet_score']:.2f})")
            obj_corr_with_kept = corr_matrix[obj_candidate_idx, keep_idx]
            print(f"      Pair corr: {obj_corr_with_kept:.4f} (Δ{obj_corr_with_kept - max_corr:+.4f})")
            print(f"      Objective: {obj_new_obj:.2f} (Δ{obj_new_obj - curr_obj:+.2f}), "
                  f"Max corr: {obj_new_max_corr:.4f} (Δ{obj_new_max_corr - curr_max_corr:+.4f})")

            print(f"\n  [2] BEST FOR MAX CORRELATION:")
            print(f"      IN:  {corr_candidate_entry['run_name']}_{corr_candidate_entry['agent_id']}{corr_candidate_maverick_tag} "
                  f"(fitness: {corr_candidate_entry['gauntlet_score']:.2f})")
            corr_corr_with_kept = corr_matrix[corr_candidate_idx, keep_idx]
            print(f"      Pair corr: {corr_corr_with_kept:.4f} (Δ{corr_corr_with_kept - max_corr:+.4f})")
            print(f"      Objective: {corr_new_obj:.2f} (Δ{corr_new_obj - curr_obj:+.2f}), "
                  f"Max corr: {corr_new_max_corr:.4f} (Δ{corr_new_max_corr - curr_max_corr:+.4f})")

            print(f"\n  → Choose swap: [1] objective, [2] max-corr, [N] skip")
            confirm = input("    [1/2/N]: ").strip().lower()

            if confirm == '1':
                new_maverick_count = sum(1 for i in obj_new_indices if entries[i].get('is_maverick', False))
                if require_maverick and new_maverick_count != initial_maverick_count:
                    print(f"\n  ⚠ WARNING: Swap would change maverick count. Rejecting swap.")
                    continue
                current_indices = obj_new_indices
                swap_count += 1
                print(f"\n  ✓ Swap #{swap_count} confirmed (objective-optimized).")
                if require_maverick:
                    print(f"     Maverick count preserved: {new_maverick_count}")
            elif confirm == '2':
                new_maverick_count = sum(1 for i in corr_new_indices if entries[i].get('is_maverick', False))
                if require_maverick and new_maverick_count != initial_maverick_count:
                    print(f"\n  ⚠ WARNING: Swap would change maverick count. Rejecting swap.")
                    continue
                current_indices = corr_new_indices
                swap_count += 1
                print(f"\n  ✓ Swap #{swap_count} confirmed (max-corr-optimized).")
                if require_maverick:
                    print(f"     Maverick count preserved: {new_maverick_count}")
            else:
                print(f"\n  ✗ Swap cancelled. Keeping current composition.")
                print(f"\n  ✓ Committee composition confirmed.")
                break

    if swap_count > 0:
        print(f"\n  {'='*56}")
        print(f"  REFINEMENT COMPLETE: {swap_count} swap(s) made")
        print(f"  {'='*56}")
    else:
        print(f"\n  No swaps made. Original committee retained.")

    return current_indices


def automatic_correlation_refinement(committee_indices: tuple, entries: list,
                                     corr_matrix: np.ndarray,
                                     max_iterations: int = 50,
                                     require_maverick: bool = False) -> tuple:
    """
    Automated refinement: iteratively swap to improve both objective and max_corr.

    Selection logic per iteration:
    1. Find highest correlation pair, identify swap candidate (lower fitness)
    2. Test all valid replacements
    3. Among candidates that IMPROVE objective, select the one with LOWEST max_corr
    4. Only apply if that candidate also LOWERS max_corr
    5. Stop when no candidate improves BOTH

    Args:
        committee_indices: Initial committee indices from optimization
        entries: Full list of Global50 entries
        corr_matrix: Full NxN correlation matrix
        max_iterations: Safety limit on number of swap iterations
        require_maverick: If True, ensure at least one maverick remains after each swap

    Returns:
        Final committee indices after all swaps
    """
    current_indices = committee_indices
    swap_count = 0

    start_obj, start_score_sum, start_avg_corr, start_max_corr = committee_objective(
        current_indices, entries, corr_matrix
    )

    print(f"\n{'='*60}")
    print("PHASE 1b: CORRELATION REFINEMENT (Automatic Deep)")
    print(f"{'='*60}")
    print(f"Starting: obj={start_obj:.2f}, max_corr={start_max_corr:.4f}")

    valid_entries = [i for i in range(len(entries)) if corr_matrix[i, i] == 1.0]

    for iteration in range(1, max_iterations + 1):
        curr_obj, curr_score_sum, curr_avg_corr, curr_max_corr = committee_objective(
            current_indices, entries, corr_matrix
        )

        drop_idx, keep_idx, max_corr, drop_entry, keep_entry = find_highest_correlation_pair(
            current_indices, entries, corr_matrix
        )

        if drop_idx is None:
            print(f"\n  No valid pairs found in committee.")
            break

        print(f"\n  Iteration {iteration}:")
        print(f"    Highest pair: {drop_entry['run_name']}_{drop_entry['agent_id']} <-> "
              f"{keep_entry['run_name']}_{keep_entry['agent_id']} (corr={max_corr:.4f})")

        remaining = [idx for idx in current_indices if idx != drop_idx]
        dropped_is_maverick = entries[drop_idx].get('is_maverick', False)
        remaining_maverick_count = sum(1 for idx in remaining if entries[idx].get('is_maverick', False))

        improving_candidates = []

        for candidate_idx in valid_entries:
            if candidate_idx in current_indices:
                continue

            new_indices = tuple(sorted(remaining + [candidate_idx]))

            if require_maverick:
                candidate_is_maverick = entries[candidate_idx].get('is_maverick', False)
                new_maverick_count = remaining_maverick_count + (1 if candidate_is_maverick else 0)
                if new_maverick_count != 1:
                    continue

            new_obj, new_score_sum, new_avg_corr, new_max_corr = committee_objective(
                new_indices, entries, corr_matrix
            )

            if new_obj > curr_obj:
                improving_candidates.append({
                    'idx': candidate_idx,
                    'new_indices': new_indices,
                    'obj': new_obj,
                    'max_corr': new_max_corr,
                    'entry': entries[candidate_idx]
                })

        print(f"    Testing {len(valid_entries) - len(current_indices)} candidates...")
        print(f"    Candidates improving objective: {len(improving_candidates)}")

        if not improving_candidates:
            print(f"    No candidate improves objective")
            print(f"    Stopping")
            break

        improving_candidates.sort(key=lambda x: x['max_corr'])
        best = improving_candidates[0]

        if best['max_corr'] >= curr_max_corr:
            print(f"    Best objective-improving candidate has max_corr={best['max_corr']:.4f} "
                  f"(not better than {curr_max_corr:.4f})")
            print(f"    Stopping")
            break

        print(f"    Best among those (lowest max_corr): "
              f"{best['entry']['run_name']}_{best['entry']['agent_id']}")
        print(f"      obj: {curr_obj:.2f} -> {best['obj']:.2f} ({best['obj'] - curr_obj:+.2f})")
        print(f"      max_corr: {curr_max_corr:.4f} -> {best['max_corr']:.4f} "
              f"({best['max_corr'] - curr_max_corr:+.4f})")
        print(f"    Swap applied")

        current_indices = best['new_indices']
        swap_count += 1

    final_obj, final_score_sum, final_avg_corr, final_max_corr = committee_objective(
        current_indices, entries, corr_matrix
    )

    print(f"\n{'='*60}")
    print(f"REFINEMENT COMPLETE: {swap_count} swap(s) made")
    print(f"{'='*60}")
    print(f"  Final: obj={final_obj:.2f}, max_corr={final_max_corr:.4f}")
    if swap_count > 0:
        print(f"  Delta: obj {final_obj - start_obj:+.2f}, "
              f"max_corr {final_max_corr - start_max_corr:+.4f}")

    return current_indices

