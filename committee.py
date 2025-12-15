"""
Committee Production Engine for Project Eigen 2 (Global50 Edition)
- Phase 1: Draft Day (Global50 Selection with Coefficient Correlation Optimization)
- Phase 2: Validation (3-Slice Holdout Testing)
- Cloud Sync: Committee roster is mirrored to cloud in global50/cw{N}/committee/

HOLDOUT CONFIGURATION:
The committee uses Config.COMMITTEE_HOLDOUT_DAYS to define a holdout period at the
END of your dataset. This data is reserved EXCLUSIVELY for committee validation and
must NEVER be used during agent training.

AGENT SOURCE:
Agents are sourced from Global50 (global50/cw{N}/global50.json) instead of Hall of Fame.
Selection optimizes for aggregate gauntlet score while minimizing coefficient correlation.

OBJECTIVE FUNCTION:
    objective = Σ(gauntlet_scores) * (1 - avg_correlation^EXPONENT)

This balances committee strength with diversity of decision-making.

CLOUD STORAGE:
Committee roster is stored in: global50/cw{N}/committee/
  - committee_roster.json: Full committee metadata
  - committee_correlation.png: Correlation heatmap visualization

USAGE:
  python committee.py --verify-only   # Check data split only
  python committee.py --draft         # Phase 1: Select committee from Global50
  python committee.py --validate      # Phase 2: Validate on holdout slices
  python committee.py --mirror        # Check cloud sync status, download if needed
"""

import os
import argparse
import json
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from itertools import combinations
import gc
import tempfile

# Project Imports
from utils.config import Config
from data.loader import StockDataLoader
from models.ddpg_agent import DDPGAgent
from utils.cloud_sync import get_cloud_sync_from_env

# --- Configuration ---
GLOBAL50_BASE_DIR = Path("global50")
COMMITTEE_DIR = Path("committee_results")  # Legacy local-only path
ROSTER_FILENAME = "committee_roster.json"
CORRELATION_FILENAME = "committee_correlation.png"


class CommitteeManager:
    """
    Manages committee selection, storage, and cloud synchronization.
    Committee files live in global50/cw{N}/committee/ alongside agents/.
    """

    def __init__(self, context_window_days: int):
        """
        Initialize CommitteeManager.

        Args:
            context_window_days: Context window for this league (e.g., 151)
        """
        self.context_window_days = context_window_days
        self.context_window_id = f"cw{context_window_days}"

        # Cloud sync
        self.cloud_sync = get_cloud_sync_from_env()

        # Local paths
        self.local_base = GLOBAL50_BASE_DIR / self.context_window_id
        self.local_committee_dir = self.local_base / "committee"
        self.local_roster_path = self.local_committee_dir / ROSTER_FILENAME
        self.local_correlation_path = self.local_committee_dir / CORRELATION_FILENAME

        # Cloud paths
        self.cloud_base = f"{self.cloud_sync.project_name}/global50/{self.context_window_id}"
        self.cloud_committee_base = f"{self.cloud_base}/committee"
        self.cloud_roster_path = f"{self.cloud_committee_base}/{ROSTER_FILENAME}"
        self.cloud_correlation_path = f"{self.cloud_committee_base}/{CORRELATION_FILENAME}"

        # Ensure local directories exist
        self.local_committee_dir.mkdir(parents=True, exist_ok=True)

    def save_roster(self, roster_data: dict, correlation_matrix: np.ndarray = None) -> bool:
        """
        Save committee roster locally and sync to cloud.

        Args:
            roster_data: Committee roster dictionary
            correlation_matrix: Optional correlation matrix for heatmap

        Returns:
            True if save and sync succeeded
        """
        # Save roster JSON locally
        with open(self.local_roster_path, 'w') as f:
            json.dump(roster_data, f, indent=2)
        print(f"✓ Roster saved locally: {self.local_roster_path}")

        # Save correlation heatmap if provided
        if correlation_matrix is not None:
            self._save_correlation_heatmap(roster_data, correlation_matrix)

        # Sync to cloud
        return self._sync_to_cloud()

    def _save_correlation_heatmap(self, roster_data: dict, correlation_matrix: np.ndarray):
        """Save correlation heatmap visualization."""
        plt.figure(figsize=(10, 8))
        labels = [f"{m['run_name']}_{m['agent_id']}"[:15] for m in roster_data['members']]

        avg_corr = roster_data.get('correlation', {}).get('average', 0)
        max_corr = roster_data.get('correlation', {}).get('max_pair', 0)

        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f",
                    vmin=-1, vmax=1, xticklabels=labels, yticklabels=labels)
        plt.title(f"Committee Coefficient Correlation\n"
                  f"(Avg: {avg_corr:.3f}, Max: {max_corr:.3f})")
        plt.tight_layout()
        plt.savefig(self.local_correlation_path, dpi=150)
        plt.close()
        print(f"✓ Correlation heatmap saved: {self.local_correlation_path}")

    def _sync_to_cloud(self) -> bool:
        """Sync local committee files to cloud storage."""
        if self.cloud_sync.provider == "local":
            print("  ⚠ Cloud sync disabled (local mode)")
            return True

        print(f"\nSyncing committee to cloud...")
        success = True

        # Upload roster JSON
        if self.local_roster_path.exists():
            if self.cloud_sync.upload_file_verified(
                str(self.local_roster_path), self.cloud_roster_path
            ):
                print(f"  ✓ Uploaded: {ROSTER_FILENAME}")
            else:
                print(f"  ✗ Failed to upload: {ROSTER_FILENAME}")
                success = False

        # Upload correlation heatmap
        if self.local_correlation_path.exists():
            if self.cloud_sync.upload_file_verified(
                str(self.local_correlation_path), self.cloud_correlation_path
            ):
                print(f"  ✓ Uploaded: {CORRELATION_FILENAME}")
            else:
                print(f"  ✗ Failed to upload: {CORRELATION_FILENAME}")
                success = False

        if success:
            print(f"  ✓ Cloud mirror: gs://{self.cloud_sync.bucket_name}/{self.cloud_committee_base}/")

        return success

    def load_roster(self) -> dict:
        """Load committee roster from local storage."""
        if not self.local_roster_path.exists():
            return None

        with open(self.local_roster_path, 'r') as f:
            return json.load(f)

    def check_mirror_status(self) -> bool:
        """
        Check synchronization status between local and cloud.
        Downloads missing files from cloud if needed.

        Returns:
            True if in sync, False if mismatch or error
        """
        print("\n" + "="*60)
        print(f"COMMITTEE MIRROR CHECK ({self.context_window_id})")
        print("="*60)

        if self.cloud_sync.provider == "local":
            print("  ⚠ Cloud sync disabled (local mode)")
            print("  Checking local files only...")
            return self._check_local_status()

        print(f"  Cloud base: gs://{self.cloud_sync.bucket_name}/{self.cloud_committee_base}/")
        print(f"  Local base: {self.local_committee_dir}/")

        # Check cloud roster exists
        cloud_roster_exists = self._cloud_file_exists(self.cloud_roster_path)
        local_roster_exists = self.local_roster_path.exists()

        print(f"\n  Roster JSON:")
        print(f"    Cloud: {'EXISTS' if cloud_roster_exists else 'MISSING'}")
        print(f"    Local: {'EXISTS' if local_roster_exists else 'MISSING'}")

        # Download from cloud if local missing
        if cloud_roster_exists and not local_roster_exists:
            print(f"  → Downloading roster from cloud...")
            if self.cloud_sync.download_file(self.cloud_roster_path, str(self.local_roster_path)):
                print(f"    ✓ Downloaded {ROSTER_FILENAME}")
                local_roster_exists = True
            else:
                print(f"    ✗ Failed to download {ROSTER_FILENAME}")

        # Check correlation heatmap
        cloud_corr_exists = self._cloud_file_exists(self.cloud_correlation_path)
        local_corr_exists = self.local_correlation_path.exists()

        print(f"\n  Correlation Heatmap:")
        print(f"    Cloud: {'EXISTS' if cloud_corr_exists else 'MISSING'}")
        print(f"    Local: {'EXISTS' if local_corr_exists else 'MISSING'}")

        if cloud_corr_exists and not local_corr_exists:
            print(f"  → Downloading correlation heatmap from cloud...")
            if self.cloud_sync.download_file(self.cloud_correlation_path, str(self.local_correlation_path)):
                print(f"    ✓ Downloaded {CORRELATION_FILENAME}")
                local_corr_exists = True
            else:
                print(f"    ✗ Failed to download {CORRELATION_FILENAME}")

        # Compare local vs cloud if both exist
        if cloud_roster_exists and local_roster_exists:
            match = self._compare_rosters()
            if match:
                print(f"\n  ✓ Local and cloud rosters MATCH")
            else:
                print(f"\n  ⚠ Local and cloud rosters DIFFER")
                print(f"    Use --draft to regenerate committee")
                return False

        # Summary
        if local_roster_exists:
            roster = self.load_roster()
            print(f"\n  Committee Status:")
            print(f"    Size: {roster.get('committee_size', 'N/A')}")
            print(f"    Objective: {roster.get('objective_value', 'N/A'):.2f}")
            print(f"    Avg Correlation: {roster.get('correlation', {}).get('average', 'N/A'):.3f}")
            print(f"    Generated: {roster.get('generated_at', 'N/A')}")
            return True
        else:
            print(f"\n  ⚠ No committee roster found")
            print(f"    Run --draft to create committee")
            return False

    def _check_local_status(self) -> bool:
        """Check local committee status when cloud is disabled."""
        if self.local_roster_path.exists():
            roster = self.load_roster()
            print(f"\n  Committee Status:")
            print(f"    Size: {roster.get('committee_size', 'N/A')}")
            print(f"    Objective: {roster.get('objective_value', 'N/A'):.2f}")
            print(f"    Avg Correlation: {roster.get('correlation', {}).get('average', 'N/A'):.3f}")
            print(f"    Generated: {roster.get('generated_at', 'N/A')}")
            return True
        else:
            print(f"\n  ⚠ No committee roster found locally")
            print(f"    Run --draft to create committee")
            return False

    def _cloud_file_exists(self, cloud_path: str) -> bool:
        """Check if a file exists in cloud storage."""
        try:
            return self.cloud_sync.file_exists(cloud_path)
        except Exception:
            return False

    def _compare_rosters(self) -> bool:
        """Compare local and cloud rosters for equality."""
        try:
            # Download cloud roster to temp file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
                temp_path = tmp.name

            if not self.cloud_sync.download_file(self.cloud_roster_path, temp_path, silent=True):
                return False

            with open(temp_path, 'r') as f:
                cloud_roster = json.load(f)

            local_roster = self.load_roster()

            # Compare key fields
            if local_roster.get('committee_size') != cloud_roster.get('committee_size'):
                return False

            local_members = set(m['filename'] for m in local_roster.get('members', []))
            cloud_members = set(m['filename'] for m in cloud_roster.get('members', []))

            return local_members == cloud_members

        except Exception as e:
            print(f"  ⚠ Error comparing rosters: {e}")
            return False
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


# --- Helper Functions ---

def calculate_max_drawdown(cumulative_returns):
    """Calculates Maximum Drawdown from a cumulative return series."""
    if len(cumulative_returns) == 0:
        return 0.0
    peak = np.maximum.accumulate(cumulative_returns)
    peak = np.where(peak == 0, 1e-9, peak)
    drawdown = (cumulative_returns - peak) / peak
    return float(np.min(drawdown))


def verify_data_split(loader):
    """
    Verifies THREE-TIER split: Training → Validation → Holdout

    Returns: (is_valid, error_message, holdout_info_dict)
    """
    total_days = len(loader.data_array_full)

    min_required = (Config.CONTEXT_WINDOW_DAYS + Config.VALIDATION_DAYS +
                    Config.COMMITTEE_HOLDOUT_DAYS + Config.MIN_HOLDING_PERIOD)
    if total_days < min_required:
        return False, f"Dataset too small: {total_days} days (need {min_required})", None

    expected_holdout_start = total_days - Config.COMMITTEE_HOLDOUT_DAYS
    holdout_end = total_days - 1

    if not hasattr(loader, 'train_end_idx') or loader.train_end_idx is None:
        return False, "Loader missing train_end_idx! Must call create_train_val_split() first.", None

    train_end = loader.train_end_idx
    train_start = 0
    train_size = train_end - train_start + 1

    if not hasattr(loader, 'val_end_idx') or loader.val_end_idx is None:
        return False, "Loader missing val_end_idx! Must call create_train_val_split() first.", None

    val_start = loader.val_start_idx
    val_end = loader.val_end_idx
    val_size = val_end - val_start + 1

    if val_end >= expected_holdout_start:
        return False, f"VALIDATION OVERLAPS WITH HOLDOUT! val_end={val_end}, holdout_start={expected_holdout_start}", None

    holdout_info = {
        'holdout_start': expected_holdout_start,
        'holdout_end': holdout_end,
        'train_start': train_start,
        'train_end': train_end,
        'val_start': val_start,
        'val_end': val_end,
    }

    print("✅ THREE-TIER DATA SPLIT VERIFICATION PASSED")
    print(f"\n  Total Days:     {total_days:,}")
    print(f"  Training:       {train_size:,} days (indices {train_start:,} to {train_end:,})")
    print(f"  Validation:     {val_size:,} days (indices {val_start:,} to {val_end:,})")
    print(f"  Holdout:        {Config.COMMITTEE_HOLDOUT_DAYS:,} days (indices {expected_holdout_start:,} to {holdout_end:,})")

    return True, None, holdout_info


def load_normalization_stats():
    """Calculates normalization stats deterministically."""
    print("Loading data to calculate normalization stats...")
    loader = StockDataLoader()
    _, stats = loader.load_and_prepare()
    return loader, stats


def load_agent_actor_only(filepath, agent_id):
    """Loads only the Actor network to save VRAM."""
    try:
        agent = DDPGAgent(agent_id=agent_id)
        checkpoint = torch.load(filepath, map_location=Config.DEVICE, weights_only=False)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.actor.eval()
        del agent.critic
        del agent.critic_target
        del agent.actor_target
        return agent
    except Exception as e:
        print(f"⚠ Error loading {filepath}: {e}")
        return None


def get_holdout_data(loader, stats, holdout_info):
    """
    Get holdout data tensor for inference.

    Returns:
        (inputs_tensor, valid_indices)
    """
    holdout_start = holdout_info['holdout_start']
    holdout_end = holdout_info['holdout_end']

    input_data = loader.data_array  # [Days, Stocks, 5_feats]

    # Valid inference points (need room for MIN_HOLDING_PERIOD forward)
    valid_indices = list(range(holdout_start, holdout_end - Config.MIN_HOLDING_PERIOD + 1))

    inputs = []
    for i in valid_indices:
        window = input_data[i - Config.CONTEXT_WINDOW_DAYS : i]
        normalized = (window - stats['mean']) / stats['std']
        inputs.append(normalized)

    inputs = np.array(inputs)
    inputs_tensor = torch.FloatTensor(inputs).to(Config.DEVICE)

    return inputs_tensor, valid_indices


# --- Global50 Loading ---

def load_global50_candidates(context_window_days: int) -> list:
    """
    Load agents from Global50, sorted by gauntlet_score descending.

    Args:
        context_window_days: Context window to load (e.g., 151)

    Returns:
        List of entry dicts with agent metadata
    """
    json_path = GLOBAL50_BASE_DIR / f"cw{context_window_days}" / "global50.json"

    if not json_path.exists():
        print(f"❌ Global50 ledger not found: {json_path}")
        return []

    with open(json_path, 'r') as f:
        data = json.load(f)

    entries = data.get('entries', [])

    # Sort by gauntlet_score descending
    entries.sort(key=lambda e: e.get('gauntlet_score', 0), reverse=True)

    print(f"✓ Loaded {len(entries)} agents from Global50 (cw{context_window_days})")

    return entries


def get_agent_filepath(entry: dict, context_window_days: int) -> Path:
    """Get the filepath for an agent's weights."""
    filename = f"{entry['run_name']}_{entry['agent_id']}.pth"
    return GLOBAL50_BASE_DIR / f"cw{context_window_days}" / "agents" / filename


# --- Correlation Calculation ---

def calculate_coefficient_correlations(entries: list, holdout_tensor: torch.Tensor,
                                        context_window_days: int) -> tuple:
    """
    Calculate pairwise coefficient correlations between agents.

    Args:
        entries: List of Global50 entry dicts
        holdout_tensor: Prepared holdout data tensor [Days, Context, Stocks, Features]
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
            # Get coefficient predictions [Days, Stocks, 2]
            actions = agent.actor(holdout_tensor).cpu().numpy()

        # Extract coefficients (first output dimension)
        # Flatten to 1D for correlation: [Days * Stocks]
        coefs = actions[:, :, 0].flatten()
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

    Objective = Σ(gauntlet_scores) * (1 - avg_correlation^EXPONENT)

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
    max_corr = 0.0

    for i_idx in range(n):
        for j_idx in range(i_idx + 1, n):
            i = indices[i_idx]
            j = indices[j_idx]
            corr = corr_matrix[i, j]
            pairs.append(corr)
            if corr > max_corr:
                max_corr = corr

    avg_corr = np.mean(pairs) if pairs else 0.0

    # Objective with correlation penalty
    penalty = 1.0 - (avg_corr ** Config.COMMITTEE_CORRELATION_EXPONENT)
    objective = score_sum * penalty

    return objective, score_sum, avg_corr, max_corr


# --- Optimization ---

def optimize_committee(entries: list, corr_matrix: np.ndarray) -> dict:
    """
    Find optimal committee using incremental exhaustive search.

    Algorithm:
    1. Start with top K agents (COMMITTEE_TOP_K_INITIAL)
    2. Exhaustive search for best COMMITTEE_SIZE combination
    3. Expand pool by 1 agent, re-optimize
    4. Stop after COMMITTEE_EARLY_STOP consecutive non-improvements

    Returns:
        Dict with best committee info
    """
    committee_size = Config.COMMITTEE_SIZE
    initial_pool = Config.COMMITTEE_TOP_K_INITIAL
    early_stop = Config.COMMITTEE_EARLY_STOP

    # Filter to entries with valid correlation data
    valid_entries = [i for i in range(len(entries))
                     if corr_matrix[i, i] == 1.0]  # Valid entries have self-correlation = 1

    if len(valid_entries) < committee_size:
        print(f"❌ Not enough valid agents ({len(valid_entries)}) for committee of {committee_size}")
        return None

    print(f"\n{'='*60}")
    print(f"COMMITTEE OPTIMIZATION")
    print(f"{'='*60}")
    print(f"  Committee Size: {committee_size}")
    print(f"  Initial Pool: top {initial_pool} agents")
    print(f"  Early Stop: {early_stop} consecutive non-improvements")
    print(f"  Valid Agents: {len(valid_entries)}")

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
        pool = valid_entries[:current_pool_size]

        # Count combinations
        from math import comb
        num_combos = comb(len(pool), committee_size)

        print(f"\n  Iteration {iteration}: Pool size {current_pool_size}, "
              f"searching {num_combos:,} combinations...")

        improved = False

        for combo in combinations(pool, committee_size):
            obj, score_sum, avg_corr, max_corr = committee_objective(
                combo, entries, corr_matrix
            )

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

        # Expand pool
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


# --- Validation ---

def evaluate_agent_on_slice(agent_path: Path, slice_tensor: torch.Tensor,
                            slice_returns: np.ndarray) -> dict:
    """
    Evaluate a single agent on a holdout slice.

    Returns dict with comprehensive fitness metrics.
    """
    agent = load_agent_actor_only(agent_path, 0)
    if agent is None:
        return {'error': 'Failed to load agent'}

    with torch.no_grad():
        actions = agent.actor(slice_tensor).cpu().numpy()

    coeffs = actions[:, :, 0]  # [Days, Stocks]

    # Active positions (coefficient > threshold)
    active = np.maximum(0, coeffs - Config.COEFFICIENT_THRESHOLD)
    active = np.minimum(active, 2.0)  # Cap leverage

    # Calculate trades and PnL
    trades_mask = active > 0
    num_trades = np.sum(trades_mask)

    if num_trades == 0:
        del agent
        return {
            'num_trades': 0,
            'win_rate': 0.0,
            'quality_ratio': 0.0,
            'expectancy': 0.0,
            'roi': 0.0,
            'fitness': -Config.ZERO_TRADES_PENALTY_GAUNTLET,
        }

    # Get returns for active positions (only investable stocks)
    investable_returns = slice_returns[:, Config.INVESTABLE_START_COL:Config.INVESTABLE_END_COL+1]
    trade_returns = investable_returns[trades_mask]
    trade_weights = active[trades_mask]

    # Weighted PnL
    weighted_returns = trade_returns * trade_weights
    total_pnl = np.sum(weighted_returns)
    total_investment = np.sum(trade_weights)

    roi = (total_pnl / total_investment * 100) if total_investment > 0 else 0.0
    win_rate = np.mean(trade_returns > 0) if len(trade_returns) > 0 else 0.0

    # Quality ratio (winners / losers)
    winners = np.sum(trade_returns > 0)
    losers = np.sum(trade_returns <= 0)
    quality_ratio = (winners / losers) if losers > 0 else float('inf')

    # Expectancy (average return per trade)
    expectancy = float(np.mean(weighted_returns)) if len(weighted_returns) > 0 else 0.0

    # Simple fitness approximation
    fitness = roi * np.log10(abs(total_pnl) + 10)
    if roi > 0:
        fitness *= (1 + win_rate ** 2)

    del agent
    torch.cuda.empty_cache()

    return {
        'num_trades': int(num_trades),
        'win_rate': float(win_rate),
        'quality_ratio': float(quality_ratio),
        'expectancy': expectancy,
        'roi': float(roi),
        'fitness': float(fitness),
    }


def evaluate_committee_on_slice(members: list, slice_tensor: torch.Tensor,
                                 slice_returns: np.ndarray,
                                 context_window_days: int,
                                 slice_start_idx: int = None,
                                 export_trades: bool = False) -> dict:
    """
    Evaluate committee consensus on a holdout slice.

    Uses voting mechanism: trade triggers when majority of committee agrees.
    Returns detailed consensus statistics.
    """
    all_coeffs = []

    for entry in members:
        filepath = get_agent_filepath(entry, context_window_days)
        agent = load_agent_actor_only(filepath, entry['agent_id'])
        if agent is None:
            continue

        with torch.no_grad():
            actions = agent.actor(slice_tensor).cpu().numpy()

        coeffs = actions[:, :, 0]
        all_coeffs.append(coeffs)

        del agent
        torch.cuda.empty_cache()

    if not all_coeffs:
        return {'error': 'No agents loaded'}

    # Stack coefficients: [Agents, Days, Stocks]
    all_coeffs = np.array(all_coeffs)
    num_members = len(all_coeffs)

    # Voting: each agent votes if coefficient > threshold
    votes = all_coeffs >= Config.COEFFICIENT_THRESHOLD  # [Agents, Days, Stocks]
    vote_counts = np.sum(votes, axis=0)  # [Days, Stocks]

    # Quorum: configurable number of members must agree
    quorum = Config.COMMITTEE_QUORUM
    triggers = vote_counts >= quorum  # [Days, Stocks]

    num_trades = np.sum(triggers)

    # Extract returns for investable stocks only
    investable_returns = slice_returns[:, Config.INVESTABLE_START_COL:Config.INVESTABLE_END_COL+1]

    # Consensus statistics
    consensus_stats = {
        'unanimity_count': 0,
        'unanimity_pct': 0.0,
        'unanimity_win_rate': 0.0,
        'unanimity_quality_ratio': 0.0,
        'min_consensus_count': 0,
        'min_consensus_pct': 0.0,
        'min_consensus_win_rate': 0.0,
        'min_consensus_quality_ratio': 0.0,
        'avg_consensus_votes': 0.0,
        'consensus_distribution': {},  # votes -> count
    }

    if num_trades == 0:
        return {
            'num_trades': 0,
            'win_rate': 0.0,
            'roi': 0.0,
            'fitness': -Config.ZERO_TRADES_PENALTY_GAUNTLET,
            'consensus_stats': consensus_stats,
        }

    # Analyze consensus levels for triggered trades
    triggered_vote_counts = vote_counts[triggers]

    # Unanimity trades (all members agree)
    unanimity_mask = triggered_vote_counts == num_members
    unanimity_count = np.sum(unanimity_mask)

    # Minimum consensus trades (exactly quorum votes)
    min_consensus_mask = triggered_vote_counts == quorum
    min_consensus_count = np.sum(min_consensus_mask)

    # Average number of votes per trade
    avg_votes = float(np.mean(triggered_vote_counts))

    # Consensus distribution
    unique_votes, vote_freq = np.unique(triggered_vote_counts, return_counts=True)
    consensus_distribution = {int(v): int(f) for v, f in zip(unique_votes, vote_freq)}

    # Calculate metrics for unanimity trades
    if unanimity_count > 0:
        unanimity_returns = investable_returns[triggers][unanimity_mask]
        unanimity_win_rate = float(np.mean(unanimity_returns > 0))
        unanimity_winners = np.sum(unanimity_returns > 0)
        unanimity_losers = np.sum(unanimity_returns <= 0)
        unanimity_quality_ratio = (unanimity_winners / unanimity_losers) if unanimity_losers > 0 else float('inf')
    else:
        unanimity_win_rate = 0.0
        unanimity_quality_ratio = 0.0

    # Calculate metrics for minimum consensus trades
    if min_consensus_count > 0:
        min_consensus_returns = investable_returns[triggers][min_consensus_mask]
        min_consensus_win_rate = float(np.mean(min_consensus_returns > 0))
        min_consensus_winners = np.sum(min_consensus_returns > 0)
        min_consensus_losers = np.sum(min_consensus_returns <= 0)
        min_consensus_quality_ratio = (min_consensus_winners / min_consensus_losers) if min_consensus_losers > 0 else float('inf')
    else:
        min_consensus_win_rate = 0.0
        min_consensus_quality_ratio = 0.0

    consensus_stats = {
        'unanimity_count': int(unanimity_count),
        'unanimity_pct': float(unanimity_count / num_trades * 100),
        'unanimity_win_rate': unanimity_win_rate,
        'unanimity_quality_ratio': unanimity_quality_ratio,
        'min_consensus_count': int(min_consensus_count),
        'min_consensus_pct': float(min_consensus_count / num_trades * 100),
        'min_consensus_win_rate': min_consensus_win_rate,
        'min_consensus_quality_ratio': min_consensus_quality_ratio,
        'avg_consensus_votes': avg_votes,
        'consensus_distribution': consensus_distribution,
    }

    # Average coefficient for triggered trades (consensus strength)
    # Use mean of ALL agents (including those who voted 0) to properly aggregate signals
    avg_coef = np.mean(all_coeffs, axis=0)  # [Days, Stocks]

    # For triggered trades, use the averaged coefficient directly (no re-thresholding)
    # The quorum already decided IF we trade; now we use the average to decide HOW MUCH
    active = np.where(triggers, avg_coef, 0)
    active = np.minimum(active, 2.0)  # Cap at 2x leverage

    trade_returns = investable_returns[triggers]
    trade_weights = active[triggers]

    weighted_returns = trade_returns * trade_weights
    total_pnl = np.sum(weighted_returns)
    total_investment = np.sum(trade_weights)

    roi = (total_pnl / total_investment * 100) if total_investment > 0 else 0.0
    win_rate = np.mean(trade_returns > 0) if len(trade_returns) > 0 else 0.0

    # Quality ratio (winners / losers)
    winners = np.sum(trade_returns > 0)
    losers = np.sum(trade_returns <= 0)
    quality_ratio = (winners / losers) if losers > 0 else float('inf')

    # Expectancy (average return per trade)
    expectancy = float(np.mean(weighted_returns)) if len(weighted_returns) > 0 else 0.0

    fitness = roi * np.log10(abs(total_pnl) + 10)
    if roi > 0:
        fitness *= (1 + win_rate ** 2)

    # Export trade details if requested
    trade_details = None
    if export_trades:
        # Get indices of triggered trades
        trigger_indices = np.argwhere(triggers)  # [(day_idx, stock_idx), ...]

        trade_details = []
        for idx, (day_idx, stock_idx) in enumerate(trigger_indices):
            # Adjust stock index to investable range
            actual_stock_idx = stock_idx + Config.INVESTABLE_START_COL

            trade_details.append({
                'day_idx': int(slice_start_idx + day_idx) if slice_start_idx is not None else int(day_idx),
                'stock_idx': int(actual_stock_idx),
                'coefficient': float(avg_coef[day_idx, stock_idx]),
                'position_size': float(active[day_idx, stock_idx]),
                'return': float(trade_returns[idx]),
                'weighted_return': float(weighted_returns[idx]),
                'vote_count': int(triggered_vote_counts[idx]),
                'is_winner': bool(trade_returns[idx] > 0),
            })

    return {
        'num_trades': int(num_trades),
        'win_rate': float(win_rate),
        'quality_ratio': quality_ratio,
        'expectancy': expectancy,
        'roi': float(roi),
        'fitness': float(fitness),
        'consensus_stats': consensus_stats,
        'trade_details': trade_details,
    }


def run_validation(manager: CommitteeManager, loader, stats, holdout_info,
                   context_window_days: int) -> dict:
    """
    Run 3-slice holdout validation on the committee.

    Returns validation results dict with comprehensive metrics.
    """
    print("\n" + "="*60)
    print("PHASE 2: VALIDATION (3-Slice Holdout Test)")
    print("="*60)

    roster = manager.load_roster()
    if roster is None:
        print(f"❌ No roster found. Run --draft first.")
        return None

    members = roster['members']
    num_slices = Config.COMMITTEE_VALIDATION_SLICES

    # Get full holdout data
    holdout_tensor, valid_indices = get_holdout_data(loader, stats, holdout_info)

    # Calculate returns
    close_idx = 1
    full_closes = loader.data_array_full[:, :, close_idx]

    holdout_returns = []
    for t in valid_indices:
        entry_price = full_closes[t]
        exit_price = full_closes[t + Config.MIN_HOLDING_PERIOD]
        ret = np.where(entry_price > 0, (exit_price - entry_price) / entry_price, 0.0)
        holdout_returns.append(ret)
    holdout_returns = np.array(holdout_returns)

    # Split into slices
    slice_size = len(valid_indices) // num_slices

    print(f"\n  Holdout days: {len(valid_indices)}")
    print(f"  Slices: {num_slices}")
    print(f"  Days per slice: {slice_size}")

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
            'avg_min_consensus_pct': 0.0,
            'avg_consensus_votes': 0.0,
        }
    }

    # Validate individual agents on each slice
    print(f"\nValidating {len(members)} individual agents...")

    for member in tqdm(members, desc="Agents"):
        entry = member  # member is already the entry dict
        filepath = get_agent_filepath(entry, context_window_days)

        agent_results = {
            'agent_id': entry['agent_id'],
            'run_name': entry['run_name'],
            'stored_gauntlet': entry['gauntlet_score'],
            'slice_metrics': [],
        }

        for s in range(num_slices):
            start_idx = s * slice_size
            end_idx = (s + 1) * slice_size if s < num_slices - 1 else len(valid_indices)

            slice_tensor = holdout_tensor[start_idx:end_idx]
            slice_returns = holdout_returns[start_idx:end_idx]

            metrics = evaluate_agent_on_slice(filepath, slice_tensor, slice_returns)
            agent_results['slice_metrics'].append({
                'slice': s,
                'fitness': metrics.get('fitness', 0.0),
                'win_rate': metrics.get('win_rate', 0.0),
                'quality_ratio': metrics.get('quality_ratio', 0.0),
                'expectancy': metrics.get('expectancy', 0.0),
                'roi': metrics.get('roi', 0.0),
                'num_trades': metrics.get('num_trades', 0),
            })

        # Calculate aggregates for this agent
        agent_results['fresh_mean_fitness'] = float(np.mean([m['fitness'] for m in agent_results['slice_metrics']]))
        agent_results['fresh_mean_win_rate'] = float(np.mean([m['win_rate'] for m in agent_results['slice_metrics']]))
        agent_results['fresh_mean_quality_ratio'] = float(np.mean([m['quality_ratio'] for m in agent_results['slice_metrics']]))
        agent_results['fresh_mean_expectancy'] = float(np.mean([m['expectancy'] for m in agent_results['slice_metrics']]))
        agent_results['fresh_mean_roi'] = float(np.mean([m['roi'] for m in agent_results['slice_metrics']]))

        results['individual_agents'].append(agent_results)

    # Validate committee consensus on each slice
    print(f"\nValidating committee consensus...")

    all_trades_data = []  # Collect all trades across slices

    for s in range(num_slices):
        start_idx = s * slice_size
        end_idx = (s + 1) * slice_size if s < num_slices - 1 else len(valid_indices)

        slice_tensor = holdout_tensor[start_idx:end_idx]
        slice_returns = holdout_returns[start_idx:end_idx]

        # Get absolute day index for trade export
        slice_start_day_idx = valid_indices[start_idx]

        metrics = evaluate_committee_on_slice(
            members, slice_tensor, slice_returns, context_window_days,
            slice_start_idx=slice_start_day_idx,
            export_trades=True
        )

        slice_result = {
            'slice': s,
            'fitness': metrics.get('fitness', 0.0),
            'win_rate': metrics.get('win_rate', 0.0),
            'quality_ratio': metrics.get('quality_ratio', 0.0),
            'expectancy': metrics.get('expectancy', 0.0),
            'roi': metrics.get('roi', 0.0),
            'num_trades': metrics.get('num_trades', 0),
            'consensus_stats': metrics.get('consensus_stats', {}),
        }

        results['committee_slices'].append(slice_result)

        # Collect trade details for CSV export
        if metrics.get('trade_details'):
            for trade in metrics['trade_details']:
                trade['slice'] = s
                all_trades_data.append(trade)

    # Calculate committee aggregates
    results['committee_aggregate']['mean_fitness'] = float(
        np.mean([s['fitness'] for s in results['committee_slices']])
    )
    results['committee_aggregate']['mean_win_rate'] = float(
        np.mean([s['win_rate'] for s in results['committee_slices']])
    )
    results['committee_aggregate']['mean_quality_ratio'] = float(
        np.mean([s['quality_ratio'] for s in results['committee_slices']])
    )
    results['committee_aggregate']['mean_expectancy'] = float(
        np.mean([s['expectancy'] for s in results['committee_slices']])
    )
    results['committee_aggregate']['mean_roi'] = float(
        np.mean([s['roi'] for s in results['committee_slices']])
    )
    results['committee_aggregate']['total_trades'] = int(
        sum([s['num_trades'] for s in results['committee_slices']])
    )

    # Calculate consensus summary
    results['consensus_summary']['avg_unanimity_pct'] = float(
        np.mean([s['consensus_stats']['unanimity_pct'] for s in results['committee_slices']])
    )
    results['consensus_summary']['avg_min_consensus_pct'] = float(
        np.mean([s['consensus_stats']['min_consensus_pct'] for s in results['committee_slices']])
    )
    results['consensus_summary']['avg_consensus_votes'] = float(
        np.mean([s['consensus_stats']['avg_consensus_votes'] for s in results['committee_slices']])
    )

    # Print comprehensive results
    print(f"\n{'='*60}")
    print("VALIDATION RESULTS")
    print(f"{'='*60}")

    print("\n" + "-"*60)
    print("INDIVIDUAL AGENTS (Slice Performance)")
    print("-"*60)
    for agent in results['individual_agents']:
        print(f"\n  {agent['run_name']}_{agent['agent_id']}:")
        print(f"    Stored Gauntlet: {agent['stored_gauntlet']:.2f}")
        print(f"    Fresh Averages: fitness={agent['fresh_mean_fitness']:.2f}, "
              f"win_rate={agent['fresh_mean_win_rate']:.2%}, "
              f"quality_ratio={agent['fresh_mean_quality_ratio']:.3f}, "
              f"expectancy={agent['fresh_mean_expectancy']:.6f}, "
              f"roi={agent['fresh_mean_roi']:.2f}%")
        for m in agent['slice_metrics']:
            print(f"      Slice {m['slice']}: fitness={m['fitness']:.2f}, "
                  f"win_rate={m['win_rate']:.2%}, quality_ratio={m['quality_ratio']:.3f}, "
                  f"expectancy={m['expectancy']:.6f}, roi={m['roi']:.2f}%, "
                  f"trades={m['num_trades']}")

    print("\n" + "-"*60)
    print("COMMITTEE PERFORMANCE (Per Slice)")
    print("-"*60)
    for s in results['committee_slices']:
        print(f"\n  Slice {s['slice']}:")
        print(f"    Fitness: {s['fitness']:.2f}")
        print(f"    Win Rate: {s['win_rate']:.2%}")
        print(f"    Quality Ratio: {s['quality_ratio']:.3f}")
        print(f"    Expectancy: {s['expectancy']:.6f}")
        print(f"    ROI: {s['roi']:.2f}%")
        print(f"    Trades: {s['num_trades']}")
        print(f"")

        cs = s['consensus_stats']
        print(f"    Consensus Breakdown:")
        print(f"      Unanimity: {cs['unanimity_count']} trades ({cs['unanimity_pct']:.1f}%)")
        if cs['unanimity_count'] > 0:
            print(f"        Win Rate: {cs['unanimity_win_rate']:.2%}")
            print(f"        Quality Ratio: {cs['unanimity_quality_ratio']:.3f}")

        print(f"      Min Consensus: {cs['min_consensus_count']} trades ({cs['min_consensus_pct']:.1f}%)")
        if cs['min_consensus_count'] > 0:
            print(f"        Win Rate: {cs['min_consensus_win_rate']:.2%}")
            print(f"        Quality Ratio: {cs['min_consensus_quality_ratio']:.3f}")

        print(f"      Avg Votes per Trade: {cs['avg_consensus_votes']:.2f}")
        print(f"      Vote Distribution: {cs['consensus_distribution']}")

    print("\n" + "-"*60)
    print("COMMITTEE AGGREGATE PERFORMANCE")
    print("-"*60)
    ca = results['committee_aggregate']
    print(f"  Mean Fitness: {ca['mean_fitness']:.2f}")
    print(f"  Mean Win Rate: {ca['mean_win_rate']:.2%}")
    print(f"  Mean Quality Ratio: {ca['mean_quality_ratio']:.3f}")
    print(f"  Mean Expectancy: {ca['mean_expectancy']:.6f}")
    print(f"  Mean ROI: {ca['mean_roi']:.2f}%")
    print(f"  Total Trades (All Slices): {ca['total_trades']}")

    print("\n" + "-"*60)
    print("CONSENSUS SUMMARY (Across All Slices)")
    print("-"*60)
    cs_sum = results['consensus_summary']
    print(f"  Avg Unanimity %: {cs_sum['avg_unanimity_pct']:.1f}%")
    print(f"  Avg Min Consensus %: {cs_sum['avg_min_consensus_pct']:.1f}%")
    print(f"  Avg Votes per Trade: {cs_sum['avg_consensus_votes']:.2f}")

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

    # Export trades to CSV files
    if all_trades_data:
        print(f"\n" + "-"*60)
        print("EXPORTING TRADE DATA")
        print("-"*60)

        # Export all trades to a single CSV
        trades_df = pd.DataFrame(all_trades_data)
        all_trades_csv = manager.local_committee_dir / "committee_all_trades.csv"
        trades_df.to_csv(all_trades_csv, index=False)
        print(f"  ✓ All trades: {all_trades_csv}")
        print(f"    Total trades: {len(trades_df)}")

        # Export per-slice CSVs
        for s in range(num_slices):
            slice_trades = [t for t in all_trades_data if t['slice'] == s]
            if slice_trades:
                slice_df = pd.DataFrame(slice_trades)
                slice_csv = manager.local_committee_dir / f"committee_slice_{s}_trades.csv"
                slice_df.to_csv(slice_csv, index=False)
                print(f"  ✓ Slice {s}: {slice_csv} ({len(slice_df)} trades)")

        # Upload CSVs to cloud
        if manager.cloud_sync.provider != "local":
            print(f"\n  Syncing trade CSVs to cloud...")

            # Upload all trades CSV
            cloud_all_trades = f"{manager.cloud_committee_base}/committee_all_trades.csv"
            if manager.cloud_sync.upload_file_verified(str(all_trades_csv), cloud_all_trades):
                print(f"    ✓ Uploaded: committee_all_trades.csv")
            else:
                print(f"    ✗ Failed to upload: committee_all_trades.csv")

            # Upload slice CSVs
            for s in range(num_slices):
                slice_csv = manager.local_committee_dir / f"committee_slice_{s}_trades.csv"
                if slice_csv.exists():
                    cloud_slice_csv = f"{manager.cloud_committee_base}/committee_slice_{s}_trades.csv"
                    if manager.cloud_sync.upload_file_verified(str(slice_csv), cloud_slice_csv):
                        print(f"    ✓ Uploaded: committee_slice_{s}_trades.csv")
                    else:
                        print(f"    ✗ Failed to upload: committee_slice_{s}_trades.csv")

    return results


# --- Phase 1: Draft Day ---

def run_draft(manager: CommitteeManager, loader, stats, holdout_info):
    """
    Phase 1: Select committee from Global50 using coefficient correlation optimization.
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
        print(f"    {i+1}. {e['run_name']}_{e['agent_id']}: "
              f"score={e['gauntlet_score']:.2f}, roi={e.get('roi', 0):.2f}%")

    # 2. Prepare holdout data
    holdout_tensor, valid_indices = get_holdout_data(loader, stats, holdout_info)
    print(f"\n  Holdout tensor shape: {holdout_tensor.shape}")

    # 3. Calculate coefficient correlations
    corr_matrix, coefficients, _ = calculate_coefficient_correlations(
        entries, holdout_tensor, context_window_days
    )

    # 4. Optimize committee selection
    result = optimize_committee(entries, corr_matrix)

    if result is None:
        print("❌ Optimization failed")
        return None

    # 5. Build roster
    committee_indices = result['committee_indices']
    committee_members = [entries[i] for i in committee_indices]

    # Build correlation matrix for committee
    n = len(committee_indices)
    committee_corr = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            committee_corr[i, j] = corr_matrix[committee_indices[i], committee_indices[j]]

    roster_data = {
        'committee_size': len(committee_members),
        'members': [
            {
                'filename': f"{e['run_name']}_{e['agent_id']}.pth",
                'agent_id': e['agent_id'],
                'run_name': e['run_name'],
                'gauntlet_score': e['gauntlet_score'],
                'roi': e.get('roi', 0.0),
                'expectancy': e.get('expectancy', 0.0),
                'quality_ratio': e.get('quality_ratio', 0.0),
                'win_ratio': e.get('win_ratio', 0.0),
            }
            for e in committee_members
        ],
        'aggregate_score': result['score_sum'],
        'objective_value': result['objective'],
        'correlation': {
            'average': result['avg_correlation'],
            'max_pair': result['max_correlation'],
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

    # Save roster and sync to cloud
    manager.save_roster(roster_data, committee_corr)

    # Print committee
    print(f"\n{'='*60}")
    print("SELECTED COMMITTEE")
    print(f"{'='*60}")

    for i, m in enumerate(roster_data['members']):
        print(f"  {i+1}. {m['run_name']}_{m['agent_id']}: "
              f"score={m['gauntlet_score']:.2f}, roi={m['roi']:.2f}%")

    print(f"\n  Aggregate Score: {result['score_sum']:.2f}")
    print(f"  Objective Value: {result['objective']:.2f}")
    print(f"  Avg Correlation: {result['avg_correlation']:.3f}")
    print(f"  Max Pair Correlation: {result['max_correlation']:.3f}")

    return roster_data


# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Committee Selection from Global50")
    parser.add_argument('--draft', action='store_true',
                        help='Run Phase 1: Draft committee from Global50')
    parser.add_argument('--validate', action='store_true',
                        help='Run Phase 2: Validate on holdout slices')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify data split')
    parser.add_argument('--mirror', action='store_true',
                        help='Check cloud sync status, download missing files')
    args = parser.parse_args()

    if not args.draft and not args.validate and not args.verify_only and not args.mirror:
        print("Usage: python committee.py [--draft] [--validate] [--verify-only] [--mirror]")
        print("\nOptions:")
        print("  --draft        Run Phase 1: Draft committee from Global50")
        print("  --validate     Run Phase 2: Validate committee on holdout slices")
        print("  --verify-only  Verify data split without running")
        print("  --mirror       Check cloud sync status, download missing files")
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

    # Handle --mirror mode
    if args.mirror:
        manager.check_mirror_status()
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

    if args.draft:
        roster = run_draft(manager, loader, stats, holdout_info)

        # Auto-run validation after draft if requested
        if roster and args.validate:
            gc.collect()
            torch.cuda.empty_cache()

    if args.validate:
        validation_results = run_validation(
            manager, loader, stats, holdout_info, context_window_days
        )

        if validation_results:
            # Update roster with validation results
            roster = manager.load_roster()
            if roster:
                roster['validation'] = validation_results
                manager.save_roster(roster)
                print(f"\n✓ Validation results added and synced to cloud")
