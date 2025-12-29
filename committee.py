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
        # Convert numpy types to native Python types for JSON serialization
        roster_data_clean = convert_numpy_types(roster_data)

        # Save roster JSON locally
        with open(self.local_roster_path, 'w') as f:
            json.dump(roster_data_clean, f, indent=2)
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

def convert_numpy_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.

    Args:
        obj: Object to convert (dict, list, numpy type, or primitive)

    Returns:
        Object with all numpy types converted to Python types
    """
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    else:
        return obj


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


def get_validation_data(loader, stats, holdout_info):
    """
    Get VALIDATION data tensor for correlation calculation during Draft.

    IMPORTANT: This uses validation data (NOT holdout) to prevent data leakage.
    The holdout period must remain unseen until final validation.

    Returns:
        (inputs_tensor, valid_indices)
    """
    val_start = holdout_info['val_start']
    val_end = holdout_info['val_end']

    input_data = loader.data_array  # [Days, Stocks, 5_feats]

    # Valid inference points within validation period
    # Need context window before and room for MIN_HOLDING_PERIOD forward
    valid_indices = list(range(
        max(val_start, Config.CONTEXT_WINDOW_DAYS),
        val_end - Config.MIN_HOLDING_PERIOD + 1
    ))

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
        # Agent i has lower fitness, swap it out
        return i, j, max_corr, entries[i], entries[j]
    else:
        # Agent j has lower fitness, swap it out
        return j, i, max_corr, entries[j], entries[i]


def find_best_swap_candidate(committee_indices: tuple, drop_idx: int, entries: list,
                              corr_matrix: np.ndarray,
                              optimize_for: str = 'objective') -> tuple:
    """
    Find the agent from the entire Global50 that best improves the committee
    when swapped in for the dropped agent.

    Args:
        committee_indices: Current committee indices
        drop_idx: Index of agent being dropped
        entries: Full list of Global50 entries
        corr_matrix: Full NxN correlation matrix
        optimize_for: What to optimize - 'objective' (default) or 'max_corr'

    Returns:
        (best_candidate_idx, new_committee_indices, new_objective, new_score_sum,
         new_avg_corr, new_max_corr, candidate_entry)
    """
    # Build remaining committee (without the dropped agent)
    remaining = [idx for idx in committee_indices if idx != drop_idx]

    best_candidate = None
    best_objective = float('-inf')
    best_max_corr = float('inf')  # For max_corr optimization
    best_new_indices = None
    best_score_sum = 0
    best_avg_corr = 0
    best_max_corr_result = 0

    # Valid entries have self-correlation = 1
    valid_entries = [i for i in range(len(entries))
                     if corr_matrix[i, i] == 1.0]

    for candidate_idx in valid_entries:
        # Skip if already in committee
        if candidate_idx in committee_indices:
            continue

        # Create new committee with swap
        new_indices = tuple(sorted(remaining + [candidate_idx]))

        # Calculate objective
        obj, score_sum, avg_corr, max_corr = committee_objective(
            new_indices, entries, corr_matrix
        )

        # Choose based on optimization target
        if optimize_for == 'max_corr':
            # Minimize max correlation (lower is better)
            if max_corr < best_max_corr:
                best_max_corr = max_corr
                best_candidate = candidate_idx
                best_new_indices = new_indices
                best_objective = obj
                best_score_sum = score_sum
                best_avg_corr = avg_corr
                best_max_corr_result = max_corr
        else:
            # Maximize objective (higher is better)
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
    # The multiplier must push the objective AWAY from zero when correlation is high.
    #
    # If score_sum > 0: High correlation should reduce it (multiply by < 1)
    # If score_sum < 0: High correlation should make it MORE negative (multiply by > 1)
    #
    # Without this fix, negative scores get "shrunk" toward zero by high correlation,
    # causing the optimizer to favor redundant failure over diverse success.
    if score_sum >= 0:
        multiplier = 1.0 - avg_corr
    else:
        multiplier = 1.0 + avg_corr

    objective = score_sum * multiplier

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


def interactive_correlation_refinement(committee_indices: tuple, entries: list,
                                         corr_matrix: np.ndarray) -> tuple:
    """
    Interactive refinement pass: iteratively swap out high-correlation agents.

    For each iteration:
    1. Find the highest correlation pair in the committee
    2. Offer to swap out the lower-fitness agent
    3. If user accepts, find best replacement from entire Global50
    4. Show old vs new metrics, confirm swap
    5. Repeat until user declines a swap

    Args:
        committee_indices: Initial committee indices from optimization
        entries: Full list of Global50 entries
        corr_matrix: Full NxN correlation matrix

    Returns:
        Final committee indices after all swaps
    """
    current_indices = committee_indices
    swap_count = 0

    print(f"\n{'='*60}")
    print("PHASE 1b: CORRELATION REFINEMENT (Interactive)")
    print(f"{'='*60}")
    print("This pass allows you to iteratively reduce correlation by swapping agents.")
    print("Press Enter for 'no' to skip, or type 'y' to swap.\n")

    while True:
        # Calculate current metrics
        curr_obj, curr_score_sum, curr_avg_corr, curr_max_corr = committee_objective(
            current_indices, entries, corr_matrix
        )

        # Find highest correlation pair
        drop_idx, keep_idx, max_corr, drop_entry, keep_entry = find_highest_correlation_pair(
            current_indices, entries, corr_matrix
        )

        if drop_idx is None:
            print("  No valid pairs found in committee.")
            break

        # Find best replacement for OBJECTIVE (default)
        (obj_candidate_idx, obj_new_indices, obj_new_obj, obj_new_score_sum,
         obj_new_avg_corr, obj_new_max_corr, obj_candidate_entry) = find_best_swap_candidate(
            current_indices, drop_idx, entries, corr_matrix, optimize_for='objective'
        )

        # Find best replacement for MAX CORRELATION reduction
        (corr_candidate_idx, corr_new_indices, corr_new_obj, corr_new_score_sum,
         corr_new_avg_corr, corr_new_max_corr, corr_candidate_entry) = find_best_swap_candidate(
            current_indices, drop_idx, entries, corr_matrix, optimize_for='max_corr'
        )

        if obj_candidate_idx is None and corr_candidate_idx is None:
            print("  ⚠ No valid replacement found for highest correlation pair.")
            break

        # Display the complete swap proposal upfront
        print(f"\n  {'─'*56}")
        print(f"  HIGHEST CORRELATION PAIR:")
        print(f"    Agent A: {drop_entry['run_name']}_{drop_entry['agent_id']} "
              f"(fitness: {drop_entry['gauntlet_score']:.2f})")
        print(f"    Agent B: {keep_entry['run_name']}_{keep_entry['agent_id']} "
              f"(fitness: {keep_entry['gauntlet_score']:.2f})")
        print(f"    Correlation: {max_corr:.4f}")

        # Check if the two strategies give different candidates
        same_candidate = (obj_candidate_idx == corr_candidate_idx)

        if same_candidate:
            # Single option - same candidate for both strategies
            candidate_idx = obj_candidate_idx
            new_indices = obj_new_indices
            new_obj = obj_new_obj
            new_score_sum = obj_new_score_sum
            new_avg_corr = obj_new_avg_corr
            new_max_corr = obj_new_max_corr
            candidate_entry = obj_candidate_entry

            print(f"\n  PROPOSED SWAP:")
            print(f"    OUT: {drop_entry['run_name']}_{drop_entry['agent_id']} "
                  f"(fitness: {drop_entry['gauntlet_score']:.2f})")
            print(f"    IN:  {candidate_entry['run_name']}_{candidate_entry['agent_id']} "
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
                current_indices = new_indices
                swap_count += 1
                print(f"\n  ✓ Swap #{swap_count} confirmed.")
            else:
                print(f"\n  ✗ Swap cancelled. Keeping current composition.")
                print(f"\n  ✓ Committee composition confirmed.")
                break
        else:
            # Two different options - show both
            print(f"\n  TWO SWAP OPTIONS AVAILABLE:")
            print(f"    OUT: {drop_entry['run_name']}_{drop_entry['agent_id']} "
                  f"(fitness: {drop_entry['gauntlet_score']:.2f})")

            # Option 1: Best for objective
            print(f"\n  [1] BEST FOR OBJECTIVE:")
            print(f"      IN:  {obj_candidate_entry['run_name']}_{obj_candidate_entry['agent_id']} "
                  f"(fitness: {obj_candidate_entry['gauntlet_score']:.2f})")
            obj_corr_with_kept = corr_matrix[obj_candidate_idx, keep_idx]
            print(f"      Pair corr: {obj_corr_with_kept:.4f} (Δ{obj_corr_with_kept - max_corr:+.4f})")
            print(f"      Objective: {obj_new_obj:.2f} (Δ{obj_new_obj - curr_obj:+.2f}), "
                  f"Max corr: {obj_new_max_corr:.4f} (Δ{obj_new_max_corr - curr_max_corr:+.4f})")

            # Option 2: Best for max correlation
            print(f"\n  [2] BEST FOR MAX CORRELATION:")
            print(f"      IN:  {corr_candidate_entry['run_name']}_{corr_candidate_entry['agent_id']} "
                  f"(fitness: {corr_candidate_entry['gauntlet_score']:.2f})")
            corr_corr_with_kept = corr_matrix[corr_candidate_idx, keep_idx]
            print(f"      Pair corr: {corr_corr_with_kept:.4f} (Δ{corr_corr_with_kept - max_corr:+.4f})")
            print(f"      Objective: {corr_new_obj:.2f} (Δ{corr_new_obj - curr_obj:+.2f}), "
                  f"Max corr: {corr_new_max_corr:.4f} (Δ{corr_new_max_corr - curr_max_corr:+.4f})")

            print(f"\n  → Choose swap: [1] objective, [2] max-corr, [N] skip")
            confirm = input("    [1/2/N]: ").strip().lower()

            if confirm == '1':
                current_indices = obj_new_indices
                swap_count += 1
                print(f"\n  ✓ Swap #{swap_count} confirmed (objective-optimized).")
            elif confirm == '2':
                current_indices = corr_new_indices
                swap_count += 1
                print(f"\n  ✓ Swap #{swap_count} confirmed (max-corr-optimized).")
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
                                      max_iterations: int = 50) -> tuple:
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

    Returns:
        Final committee indices after all swaps
    """
    current_indices = committee_indices
    swap_count = 0

    # Calculate starting metrics
    start_obj, start_score_sum, start_avg_corr, start_max_corr = committee_objective(
        current_indices, entries, corr_matrix
    )

    print(f"\n{'='*60}")
    print("PHASE 1b: CORRELATION REFINEMENT (Automatic Deep)")
    print(f"{'='*60}")
    print(f"Starting: obj={start_obj:.2f}, max_corr={start_max_corr:.4f}")

    # Valid entries for candidate search
    valid_entries = [i for i in range(len(entries))
                     if corr_matrix[i, i] == 1.0]

    for iteration in range(1, max_iterations + 1):
        # Calculate current metrics
        curr_obj, curr_score_sum, curr_avg_corr, curr_max_corr = committee_objective(
            current_indices, entries, corr_matrix
        )

        # Find highest correlation pair
        drop_idx, keep_idx, max_corr, drop_entry, keep_entry = find_highest_correlation_pair(
            current_indices, entries, corr_matrix
        )

        if drop_idx is None:
            print(f"\n  No valid pairs found in committee.")
            break

        print(f"\n  Iteration {iteration}:")
        print(f"    Highest pair: {drop_entry['run_name']}_{drop_entry['agent_id']} <-> "
              f"{keep_entry['run_name']}_{keep_entry['agent_id']} (corr={max_corr:.4f})")

        # Build remaining committee (without the dropped agent)
        remaining = [idx for idx in current_indices if idx != drop_idx]

        # Test all valid candidates
        improving_candidates = []

        for candidate_idx in valid_entries:
            # Skip if already in committee
            if candidate_idx in current_indices:
                continue

            # Create new committee with swap
            new_indices = tuple(sorted(remaining + [candidate_idx]))

            # Calculate metrics for this swap
            new_obj, new_score_sum, new_avg_corr, new_max_corr = committee_objective(
                new_indices, entries, corr_matrix
            )

            # Only consider if objective IMPROVES
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

        # Among those that improve objective, pick the one with lowest max_corr
        improving_candidates.sort(key=lambda x: x['max_corr'])
        best = improving_candidates[0]

        # Only apply if max_corr also improves
        if best['max_corr'] >= curr_max_corr:
            print(f"    Best objective-improving candidate has max_corr={best['max_corr']:.4f} "
                  f"(not better than {curr_max_corr:.4f})")
            print(f"    Stopping")
            break

        # Apply the swap
        print(f"    Best among those (lowest max_corr): "
              f"{best['entry']['run_name']}_{best['entry']['agent_id']}")
        print(f"      obj: {curr_obj:.2f} -> {best['obj']:.2f} ({best['obj'] - curr_obj:+.2f})")
        print(f"      max_corr: {curr_max_corr:.4f} -> {best['max_corr']:.4f} "
              f"({best['max_corr'] - curr_max_corr:+.4f})")
        print(f"    Swap applied")

        current_indices = best['new_indices']
        swap_count += 1

    # Final summary
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


# --- Committee Agent Class ---

class CommitteeAgent:
    """
    Committee-as-Agent: Aggregates multiple agents' decisions into a single action vector.

    This class acts as a drop-in replacement for a single agent, implementing the same
    interface (predict action from observation) but using committee consensus logic internally.
    """

    def __init__(self, members: list, context_window_days: int, track_consensus: bool = False):
        """
        Initialize committee agent.

        Args:
            members: List of member dicts with agent metadata and stats
            context_window_days: Context window for loading agent files
            track_consensus: If True, track consensus statistics per step
        """
        self.members = members
        self.context_window_days = context_window_days
        self.loaded_agents = []
        self.track_consensus = track_consensus

        # Consensus tracking
        if self.track_consensus:
            self.consensus_history = {
                'unanimity_count': 0,
                'min_consensus_count': 0,
                'total_steps': 0,
                'total_trades': 0,
                'trades_by_quorum': 0,
                'trades_by_conviction': 0,
                'trades_vetoed': 0,
                'avg_votes_per_trade': [],
            }
            # Track stocks already signaled this episode to prevent double-counting
            self.signaled_stocks = set()

        # Load all member agents (actor only)
        print(f"Loading {len(members)} committee members...")
        for member in tqdm(members, desc="Loading agents"):
            filepath = get_agent_filepath(member, context_window_days)
            agent = load_agent_actor_only(filepath, member['agent_id'])
            if agent is None:
                raise ValueError(f"Failed to load agent: {filepath}")
            self.loaded_agents.append(agent)

        print(f"✓ Committee agent ready with {len(self.loaded_agents)} members")

    def predict_action(self, observation: np.ndarray, held_stock_ids: list = None) -> np.ndarray:
        """
        Generate committee consensus action from observation.

        Args:
            observation: Normalized observation [context_window, num_stocks, features]
            held_stock_ids: Optional list of stock IDs already held (prevents re-signaling)

        Returns:
            action: [num_stocks, 2] array with [coefficient, sale_target] per stock
        """
        # Convert to tensor
        obs_tensor = torch.FloatTensor(observation).unsqueeze(0).to(Config.DEVICE)

        # Collect predictions from all members
        all_actions = []

        with torch.no_grad():
            for agent in self.loaded_agents:
                action = agent.actor(obs_tensor).cpu().numpy()[0]  # [num_stocks, 2]
                all_actions.append(action)

        # Stack: [num_members, num_stocks, 2]
        all_actions = np.array(all_actions)

        # Extract coefficients and sale targets
        all_coeffs = all_actions[:, :, 0]  # [num_members, num_stocks]
        all_sale_targets = all_actions[:, :, 1]  # [num_members, num_stocks]

        # Apply committee consensus logic
        final_coeffs = self._apply_consensus(all_coeffs, held_stock_ids)

        # Average sale targets for stocks that passed consensus
        # For stocks with coeff=0 (not approved), sale target doesn't matter
        final_sale_targets = np.mean(all_sale_targets, axis=0)

        # Combine into action
        action = np.stack([final_coeffs, final_sale_targets], axis=1)  # [num_stocks, 2]

        return action

    def _apply_consensus(self, all_coeffs: np.ndarray, held_stock_ids: list = None) -> np.ndarray:
        """
        Apply committee consensus logic to coefficient predictions.

        Args:
            all_coeffs: [num_members, num_stocks] coefficient predictions
            held_stock_ids: List of stock IDs to exclude from signaling

        Returns:
            final_coeffs: [num_stocks] consensus coefficients
        """
        num_members, num_stocks = all_coeffs.shape

        # 1. Standard Voting (Quorum) - coeff >= global threshold
        votes = all_coeffs >= Config.COEFFICIENT_THRESHOLD
        vote_counts = np.sum(votes, axis=0)  # [num_stocks]
        is_quorum = vote_counts >= Config.COMMITTEE_QUORUM

        # 2. Conviction Check (Stock-Specific Override)
        # Conviction = agent exceeds their own P99 threshold for that stock
        conviction_thresholds = np.array([
            m['stats']['conviction_threshold_vector'] for m in self.members
        ])  # [num_members, num_stocks]

        agent_convictions = all_coeffs > conviction_thresholds
        is_conviction_raw = np.any(agent_convictions, axis=0)  # [num_stocks]

        # SOCIAL PROOF: Conviction counts as a vote for the social proof check.
        # An agent with conviction is "voting" even if below global threshold.
        # This allows a specialist (coeff=0.9, p99=0.8) to count toward social proof.
        effective_votes = votes | agent_convictions  # [num_members, num_stocks]
        effective_vote_counts = np.sum(effective_votes, axis=0)  # [num_stocks]

        # A conviction trade requires at least 2 total supporters (votes OR convictions)
        # This prevents a single hallucinating agent from triggering trades alone.
        is_conviction = is_conviction_raw & (effective_vote_counts >= 2)

        # 3. Veto Check (Fixed number of silent members)
        is_silent = all_coeffs < Config.COMMITTEE_VETO_THRESHOLD
        silent_counts = np.sum(is_silent, axis=0)
        is_vetoed = silent_counts >= Config.COMMITTEE_VETO_COUNT

        # 4. Final Decision Logic
        # PASS if: (Quorum OR Conviction) AND (NOT Veto)
        should_trade = (is_quorum | is_conviction) & (~is_vetoed)

        # FILTER: Mask out stocks we already hold to prevent "Ghost Signals" in stats
        if held_stock_ids:
            held_mask = np.zeros(num_stocks, dtype=bool)
            held_mask[held_stock_ids] = True
            # If we hold it, we don't "trade" it (prevents stats inflation)
            should_trade = should_trade & (~held_mask)

        # Track consensus stats if enabled
        if self.track_consensus:
            self.consensus_history['total_steps'] += 1

            # Filter to only count stocks we haven't signaled before (prevents double-counting)
            stocks_to_count = should_trade.copy()
            for stock_id in range(num_stocks):
                if should_trade[stock_id] and stock_id in self.signaled_stocks:
                    stocks_to_count[stock_id] = False  # Already counted this stock
                elif should_trade[stock_id]:
                    self.signaled_stocks.add(stock_id)  # Mark as signaled

            # Count trades approved by each mechanism (only new signals)
            trades_approved = np.sum(stocks_to_count)
            self.consensus_history['total_trades'] += int(trades_approved)

            # Unanimity: all members voted for the trade
            unanimity = np.sum(vote_counts[stocks_to_count] == num_members)
            self.consensus_history['unanimity_count'] += int(unanimity)

            # Min consensus: exactly quorum votes
            min_consensus = np.sum(vote_counts[stocks_to_count] == Config.COMMITTEE_QUORUM)
            self.consensus_history['min_consensus_count'] += int(min_consensus)

            # Trades by Quorum vs Conviction (MUTUALLY EXCLUSIVE)
            # For trades that actually happened, determine which mechanism approved them
            # If quorum was met, credit quorum (even if conviction also triggered)
            # Only credit conviction for trades where quorum was NOT met (conviction "rescues")
            is_quorum_trade = is_quorum & stocks_to_count
            is_conviction_only_trade = (~is_quorum) & stocks_to_count

            # Verify mathematical correctness: these should sum to total trades
            self.consensus_history['trades_by_quorum'] += int(np.sum(is_quorum_trade))
            self.consensus_history['trades_by_conviction'] += int(np.sum(is_conviction_only_trade))

            # Trades vetoed (Only count vetoes on signals that otherwise would have passed)
            # This filters out "vetoes" on stocks nobody wanted anyway
            potential_trades = (is_quorum | is_conviction)
            if held_stock_ids:
                potential_trades = potential_trades & (~held_mask)

            vetoed_trades = potential_trades & is_vetoed
            self.consensus_history['trades_vetoed'] += int(np.sum(vetoed_trades))

            # Average votes per trade
            if trades_approved > 0:
                avg_votes = np.mean(vote_counts[stocks_to_count])
                self.consensus_history['avg_votes_per_trade'].append(float(avg_votes))

        # 5. Signal Aggregation
        # For approved trades, average coefficients from ALL supporting agents
        # (both standard voters AND conviction agents)
        final_coeffs = np.zeros(num_stocks, dtype=np.float32)

        for stock_idx in range(num_stocks):
            if should_trade[stock_idx]:
                # Get all agents who support this trade (vote OR conviction)
                # This ensures the conviction specialist is included in the average
                supporting_agents_mask = effective_votes[:, stock_idx]  # Boolean mask

                if np.any(supporting_agents_mask):
                    # Average coefficients from all supporting agents
                    supporting_coeffs = all_coeffs[supporting_agents_mask, stock_idx]
                    final_coeffs[stock_idx] = np.mean(supporting_coeffs)
                else:
                    # Fallback to threshold (shouldn't happen, but safety)
                    final_coeffs[stock_idx] = Config.COEFFICIENT_THRESHOLD

        return final_coeffs

    def get_consensus_summary(self) -> dict:
        """
        Get summary of consensus statistics from tracked history.

        Returns:
            Dict with consensus metrics (empty if tracking disabled)
        """
        if not self.track_consensus:
            return {'note': 'Consensus tracking was not enabled'}

        history = self.consensus_history
        total_steps = history['total_steps']
        total_trades = history['total_trades']

        if total_steps == 0:
            return {'note': 'No steps recorded'}

        return {
            'total_steps': total_steps,
            'total_trades': total_trades,
            'avg_trades_per_step': total_trades / total_steps if total_steps > 0 else 0,
            'unanimity_pct': 100.0 * history['unanimity_count'] / total_trades if total_trades > 0 else 0,
            'min_consensus_pct': 100.0 * history['min_consensus_count'] / total_trades if total_trades > 0 else 0,
            'trades_by_quorum': history['trades_by_quorum'],
            'trades_by_conviction': history['trades_by_conviction'],
            'trades_vetoed': history['trades_vetoed'],
            'avg_consensus_votes': float(np.mean(history['avg_votes_per_trade'])) if history['avg_votes_per_trade'] else 0,
        }

    def reset_episode(self):
        """Reset episode-level tracking (call at start of each new episode)."""
        if self.track_consensus:
            self.signaled_stocks = set()

    def cleanup(self):
        """Release GPU memory from loaded agents."""
        for agent in self.loaded_agents:
            del agent
        self.loaded_agents.clear()
        torch.cuda.empty_cache()


def calculate_agent_stats_vectorized(agent_coeff_history_2d: np.ndarray,
                                      percentile: int = 95) -> np.ndarray:
    """
    Calculates the Nth percentile conviction threshold for each stock.

    CRITICAL: Only considers coefficients that would actually trigger a trade
    (>= Config.COEFFICIENT_THRESHOLD). Including non-trading noise (< 1.0) drags
    the percentile down, allowing sub-threshold signals to masquerade as 'high conviction'.

    Args:
        agent_coeff_history_2d: Numpy array [Days, Stocks] for a single agent
        percentile: Percentile to use for conviction threshold (default: 95)

    Returns:
        threshold_vector: Numpy array [Stocks] of Nth percentile conviction thresholds
    """
    days, num_stocks = agent_coeff_history_2d.shape
    threshold_vector = np.zeros(num_stocks, dtype=np.float32)

    # Filter: Only look at coefficients that are actual trades
    # We use Config.COEFFICIENT_THRESHOLD (1.0) as the floor
    valid_trades_mask = agent_coeff_history_2d >= Config.COEFFICIENT_THRESHOLD
    all_active = agent_coeff_history_2d[valid_trades_mask]

    # Calculate Global percentile fallback
    # If the agent has NEVER traded (or very rarely), we set a high fallback
    # so it cannot easily trigger conviction on noise.
    if len(all_active) > 0:
        global_threshold = np.percentile(all_active, percentile)
    else:
        # Agent is a ghost (no trades > 1.0). Set threshold to infinity to disable conviction.
        global_threshold = 100.0

    for i in range(num_stocks):
        # Extract history for this specific stock
        stock_coeffs = agent_coeff_history_2d[:, i]

        # Only calculate percentile based on actual trades for this stock
        active_coeffs = stock_coeffs[stock_coeffs >= Config.COEFFICIENT_THRESHOLD]

        if len(active_coeffs) >= 20:
            # Sufficient history for this stock
            threshold_vector[i] = np.percentile(active_coeffs, percentile)
        else:
            # Insufficient history, fallback to global threshold
            threshold_vector[i] = global_threshold

    return threshold_vector


def recalculate_conviction_thresholds(members: list, loader, stats, holdout_info,
                                       context_window_days: int, percentile: int) -> list:
    """
    Recalculate conviction thresholds for all committee members at a given percentile.

    This creates a deep copy of members with updated conviction_threshold_vector values,
    avoiding pollution of the original roster data.

    Args:
        members: Original list of member dicts from roster
        loader: StockDataLoader instance
        stats: Normalization stats dict
        holdout_info: Holdout period info dict
        context_window_days: Context window size
        percentile: Percentile to use for conviction threshold (e.g., 90, 95, 99)

    Returns:
        New list of member dicts with recalculated conviction thresholds
    """
    import copy

    # Deep copy to avoid modifying original
    new_members = copy.deepcopy(members)

    # Get validation data tensor (same as used during draft)
    val_tensor, valid_indices = get_validation_data(loader, stats, holdout_info)
    num_days = len(valid_indices)
    num_stocks = Config.NUM_INVESTABLE_STOCKS

    for member in tqdm(new_members, desc=f"Recalculating P{percentile} thresholds"):
        filepath = get_agent_filepath(member, context_window_days)
        agent = load_agent_actor_only(filepath, 0)

        if agent is not None:
            # Generate coefficients using same method as calculate_coefficient_correlations
            with torch.no_grad():
                batch_size = 32
                num_samples = val_tensor.shape[0]
                all_coefs = []

                for start_idx in range(0, num_samples, batch_size):
                    end_idx = min(start_idx + batch_size, num_samples)
                    batch = val_tensor[start_idx:end_idx]
                    batch_actions = agent.actor(batch).cpu().numpy()
                    # Extract coefficients (first output dimension)
                    all_coefs.append(batch_actions[:, :, 0])

                # Concatenate to [Days, Stocks]
                agent_coeffs_2d = np.concatenate(all_coefs, axis=0)

            # Recalculate with new percentile
            conviction_threshold_vector = calculate_agent_stats_vectorized(
                agent_coeffs_2d, percentile=percentile
            )
            member['stats']['conviction_threshold_vector'] = conviction_threshold_vector.tolist()

            del agent
            torch.cuda.empty_cache()

    return new_members


# --- Validation ---

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

    # Load agent (actor only)
    agent = load_agent_actor_only(agent_path, 0)
    if agent is None:
        return {'error': 'Failed to load agent'}

    # Create trading environment for this slice
    env = TradingEnvironment(
        data_array=loader.data_array,
        dates=loader.dates,
        normalization_stats=stats,
        start_idx=slice_start_idx,
        end_idx=slice_end_idx,
        data_array_full=loader.data_array_full,
        is_training=False,  # No observation noise during validation
        gauntlet_mode=True  # Use soft zero-trades penalty
    )

    # Run episode
    obs, info = env.reset()
    terminated = False

    with torch.no_grad():
        while not terminated:
            # Convert observation to tensor and get action
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(Config.DEVICE)
            action = agent.actor(obs_tensor).cpu().numpy()[0]  # [num_stocks, 2]

            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)

    # Get episode summary
    summary = env.get_episode_summary()

    # Clean up
    del agent
    del env
    torch.cuda.empty_cache()

    # Return metrics in expected format
    return {
        'num_trades': summary['num_trades'],
        'win_rate': summary['win_rate'],
        'quality_ratio': (summary['num_wins'] / summary['num_losses']) if summary['num_losses'] > 0 else float('inf'),
        'expectancy': summary['avg_reward_per_trade'],
        'roi': summary['roi'],
        'fitness': summary['total_reward'],  # Total reward is the fitness
    }


def evaluate_committee_on_slice(members: list, loader, stats,
                                 context_window_days: int,
                                 slice_start_idx: int,
                                 slice_end_idx: int) -> dict:
    """
    Evaluate committee consensus on a holdout slice using TradingEnvironment.

    The committee acts as a single agent, using consensus logic to generate actions
    that are then executed through the standard trading environment.

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

    # Create committee agent (loads all members) with consensus tracking enabled
    committee = CommitteeAgent(members, context_window_days, track_consensus=True)

    # Create trading environment for this slice
    env = TradingEnvironment(
        data_array=loader.data_array,
        dates=loader.dates,
        normalization_stats=stats,
        start_idx=slice_start_idx,
        end_idx=slice_end_idx,
        data_array_full=loader.data_array_full,
        is_training=False,  # No observation noise during validation
        gauntlet_mode=True  # Use soft zero-trades penalty
    )

    # Run episode with committee making decisions
    obs, info = env.reset()
    terminated = False

    # Reset committee episode tracking
    committee.reset_episode()

    while not terminated:
        # Get currently held stocks to prevent re-signaling
        held_ids = list(env.open_positions.keys())

        # Get committee consensus action
        action = committee.predict_action(obs, held_stock_ids=held_ids)  # [num_stocks, 2]

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

    # Get episode summary
    summary = env.get_episode_summary()

    # Get consensus statistics
    consensus_stats = committee.get_consensus_summary()

    # Extract closed trades for CSV export and enrich with additional fields
    closed_trades = summary.get('closed_trades', [])

    # Enrich closed trades with stock_name and exit_date
    for trade in closed_trades:
        # Add stock_name from loader's column_names
        stock_id = trade.get('stock_id')
        if stock_id is not None and loader.column_names:
            # stock_id is relative to investable stocks (starts at column 9)
            actual_col_idx = Config.INVESTABLE_START_COL + stock_id
            if actual_col_idx < len(loader.column_names):
                trade['stock_name'] = loader.column_names[actual_col_idx]
            else:
                trade['stock_name'] = f"UNKNOWN_{stock_id}"
        else:
            trade['stock_name'] = f"UNKNOWN_{stock_id}"

        # Add exit_date (calendar date when position was closed)
        # The 'day' field in the trade is already the exit date (when close action occurred)
        trade['exit_date'] = trade.get('day', '')

    # Clean up
    committee.cleanup()
    del env
    torch.cuda.empty_cache()

    # Return metrics in expected format
    return {
        'num_trades': summary['num_trades'],
        'win_rate': summary['win_rate'],
        'quality_ratio': (summary['num_wins'] / summary['num_losses']) if summary['num_losses'] > 0 else float('inf'),
        'expectancy': summary['avg_reward_per_trade'],
        'roi': summary['roi'],
        'fitness': summary['total_reward'],  # Total reward is the fitness
        'consensus_stats': consensus_stats,
        'closed_trades': closed_trades,  # Include for CSV export
        'raw_pnl': summary.get('raw_pnl', 0.0),
        'peak_capital_employed': summary.get('peak_capital_employed', 0.0),
    }


def run_validation(manager: CommitteeManager, loader, stats, holdout_info,
                   context_window_days: int, members_override: list = None,
                   conviction_percentile: int = None) -> dict:
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

    Returns validation results dict with comprehensive metrics.
    """
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
        # Recalculate conviction thresholds with custom percentile
        print(f"  Recalculating conviction thresholds at P{conviction_percentile}...")
        members = recalculate_conviction_thresholds(
            roster['members'], loader, stats, holdout_info,
            context_window_days, conviction_percentile
        )
    else:
        members = roster['members']
    num_slices = Config.COMMITTEE_VALIDATION_SLICES

    # Episode structure (matching training):
    # - Each episode is TRADING_PERIOD_DAYS (125) + SETTLEMENT_PERIOD_DAYS (30) = 155 days
    # - Episodes can overlap
    episode_length = Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

    # Calculate validation range (absolute day indices)
    val_start_idx = holdout_info['val_start']
    val_end_idx = holdout_info['val_end']

    # Calculate holdout range (absolute day indices)
    holdout_start_idx = holdout_info['holdout_start']
    holdout_end_idx = holdout_info['holdout_end']

    # Total available data
    val_days = val_end_idx - val_start_idx + 1
    holdout_days = holdout_end_idx - holdout_start_idx + 1
    total_days = val_days + holdout_days

    # Generate deterministic slices: 3 from validation, 2 from holdout
    # Each slice is exactly episode_length days
    num_val_slices = 3
    num_holdout_slices = 2

    # Calculate deterministic start points (evenly spaced)
    # For validation: divide available range into num_val_slices segments
    val_usable_range = val_days - episode_length
    val_step = val_usable_range // (num_val_slices - 1) if num_val_slices > 1 else 0

    # For holdout: divide available range into num_holdout_slices segments
    holdout_usable_range = holdout_days - episode_length
    holdout_step = holdout_usable_range // (num_holdout_slices - 1) if num_holdout_slices > 1 else 0

    print(f"\n  Episode length: {episode_length} days ({Config.TRADING_PERIOD_DAYS} trading + {Config.SETTLEMENT_PERIOD_DAYS} settlement)")
    print(f"  Validation range: indices {val_start_idx} to {val_end_idx} ({val_days} days)")
    print(f"    → {num_val_slices} episodes of {episode_length} days each")
    print(f"  Holdout range: indices {holdout_start_idx} to {holdout_end_idx} ({holdout_days} days)")
    print(f"    → {num_holdout_slices} episodes of {episode_length} days each")
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
            # Calculate absolute slice indices (deterministic, evenly spaced)
            # Each slice is exactly episode_length days
            if s < num_val_slices:
                # Validation slices (0, 1, 2) - evenly spaced across validation period
                slice_start = val_start_idx + (s * val_step)
                slice_end = slice_start + episode_length  # Exclusive end
                slice_type = 'validation'
            else:
                # Holdout slices (3, 4) - evenly spaced across holdout period
                holdout_slice_idx = s - num_val_slices
                slice_start = holdout_start_idx + (holdout_slice_idx * holdout_step)
                slice_end = slice_start + episode_length  # Exclusive end
                slice_type = 'holdout'

            # Get date strings for this slice
            start_date = loader.dates[slice_start]
            end_date = loader.dates[slice_end - 1]  # -1 because slice_end is exclusive

            metrics = evaluate_agent_on_slice(filepath, loader, stats, slice_start, slice_end)
            agent_results['slice_metrics'].append({
                'slice': s,
                'slice_type': slice_type,
                'start_idx': slice_start,
                'end_idx': slice_end - 1,  # Store inclusive end
                'start_date': start_date,
                'end_date': end_date,
                'num_days': slice_end - slice_start,
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

    for s in range(num_slices):
        # Calculate absolute slice indices (deterministic, evenly spaced)
        # Each slice is exactly episode_length days
        if s < num_val_slices:
            # Validation slices (0, 1, 2) - evenly spaced across validation period
            slice_start = val_start_idx + (s * val_step)
            slice_end = slice_start + episode_length  # Exclusive end
            slice_type = 'validation'
        else:
            # Holdout slices (3, 4) - evenly spaced across holdout period
            holdout_slice_idx = s - num_val_slices
            slice_start = holdout_start_idx + (holdout_slice_idx * holdout_step)
            slice_end = slice_start + episode_length  # Exclusive end
            slice_type = 'holdout'

        # Get date strings for this slice
        start_date = loader.dates[slice_start]
        end_date = loader.dates[slice_end - 1]  # -1 because slice_end is exclusive

        metrics = evaluate_committee_on_slice(
            members, loader, stats, context_window_days, slice_start, slice_end
        )

        # Save closed trades to CSV and upload to cloud
        closed_trades = metrics.get('closed_trades', [])
        if closed_trades:
            csv_filename = f"committee_slice_{s}_{slice_type}_{start_date}_to_{end_date}.csv"
            csv_path = manager.local_committee_dir / csv_filename

            # Write CSV with explicit column ordering for readability
            import csv
            # Define preferred column order (new fields: stock_name, exit_date, coefficient)
            preferred_columns = [
                'stock_id', 'stock_name', 'entry_date', 'exit_date', 'days_held',
                'entry_price', 'exit_price', 'gain_pct', 'coefficient', 'reason',
                'base_reward', 'forced_exit_penalty', 'reward', 'day', 'action'
            ]
            # Get actual columns from first trade, preserving any extras
            actual_columns = list(closed_trades[0].keys())
            # Order: preferred columns first (if present), then any remaining
            fieldnames = [c for c in preferred_columns if c in actual_columns]
            fieldnames += [c for c in actual_columns if c not in fieldnames]

            with open(csv_path, 'w', newline='') as f:
                if len(closed_trades) > 0:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(closed_trades)

            # Generate xlsx position tracking file
            xlsx_path = csv_path.parent / f"{csv_path.stem}_positions.xlsx"
            try:
                from transform_trades_to_positions import transform_trades_to_positions

                # Parse slice dates for the full range
                slice_start_dt = datetime.strptime(start_date, "%d-%m-%y")
                slice_end_dt = datetime.strptime(end_date, "%d-%m-%y")

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
            cloud_csv_path = f"{manager.cloud_committee_base}/{csv_filename}"
            if manager.cloud_sync.provider != "local":
                if manager.cloud_sync.upload_file_verified(str(csv_path), cloud_csv_path):
                    print(f"  ✓ Uploaded CSV: {csv_filename}")
                else:
                    print(f"  ✗ Failed to upload CSV: {csv_filename}")

                # Upload xlsx if it exists
                if xlsx_path and xlsx_path.exists():
                    xlsx_filename = xlsx_path.name
                    cloud_xlsx_path = f"{manager.cloud_committee_base}/{xlsx_filename}"
                    if manager.cloud_sync.upload_file_verified(str(xlsx_path), cloud_xlsx_path):
                        print(f"  ✓ Uploaded xlsx: {xlsx_filename}")
                    else:
                        print(f"  ✗ Failed to upload xlsx: {xlsx_filename}")

        slice_result = {
            'slice': s,
            'slice_type': slice_type,
            'start_idx': slice_start,
            'end_idx': slice_end - 1,  # Store inclusive end
            'start_date': start_date,
            'end_date': end_date,
            'num_days': slice_end - slice_start,
            'fitness': metrics.get('fitness', 0.0),
            'win_rate': metrics.get('win_rate', 0.0),
            'quality_ratio': metrics.get('quality_ratio', 0.0),
            'expectancy': metrics.get('expectancy', 0.0),
            'roi': metrics.get('roi', 0.0),
            'num_trades': metrics.get('num_trades', 0),
            'consensus_stats': metrics.get('consensus_stats', {'note': 'Consensus applied per-step'}),
            'raw_pnl': metrics.get('raw_pnl', 0.0),
            'peak_capital_employed': metrics.get('peak_capital_employed', 0.0),
            'csv_filename': csv_filename if closed_trades else None,
        }

        results['committee_slices'].append(slice_result)

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

    # Aggregate consensus stats across all slices
    all_consensus_stats = [s['consensus_stats'] for s in results['committee_slices']]

    # Calculate averages only if we have valid tracking data
    if all(isinstance(cs, dict) and 'avg_consensus_votes' in cs for cs in all_consensus_stats):
        results['consensus_summary'] = {
            'avg_unanimity_pct': float(np.mean([cs['unanimity_pct'] for cs in all_consensus_stats])),
            'avg_min_consensus_pct': float(np.mean([cs['min_consensus_pct'] for cs in all_consensus_stats])),
            'avg_consensus_votes': float(np.mean([cs['avg_consensus_votes'] for cs in all_consensus_stats])),
            'total_trades_by_quorum': int(sum([cs['trades_by_quorum'] for cs in all_consensus_stats])),
            'total_trades_by_conviction': int(sum([cs['trades_by_conviction'] for cs in all_consensus_stats])),
            'total_trades_vetoed': int(sum([cs['trades_vetoed'] for cs in all_consensus_stats])),
        }
    else:
        results['consensus_summary'] = {
            'note': 'Consensus tracking was not enabled or no data available'
        }

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

        # Consensus stats
        cs = s['consensus_stats']
        if 'note' in cs:
            print(f"    Consensus: {cs['note']}")
        elif 'avg_consensus_votes' in cs:
            print(f"    Consensus:")
            print(f"      Unanimity: {cs['unanimity_pct']:.1f}%, "
                  f"Min Quorum: {cs['min_consensus_pct']:.1f}%, "
                  f"Avg Votes: {cs['avg_consensus_votes']:.2f}")
            print(f"      By Quorum: {cs['trades_by_quorum']}, "
                  f"By Conviction: {cs['trades_by_conviction']}, "
                  f"Vetoed: {cs['trades_vetoed']}")

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
    print("CONSENSUS SUMMARY")
    print("-"*60)
    cs_sum = results['consensus_summary']
    if 'note' in cs_sum:
        print(f"  {cs_sum['note']}")
    else:
        print(f"  Unanimity Rate: {cs_sum['avg_unanimity_pct']:.1f}%")
        print(f"  Min Consensus Rate: {cs_sum['avg_min_consensus_pct']:.1f}%")
        print(f"  Avg Votes per Trade: {cs_sum['avg_consensus_votes']:.2f}")
        print(f"  Trades by Quorum: {cs_sum['total_trades_by_quorum']}")
        print(f"  Trades by Conviction: {cs_sum['total_trades_by_conviction']}")
        print(f"  Trades Vetoed: {cs_sum['total_trades_vetoed']}")

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

    return results


def run_quorum_sweep(manager: CommitteeManager, loader, stats, holdout_info,
                     context_window_days: int, quorum_values: list) -> dict:
    """
    Run validation with multiple quorum values for A/B testing.

    Args:
        manager: CommitteeManager instance
        loader: StockDataLoader instance
        stats: Normalization stats dict
        holdout_info: Holdout period info dict
        context_window_days: Context window size
        quorum_values: List of quorum values to test (e.g., [2, 3, 4, 5])

    Returns:
        Dict mapping quorum value to validation results
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

        # Override quorum
        Config.COMMITTEE_QUORUM = quorum

        # Run validation
        results = run_validation(manager, loader, stats, holdout_info, context_window_days)

        if results:
            all_results[quorum] = {
                'aggregate': results['committee_aggregate'],
                'consensus': results['consensus_summary'],
            }

        # Clean up between runs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Restore original quorum
    Config.COMMITTEE_QUORUM = original_quorum

    # Print comparison summary
    print("\n" + "="*60)
    print("QUORUM SWEEP COMPARISON SUMMARY")
    print("="*60)

    # Header
    print(f"\n{'Quorum':<8} {'Fitness':<10} {'Win Rate':<10} {'Quality':<10} {'Expectancy':<12} {'ROI':<10} {'Trades':<8}")
    print("-" * 78)

    for quorum in sorted(all_results.keys()):
        agg = all_results[quorum]['aggregate']
        print(f"{quorum:<8} {agg['mean_fitness']:<10.2f} {agg['mean_win_rate']*100:<10.1f}% "
              f"{agg['mean_quality_ratio']:<10.3f} {agg['mean_expectancy']:<12.6f} "
              f"{agg['mean_roi']:<10.2f}% {agg['total_trades']:<8}")

    # Consensus breakdown
    print(f"\n{'Quorum':<8} {'By Quorum':<12} {'By Conviction':<14} {'Vetoed':<10} {'Unanimity':<12} {'Avg Votes':<10}")
    print("-" * 78)

    for quorum in sorted(all_results.keys()):
        cs = all_results[quorum]['consensus']
        if 'note' not in cs:
            print(f"{quorum:<8} {cs.get('total_trades_by_quorum', 0):<12} "
                  f"{cs.get('total_trades_by_conviction', 0):<14} "
                  f"{cs.get('total_trades_vetoed', 0):<10} "
                  f"{cs.get('avg_unanimity_pct', 0):<12.1f}% "
                  f"{cs.get('avg_consensus_votes', 0):<10.2f}")
        else:
            print(f"{quorum:<8} {cs['note']}")

    # Find best quorum by fitness
    best_quorum = max(all_results.keys(), key=lambda q: all_results[q]['aggregate']['mean_fitness'])
    best_fitness = all_results[best_quorum]['aggregate']['mean_fitness']

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: Quorum {best_quorum} achieved highest mean fitness ({best_fitness:.2f})")
    print(f"{'='*60}")

    return all_results


def run_conviction_sweep(manager: CommitteeManager, loader, stats, holdout_info,
                         context_window_days: int, percentile_values: list) -> dict:
    """
    Run validation with multiple conviction percentile values for A/B testing.

    Args:
        manager: CommitteeManager instance
        loader: StockDataLoader instance
        stats: Normalization stats dict
        holdout_info: Holdout period info dict
        context_window_days: Context window size
        percentile_values: List of percentile values to test (e.g., [90, 95, 99])

    Returns:
        Dict mapping percentile value to validation results
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

        # Recalculate conviction thresholds with this percentile
        print(f"  Recalculating thresholds for {len(roster['members'])} members...")
        members_with_new_thresholds = recalculate_conviction_thresholds(
            roster['members'], loader, stats, holdout_info,
            context_window_days, percentile
        )

        # Run validation with recalculated members
        results = run_validation(
            manager, loader, stats, holdout_info, context_window_days,
            members_override=members_with_new_thresholds
        )

        if results:
            all_results[percentile] = {
                'aggregate': results['committee_aggregate'],
                'consensus': results['consensus_summary'],
            }

        # Clean up between runs
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print comparison summary
    print("\n" + "="*60)
    print("CONVICTION SWEEP COMPARISON SUMMARY")
    print("="*60)

    # Header
    print(f"\n{'Percentile':<12} {'Fitness':<10} {'Win Rate':<10} {'Quality':<10} {'Expectancy':<12} {'ROI':<10} {'Trades':<8}")
    print("-" * 82)

    for percentile in sorted(all_results.keys()):
        agg = all_results[percentile]['aggregate']
        print(f"P{percentile:<11} {agg['mean_fitness']:<10.2f} {agg['mean_win_rate']*100:<10.1f}% "
              f"{agg['mean_quality_ratio']:<10.3f} {agg['mean_expectancy']:<12.6f} "
              f"{agg['mean_roi']:<10.2f}% {agg['total_trades']:<8}")

    # Consensus breakdown (conviction trades are the key metric here)
    print(f"\n{'Percentile':<12} {'By Quorum':<12} {'By Conviction':<14} {'Vetoed':<10} {'Unanimity':<12}")
    print("-" * 70)

    for percentile in sorted(all_results.keys()):
        cs = all_results[percentile]['consensus']
        if 'note' not in cs:
            print(f"P{percentile:<11} {cs.get('total_trades_by_quorum', 0):<12} "
                  f"{cs.get('total_trades_by_conviction', 0):<14} "
                  f"{cs.get('total_trades_vetoed', 0):<10} "
                  f"{cs.get('avg_unanimity_pct', 0):<12.1f}%")
        else:
            print(f"P{percentile:<11} {cs['note']}")

    # Find best percentile by fitness
    best_percentile = max(all_results.keys(), key=lambda p: all_results[p]['aggregate']['mean_fitness'])
    best_fitness = all_results[best_percentile]['aggregate']['mean_fitness']

    # Also show conviction trade counts for context
    best_conviction_trades = all_results[best_percentile]['consensus'].get('total_trades_by_conviction', 'N/A')

    print(f"\n{'='*60}")
    print(f"RECOMMENDATION: P{best_percentile} achieved highest mean fitness ({best_fitness:.2f})")
    print(f"  Conviction trades at P{best_percentile}: {best_conviction_trades}")
    print(f"{'='*60}")

    return all_results


# --- Phase 1: Draft Day ---

def run_draft(manager: CommitteeManager, loader, stats, holdout_info, deep: bool = False):
    """
    Phase 1: Select committee from Global50 using coefficient correlation optimization.

    IMPORTANT: Correlation is calculated on VALIDATION data (not holdout) to prevent
    data leakage. The holdout period remains unseen until Phase 2 validation.

    Args:
        manager: CommitteeManager instance
        loader: StockDataLoader instance
        stats: Normalization statistics
        holdout_info: Holdout period information
        deep: If True, use automatic deep refinement instead of interactive refinement
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

    # 2. Prepare VALIDATION data for correlation calculation (NOT holdout - prevents data leakage)
    val_tensor, valid_indices = get_validation_data(loader, stats, holdout_info)
    print(f"\n  Validation tensor shape: {val_tensor.shape}")
    print(f"  (Using validation period for correlation - holdout remains unseen)")

    # 3. Calculate coefficient correlations on validation data
    corr_matrix, coefficients, _ = calculate_coefficient_correlations(
        entries, val_tensor, context_window_days
    )

    # 4. Optimize committee selection
    result = optimize_committee(entries, corr_matrix)

    if result is None:
        print("❌ Optimization failed")
        return None

    # 4b. Correlation refinement pass
    if deep:
        refined_indices = automatic_correlation_refinement(
            result['committee_indices'], entries, corr_matrix
        )
    else:
        refined_indices = interactive_correlation_refinement(
            result['committee_indices'], entries, corr_matrix
        )

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

    # Calculate conviction thresholds for each committee member
    print(f"\nCalculating conviction thresholds for committee members...")
    members_with_stats = []
    for idx, e in enumerate(committee_members):
        original_idx = committee_indices[idx]
        agent_coeffs_1d = coefficients[original_idx]

        if agent_coeffs_1d is not None:
            # Reshape to [Days, Stocks]
            num_days = len(valid_indices)
            num_stocks = Config.NUM_INVESTABLE_STOCKS
            agent_coeffs_2d = agent_coeffs_1d.reshape(num_days, num_stocks)

            # Calculate conviction threshold vector
            conviction_threshold_vector = calculate_agent_stats_vectorized(agent_coeffs_2d)

            member_data = {
                'filename': f"{e['run_name']}_{e['agent_id']}.pth",
                'agent_id': e['agent_id'],
                'run_name': e['run_name'],
                'gauntlet_score': e['gauntlet_score'],
                'roi': e.get('roi', 0.0),
                'expectancy': e.get('expectancy', 0.0),
                'quality_ratio': e.get('quality_ratio', 0.0),
                'win_ratio': e.get('win_ratio', 0.0),
                'is_maverick': e.get('is_maverick', False),
                'stats': {
                    'conviction_threshold_vector': conviction_threshold_vector.tolist()
                }
            }
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

    # Save roster and sync to cloud
    manager.save_roster(roster_data, committee_corr)

    # Print committee
    print(f"\n{'='*60}")
    print("FINAL COMMITTEE")
    print(f"{'='*60}")

    for i, m in enumerate(roster_data['members']):
        maverick_tag = " [M]" if m.get('is_maverick', False) else ""
        print(f"  {i+1}. {m['run_name']}_{m['agent_id']}{maverick_tag}: "
              f"score={m['gauntlet_score']:.2f}, roi={m['roi']:.2f}%")

    print(f"\n  Aggregate Score: {final_score_sum:.2f}")
    print(f"  Objective Value: {final_obj:.2f}")
    print(f"  Avg Correlation: {final_avg_corr:.3f}")
    print(f"  Max Pair Correlation: {final_max_corr:.3f}")

    return roster_data


# --- Simulation ---

def parse_date_input(date_str: str) -> datetime:
    """
    Parse user input date string in DD-MM-YY format.

    Args:
        date_str: Date string like '29-07-22'

    Returns:
        datetime object

    Raises:
        ValueError if format is invalid
    """
    try:
        return datetime.strptime(date_str, "%d-%m-%y")
    except ValueError:
        raise ValueError(f"Invalid date format '{date_str}'. Expected DD-MM-YY (e.g., 29-07-22)")


def find_date_index(loader, target_date: datetime) -> int:
    """
    Find the index in loader.dates that matches or is closest to target_date.

    Args:
        loader: StockDataLoader with dates array
        target_date: Target datetime to find

    Returns:
        Index of matching or nearest date

    Raises:
        ValueError if date is out of range
    """
    # Convert loader dates to datetime objects for comparison
    loader_dates = []
    for d in loader.dates:
        if isinstance(d, str):
            loader_dates.append(datetime.strptime(d, "%d-%m-%y"))
        else:
            # numpy datetime64 or pandas Timestamp
            loader_dates.append(pd.Timestamp(d).to_pydatetime())

    # Check bounds
    if target_date < loader_dates[0]:
        raise ValueError(f"Date {target_date.strftime('%d-%m-%y')} is before dataset start ({loader_dates[0].strftime('%d-%m-%y')})")
    if target_date > loader_dates[-1]:
        raise ValueError(f"Date {target_date.strftime('%d-%m-%y')} is after dataset end ({loader_dates[-1].strftime('%d-%m-%y')})")

    # Find exact match or nearest
    for i, d in enumerate(loader_dates):
        if d >= target_date:
            return i

    return len(loader_dates) - 1


def simulate_committee_continuous(members: list, loader, stats,
                                   context_window_days: int,
                                   start_idx: int,
                                   trading_end_idx: int,
                                   settlement_end_idx: int) -> dict:
    """
    Run committee simulation continuously over an arbitrary time period.

    Unlike evaluate_committee_on_slice (which uses fixed TRADING_PERIOD_DAYS),
    this function allows continuous trading for any period length.

    Args:
        members: List of committee member dicts with agent metadata and stats
        loader: StockDataLoader instance
        stats: Normalization statistics
        context_window_days: Context window for loading agent files
        start_idx: First trading day index
        trading_end_idx: Last day to open new positions (user's "last trading day")
        settlement_end_idx: End of settlement period (exclusive)

    Returns:
        Dict with comprehensive metrics
    """
    from environment.trading_env import TradingEnvironment

    # Create committee agent with consensus tracking
    committee = CommitteeAgent(members, context_window_days, track_consensus=True)

    # Create trading environment
    # trading_end_idx controls when new positions stop being opened
    env = TradingEnvironment(
        data_array=loader.data_array,
        dates=loader.dates,
        normalization_stats=stats,
        start_idx=start_idx,
        end_idx=settlement_end_idx,
        trading_end_idx=trading_end_idx + 1,  # +1 because env uses exclusive end for trading
        data_array_full=loader.data_array_full,
        is_training=False,
        gauntlet_mode=True
    )

    # Run simulation
    obs, info = env.reset()
    terminated = False

    committee.reset_episode()

    while not terminated:
        held_ids = list(env.open_positions.keys())
        action = committee.predict_action(obs, held_stock_ids=held_ids)
        obs, reward, terminated, truncated, info = env.step(action)

    # Get results
    summary = env.get_episode_summary()
    consensus_stats = committee.get_consensus_summary()

    # Enrich closed trades
    closed_trades = summary.get('closed_trades', [])
    for trade in closed_trades:
        stock_id = trade.get('stock_id')
        if stock_id is not None and loader.column_names:
            actual_col_idx = Config.INVESTABLE_START_COL + stock_id
            if actual_col_idx < len(loader.column_names):
                trade['stock_name'] = loader.column_names[actual_col_idx]
            else:
                trade['stock_name'] = f"UNKNOWN_{stock_id}"
        else:
            trade['stock_name'] = f"UNKNOWN_{stock_id}"
        trade['exit_date'] = trade.get('day', '')

    # Cleanup
    committee.cleanup()
    del env
    torch.cuda.empty_cache()

    return {
        'num_trades': summary['num_trades'],
        'win_rate': summary['win_rate'],
        'quality_ratio': (summary['num_wins'] / summary['num_losses']) if summary['num_losses'] > 0 else float('inf'),
        'expectancy': summary['avg_reward_per_trade'],
        'roi': summary['roi'],
        'fitness': summary['total_reward'],
        'consensus_stats': consensus_stats,
        'closed_trades': closed_trades,
        'raw_pnl': summary.get('raw_pnl', 0.0),
        'peak_capital_employed': summary.get('peak_capital_employed', 0.0),
        'trading_days': trading_end_idx - start_idx + 1,
        'settlement_days': settlement_end_idx - trading_end_idx - 1,
    }


def run_simulation(manager: CommitteeManager, loader, stats, context_window_days: int):
    """
    Interactive simulation mode: Run committee over a user-specified date range.

    This replicates how the committee would actually be deployed:
    1. User specifies first and last TRADING day
    2. System automatically adds settlement period after last trading day
    3. Committee can open new positions continuously until last trading day
    4. Outputs results and generates .xlsx position tracking file

    Unlike validation mode (which uses fixed 125-day trading periods), simulation
    mode allows continuous trading for any period length.
    """
    print("\n" + "="*60)
    print("COMMITTEE SIMULATION MODE")
    print("="*60)

    # Load roster
    roster = manager.load_roster()
    if roster is None:
        print("❌ No committee roster found. Run --draft first.")
        return None

    members = roster['members']
    print(f"\n✓ Loaded committee with {len(members)} members")

    # Show available date range
    first_date = loader.dates[0]
    last_date = loader.dates[-1]

    # Convert to display format
    if isinstance(first_date, str):
        first_date_display = first_date
    else:
        first_date_display = pd.Timestamp(first_date).strftime('%d-%m-%y')

    if isinstance(last_date, str):
        last_date_display = last_date
    else:
        last_date_display = pd.Timestamp(last_date).strftime('%d-%m-%y')

    print(f"\n  Dataset date range: {first_date_display} to {last_date_display}")
    print(f"  Total days in dataset: {len(loader.dates)}")
    print(f"\n  Simulation structure:")
    print(f"    Context window: {Config.CONTEXT_WINDOW_DAYS} days (required before first trade)")
    print(f"    Settlement period: {Config.SETTLEMENT_PERIOD_DAYS} days (auto-added after last trading day)")
    print(f"\n  NOTE: Unlike validation mode, simulation allows CONTINUOUS trading")
    print(f"        for any period length (not limited to {Config.TRADING_PERIOD_DAYS}-day episodes).")

    # Get first trading day from user
    print(f"\n" + "-"*60)
    print("Enter simulation date range (format: DD-MM-YY)")
    print("-"*60)

    # Calculate earliest valid first trading day (need context window before it)
    earliest_first_idx = Config.CONTEXT_WINDOW_DAYS
    earliest_first_date = loader.dates[earliest_first_idx]
    if isinstance(earliest_first_date, str):
        earliest_display = earliest_first_date
    else:
        earliest_display = pd.Timestamp(earliest_first_date).strftime('%d-%m-%y')

    print(f"\n  Earliest valid first trading day: {earliest_display}")
    print(f"    (Requires {Config.CONTEXT_WINDOW_DAYS} days of context before this)")

    while True:
        first_day_input = input("\n  First trading day: ").strip()
        try:
            first_day_dt = parse_date_input(first_day_input)
            first_day_idx = find_date_index(loader, first_day_dt)

            # Validate we have enough context
            if first_day_idx < Config.CONTEXT_WINDOW_DAYS:
                print(f"  ❌ Need at least {Config.CONTEXT_WINDOW_DAYS} days of context before first trading day.")
                print(f"     Earliest valid: {earliest_display}")
                continue

            break
        except ValueError as e:
            print(f"  ❌ {e}")
            continue

    # Calculate latest valid last trading day (need room for settlement after)
    latest_last_idx = len(loader.dates) - Config.SETTLEMENT_PERIOD_DAYS - 1
    if latest_last_idx <= first_day_idx:
        print(f"\n❌ Not enough data after {first_day_input} for trading + settlement.")
        return None

    latest_last_date = loader.dates[latest_last_idx]
    if isinstance(latest_last_date, str):
        latest_last_display = latest_last_date
    else:
        latest_last_display = pd.Timestamp(latest_last_date).strftime('%d-%m-%y')

    print(f"\n  Latest valid last trading day: {latest_last_display}")
    print(f"    (Need {Config.SETTLEMENT_PERIOD_DAYS} days after for settlement)")

    while True:
        last_day_input = input("\n  Last trading day: ").strip()
        try:
            last_day_dt = parse_date_input(last_day_input)
            last_day_idx = find_date_index(loader, last_day_dt)

            # Validate it's after first day
            if last_day_idx <= first_day_idx:
                print(f"  ❌ Last trading day must be after first trading day.")
                continue

            # Validate we have room for settlement
            settlement_end_idx = last_day_idx + Config.SETTLEMENT_PERIOD_DAYS + 1
            if settlement_end_idx > len(loader.dates):
                print(f"  ❌ Not enough data for settlement period after {last_day_input}.")
                print(f"     Latest valid: {latest_last_display}")
                continue

            break
        except ValueError as e:
            print(f"  ❌ {e}")
            continue

    # Display simulation parameters
    actual_first_date = loader.dates[first_day_idx]
    actual_last_trading_date = loader.dates[last_day_idx]
    actual_settlement_end_date = loader.dates[settlement_end_idx - 1]

    if isinstance(actual_first_date, str):
        actual_first_display = actual_first_date
    else:
        actual_first_display = pd.Timestamp(actual_first_date).strftime('%d-%m-%y')

    if isinstance(actual_last_trading_date, str):
        actual_last_trading_display = actual_last_trading_date
    else:
        actual_last_trading_display = pd.Timestamp(actual_last_trading_date).strftime('%d-%m-%y')

    if isinstance(actual_settlement_end_date, str):
        actual_settlement_end_display = actual_settlement_end_date
    else:
        actual_settlement_end_display = pd.Timestamp(actual_settlement_end_date).strftime('%d-%m-%y')

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

    # Confirm before running
    confirm = input("\n  Proceed with simulation? [Y/n]: ").strip().lower()
    if confirm in ('n', 'no'):
        print("  Simulation cancelled.")
        return None

    # Run the simulation
    print(f"\n" + "="*60)
    print("RUNNING SIMULATION")
    print("="*60)

    # Use continuous simulation
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

    # Display consensus stats
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
        # Generate unique filename with simulation date range (trading period)
        csv_filename = f"simulation_{actual_first_display.replace('-', '')}_to_{actual_last_trading_display.replace('-', '')}.csv"
        csv_path = manager.local_committee_dir / csv_filename

        import csv
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

        # Generate .xlsx position tracking file using transform_trades_to_positions
        xlsx_path = csv_path.parent / f"{csv_path.stem}_positions.xlsx"

        print(f"\n  Generating position tracking Excel file...")
        try:
            from transform_trades_to_positions import transform_trades_to_positions

            # Parse simulation dates for the full range (trading + settlement)
            sim_start = datetime.strptime(actual_first_display, "%d-%m-%y")
            sim_end = datetime.strptime(actual_settlement_end_display, "%d-%m-%y")

            transform_trades_to_positions(
                str(csv_path), str(xlsx_path),
                start_date=sim_start, end_date=sim_end
            )
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


# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Committee Selection from Global50")
    parser.add_argument('--draft', action='store_true',
                        help='Run Phase 1: Draft committee from Global50')
    parser.add_argument('--draft-deep', action='store_true',
                        help='Run Phase 1 with automatic deep refinement (no manual swaps)')
    parser.add_argument('--validate', action='store_true',
                        help='Run Phase 2: Validate on holdout slices')
    parser.add_argument('--verify-only', action='store_true',
                        help='Only verify data split')
    parser.add_argument('--mirror', action='store_true',
                        help='Check cloud sync status, download missing files')
    parser.add_argument('--simulate', action='store_true',
                        help='Simulate committee deployment over a custom date range')
    parser.add_argument('--quorum', type=int, default=None,
                        help=f'Override quorum threshold (default: {Config.COMMITTEE_QUORUM})')
    parser.add_argument('--sweep-quorum', type=str, default=None,
                        help='Sweep multiple quorum values, comma-separated (e.g., "2,3,4,5")')
    parser.add_argument('--conviction-percentile', type=int, default=None,
                        help='Override conviction percentile threshold (default: 95)')
    parser.add_argument('--sweep-conviction', type=str, default=None,
                        help='Sweep multiple conviction percentiles, comma-separated (e.g., "90,95,99")')
    args = parser.parse_args()

    if not args.draft and not args.draft_deep and not args.validate and not args.verify_only and not args.mirror and not args.simulate and not args.sweep_quorum and not args.sweep_conviction:
        print("Usage: python committee.py [--draft] [--draft-deep] [--validate] [--verify-only] [--mirror] [--simulate]")
        print("\nOptions:")
        print("  --draft          Run Phase 1: Draft committee from Global50 (interactive refinement)")
        print("  --draft-deep     Run Phase 1 with automatic deep refinement (no manual swaps)")
        print("  --validate       Run Phase 2: Validate committee on holdout slices")
        print("  --verify-only    Verify data split without running")
        print("  --mirror         Check cloud sync status, download missing files")
        print("  --simulate       Simulate committee deployment over a custom date range")
        print("  --quorum N       Override quorum threshold for validation (default: 3)")
        print("  --sweep-quorum   Sweep multiple quorum values (e.g., '2,3,4,5')")
        print("  --conviction-percentile N  Override conviction percentile (default: 95)")
        print("  --sweep-conviction  Sweep multiple conviction percentiles (e.g., '90,95,99')")
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

    if args.draft or args.draft_deep:
        roster = run_draft(manager, loader, stats, holdout_info, deep=args.draft_deep)

        # Auto-run validation after draft if requested
        if roster and args.validate:
            gc.collect()
            torch.cuda.empty_cache()

    # Handle --sweep-quorum (runs validation multiple times with different quorums)
    if args.sweep_quorum:
        quorum_values = [int(q.strip()) for q in args.sweep_quorum.split(',')]
        run_quorum_sweep(manager, loader, stats, holdout_info, context_window_days, quorum_values)
        exit(0)

    # Handle --sweep-conviction (runs validation multiple times with different percentiles)
    if args.sweep_conviction:
        percentile_values = [int(p.strip()) for p in args.sweep_conviction.split(',')]
        run_conviction_sweep(manager, loader, stats, holdout_info, context_window_days, percentile_values)
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
            # Update roster with validation results (only if not using custom overrides)
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
