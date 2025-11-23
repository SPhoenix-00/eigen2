"""
Committee Production Engine for Project Eigen 2
- Phase 1: Draft Day (Pairwise Decorrelation & Sharpe Selection)
- Phase 2: Calibration (Horizon-Matched Tuning & Bootstrapping)
- Phase 3: Inference (Placeholder for Production)

HOLDOUT CONFIGURATION:
The committee uses Config.COMMITTEE_HOLDOUT_DAYS to define a holdout period at the
END of your dataset. This data is reserved EXCLUSIVELY for committee validation and
must NEVER be used during agent training.

Example:
  - Full dataset: 5000 days
  - Config.COMMITTEE_HOLDOUT_DAYS = 252 (1 year)
  - Training data: Days 0-4747 (available for training)
  - Holdout data: Days 4748-4999 (NEVER seen by agents)

REQUIREMENTS FOR DATA LOADER:
  loader.data_array : np.ndarray
      Normalized feature data [Days, Stocks, Features] (5-feature set)

  loader.data_array_full : np.ndarray
      Full raw data [Days, Stocks, Features] (9-feature OHLCV set)

  loader.train_end_idx : int (OPTIONAL)
      Last index used for training. If not provided, computed as:
      (total_days - Config.COMMITTEE_HOLDOUT_DAYS - Config.MIN_HOLDING_PERIOD)

The script will automatically compute holdout indices and verify no overlap.

USAGE:
  python committee.py --verify-only   # Check data split only
  python committee.py --draft         # Phase 1: Select committee
  python committee.py --calibrate     # Phase 2: Tune thresholds
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
import gc
from itertools import combinations

# Project Imports
from utils.config import Config
from data.loader import StockDataLoader
from models.ddpg_agent import DDPGAgent

# --- Configuration ---
HOF_DIR = Path("/workspace/hall_of_fame")
COMMITTEE_DIR = Path("committee_results")
ROSTER_FILE = COMMITTEE_DIR / "committee_roster.json"
CALIBRATION_FILE = COMMITTEE_DIR / "calibration_results.csv"

# --- metrics ---
MIN_TRADES_FOR_SIGNIFICANCE = 50
MAX_PAIRWISE_CORRELATION = 0.70
BOOTSTRAP_ROUNDS = 1000
TRANSACTION_COST_BPS = 20  # Basis points per round trip (slippage + commissions)

# --- Helper Functions ---

def calculate_max_drawdown(cumulative_returns):
    """Calculates Maximum Drawdown from a cumulative return series."""
    if len(cumulative_returns) == 0: return 0.0
    peak = np.maximum.accumulate(cumulative_returns)
    # Avoid division by zero if peak starts at 0
    peak = np.where(peak == 0, 1e-9, peak)
    drawdown = (cumulative_returns - peak) / peak
    return np.min(drawdown)

def bootstrap_expectancy(gains, rounds=BOOTSTRAP_ROUNDS):
    """
    Resample trades to estimate 95% Confidence Interval of Expectancy.
    Returns: (lower_bound, mean, upper_bound)
    """
    if len(gains) < 10: return -1.0, 0.0, 1.0
    
    means = []
    # Separate wins and losses to preserve win-rate structure in resampling? 
    # No, simplistic resampling of the outcome vector is standard for expectancy.
    
    for _ in range(rounds):
        sample = np.random.choice(gains, size=len(gains), replace=True)
        
        wins = sample[sample > 0]
        losses = abs(sample[sample <= 0])
        
        win_rate = len(wins) / len(sample)
        loss_rate = 1.0 - win_rate
        avg_win = np.mean(wins) if len(wins) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        means.append(expectancy)
        
    return np.percentile(means, 2.5), np.mean(means), np.percentile(means, 97.5)

def verify_data_split(loader):
    """
    Verifies that the holdout period is properly configured and separate from training.
    Uses Config.COMMITTEE_HOLDOUT_DAYS to compute the holdout period.

    Returns: (is_valid, error_message, holdout_info_dict)
    """
    total_days = len(loader.data_array_full)

    # Check minimum dataset size
    min_required = Config.CONTEXT_WINDOW_DAYS + Config.COMMITTEE_HOLDOUT_DAYS + Config.MIN_HOLDING_PERIOD
    if total_days < min_required:
        return False, (
            f"Dataset too small: {total_days} days\n"
            f"  Need at least {min_required} days:\n"
            f"    - Context: {Config.CONTEXT_WINDOW_DAYS}\n"
            f"    - Holdout: {Config.COMMITTEE_HOLDOUT_DAYS}\n"
            f"    - Min holding: {Config.MIN_HOLDING_PERIOD}"
        ), None

    # Compute holdout indices (last N days)
    holdout_start = total_days - Config.COMMITTEE_HOLDOUT_DAYS
    holdout_end = total_days - 1

    # Determine training end
    # If loader has explicit train_end_idx, use it. Otherwise compute it.
    if hasattr(loader, 'train_end_idx') and loader.train_end_idx is not None:
        train_end = loader.train_end_idx
        print(f"Using loader.train_end_idx: {train_end}")
    else:
        # Default: everything before holdout is available for training
        # Subtract MIN_HOLDING_PERIOD to ensure we have data for forward returns
        train_end = holdout_start - 1 - Config.MIN_HOLDING_PERIOD
        print(f"Computed train_end_idx: {train_end} (holdout_start - MIN_HOLDING_PERIOD - 1)")

    train_start = 0
    train_size = train_end - train_start + 1
    holdout_size = Config.COMMITTEE_HOLDOUT_DAYS
    gap = holdout_start - train_end - 1

    # Verify no overlap
    if holdout_start <= train_end:
        return False, (
            f"Holdout overlaps with training!\n"
            f"  Training ends at: {train_end}\n"
            f"  Holdout starts at: {holdout_start}\n"
            f"  Gap: {gap} days (NEGATIVE - OVERLAP DETECTED!)"
        ), None

    # Verify minimum sizes
    if train_size < Config.CONTEXT_WINDOW_DAYS:
        return False, f"Training set too small: {train_size} days (need at least {Config.CONTEXT_WINDOW_DAYS})", None

    if holdout_size < 100:
        return False, f"Holdout too small: {holdout_size} days (recommend at least 100 for statistical power)", None

    # Package holdout info for later use
    holdout_info = {
        'holdout_start': holdout_start,
        'holdout_end': holdout_end,
        'train_start': train_start,
        'train_end': train_end,
        'gap': gap
    }

    print("✅ Data Split Verification PASSED")
    print(f"  Total Days:     {total_days:,}")
    print(f"  Training:       {train_size:,} days (indices {train_start:,} to {train_end:,})")
    print(f"  Gap:            {gap:,} days")
    print(f"  Holdout:        {holdout_size:,} days (indices {holdout_start:,} to {holdout_end:,})")
    print(f"  Train/Holdout:  {train_size/holdout_size:.1f}x ratio")
    print(f"\n  ⚠ CRITICAL: Ensure your agents were trained ONLY on data up to index {train_end}")

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
        checkpoint = torch.load(filepath, map_location=Config.DEVICE)
        agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        agent.actor.eval()
        # Delete critic/targets to save memory
        del agent.critic
        del agent.critic_target
        del agent.actor_target
        return agent
    except Exception as e:
        print(f"⚠ Corrupt checkpoint {filepath.name}: {e}")
        return None

def get_out_of_sample_data(loader, stats, holdout_info):
    """
    Get strictly Out-Of-Sample (Holdout) data based on Config.COMMITTEE_HOLDOUT_DAYS.
    Uses the holdout_info dict computed by verify_data_split().

    Args:
        loader: StockDataLoader instance
        stats: Normalization statistics
        holdout_info: Dict with holdout_start, holdout_end, train_end

    Returns:
        (inputs_tensor, valid_indices)
    """
    # Extract holdout boundaries
    holdout_start = holdout_info['holdout_start']
    holdout_end = holdout_info['holdout_end']
    train_end = holdout_info['train_end']

    # Data for Context (Features) - Uses 5-feature set
    input_data = loader.data_array  # [Days, Stocks, 5_feats]

    # Valid inference points in holdout period
    # Need to leave room for MIN_HOLDING_PERIOD forward returns
    valid_indices = range(holdout_start, holdout_end - Config.MIN_HOLDING_PERIOD + 1)

    inputs = []

    print(f"\nPreparing {len(valid_indices)} days of Holdout data...")
    print(f"  Holdout period: Indices {holdout_start} to {holdout_end}")
    print(f"  Training ended at: Index {train_end}")
    print(f"  Gap: {holdout_start - train_end - 1} days\n")

    for i in valid_indices:
        # Normalize context window deterministically
        window = input_data[i - Config.CONTEXT_WINDOW_DAYS : i]
        normalized = (window - stats['mean']) / stats['std']
        inputs.append(normalized)

    inputs = np.array(inputs)
    # Tensor: [Batch, Context, Stocks, Feats]
    inputs_tensor = torch.FloatTensor(inputs).to(Config.DEVICE)

    return inputs_tensor, valid_indices

# --- Phase 1: Draft Day ---

def run_draft(loader, stats, holdout_info):
    COMMITTEE_DIR.mkdir(exist_ok=True)
    print("\n" + "="*60)
    print("PHASE 1: DRAFT DAY (Robust Selection)")
    print("="*60)

    # 1. Scan Hall of Fame
    agent_files = sorted(list(HOF_DIR.glob("*.pth")))
    if not agent_files:
        print(f"❌ No agents found in {HOF_DIR}")
        return

    # 2. Prepare Data
    market_tensor, valid_indices = get_out_of_sample_data(loader, stats, holdout_info)
    
    # 3. Generate Horizon-Matched PnL Curves
    print(f"Auditing {len(agent_files)} agents on {Config.MIN_HOLDING_PERIOD}-day hold returns...")
    
    # Pre-calculate market returns for the hold period
    # We need (Close_t+20 - Close_t) / Close_t
    # Index 1 is Close price in standard OHLCV arrays
    close_idx = 1 
    full_closes = loader.data_array_full[:, :, close_idx]
    
    # Calculate N-day forward return for every day in valid_indices
    # shape: [Num_Test_Days, Num_Stocks]
    period_returns = []
    for t in valid_indices:
        entry_price = full_closes[t]
        # Enforce Holding Period Mismatch Fix (Point 2)
        exit_price = full_closes[t + Config.MIN_HOLDING_PERIOD]
        
        # Safety for NaNs
        ret = np.where(entry_price > 0, (exit_price - entry_price) / entry_price, 0.0)
        period_returns.append(ret)
    
    period_returns = np.array(period_returns) # [Days, Stocks]

    agent_curves = {} # {filename: daily_portfolio_pnl}
    agent_sharpes = {}

    for fpath in tqdm(agent_files, desc="Simulating"):
        agent = load_agent_actor_only(fpath, 0)
        if agent is None: continue
        
        with torch.no_grad():
            # Action: [Days, Stocks, 2]
            actions = agent.actor(market_tensor).cpu().numpy()
            
        # Calculate PnL
        coeffs = actions[:, :, 0]
        
        # Filter: Only trades that would actually trigger (Threshold > 1.0)
        # We use 1.0 here as the baseline definition of "Active" for correlation purposes
        active_pos = np.maximum(0, coeffs - 1.0) 
        
        # Cap leverage for simulation safety
        active_pos = np.minimum(active_pos, 2.0) 
        
        # Agent Daily PnL = Sum(Position * Period_Return) across stocks
        # This represents the PnL "realized" for trades initiated on day t
        daily_pnl = np.sum(active_pos * period_returns, axis=1)
        
        agent_curves[fpath.name] = daily_pnl
        
        # Calculate Sharpe (Annualized)
        if np.std(daily_pnl) > 1e-6:
            sharpe = np.mean(daily_pnl) / np.std(daily_pnl) * np.sqrt(252)
        else:
            sharpe = -999.0
            
        agent_sharpes[fpath.name] = sharpe
        
        del agent
        torch.cuda.empty_cache()

    # 4. Draft Logic: Greedy Pairwise Decorrelation
    print("\nDrafting Committee (Optimizing for Uncorrelated Sharpe)...")
    
    # Sort candidates by Sharpe Ratio (Risk-Adjusted Quality)
    sorted_candidates = sorted(agent_sharpes.keys(), key=lambda x: agent_sharpes[x], reverse=True)
    
    # Convert curves to DataFrame for easy correlation
    df_pnl = pd.DataFrame(agent_curves)
    
    # Pick Captain (Best Sharpe)
    captain = sorted_candidates[0]
    drafted = [captain]
    print(f"  1. {captain} (Sharpe: {agent_sharpes[captain]:.2f})")
    
    target_size = 9
    
    for candidate in sorted_candidates[1:]:
        if len(drafted) >= target_size: break
        
        # Pairwise Correlation Check (Point 5)
        is_uncorrelated = True
        max_pairwise = -1.0
        conflict_agent = None
        
        cand_curve = df_pnl[candidate]
        
        for member in drafted:
            member_curve = df_pnl[member]
            corr = np.corrcoef(cand_curve, member_curve)[0, 1]
            
            if corr > MAX_PAIRWISE_CORRELATION:
                is_uncorrelated = False
                conflict_agent = member
                max_pairwise = corr
                break # Fail fast
        
        if is_uncorrelated:
            drafted.append(candidate)
            print(f"  {len(drafted)}. {candidate} (Sharpe: {agent_sharpes[candidate]:.2f})")
        else:
            # Verbose reject (optional)
            pass 
            
    # 5. Save Roster
    roster_data = {
        'committee_size': len(drafted),
        'members': drafted,
        'test_period_start': int(valid_indices[0]),
        'test_period_end': int(valid_indices[-1]),
        'generated_at': str(datetime.now())
    }
    
    with open(ROSTER_FILE, 'w') as f:
        json.dump(roster_data, f, indent=4)
        
    print(f"\n✓ Roster saved to {ROSTER_FILE}")
    
    # Save Correlation Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_pnl[drafted].corr(), annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title(f"Committee Correlation (Max Allowed: {MAX_PAIRWISE_CORRELATION})")
    plt.tight_layout()
    plt.savefig(COMMITTEE_DIR / "committee_correlation.png")


# --- Phase 2: Calibration ---

def run_calibration(loader, stats, holdout_info):
    if not ROSTER_FILE.exists():
        print(f"❌ No roster file found. Run --draft first.")
        return

    print("\n" + "="*60)
    print("PHASE 2: CALIBRATION (Bootstrap & Risk Metrics)")
    print("="*60)

    with open(ROSTER_FILE, 'r') as f:
        roster = json.load(f)
    members = roster['members']

    # 1. Prepare Data
    market_tensor, valid_indices = get_out_of_sample_data(loader, stats, holdout_info)
    
    # 2. Pre-Calculate Votes
    print(f"Calculating votes for {len(members)} agents...")
    all_votes = [] # [Days, Agents, Stocks, 2]
    
    for fname in tqdm(members):
        fpath = HOF_DIR / fname
        agent = load_agent_actor_only(fpath, 0)
        with torch.no_grad():
            actions = agent.actor(market_tensor).cpu().numpy()
            all_votes.append(actions)
        del agent
        
    all_votes = np.array(all_votes).transpose(1, 0, 2, 3)
    
    # 3. Prepare Returns Data (20-Day Hold)
    print("Preparing return data...")
    close_idx = 1
    full_closes = loader.data_array_full[:, :, close_idx]
    
    # Matrix of trade outcomes: [Days, Stocks]
    # Entry at T, Exit at T+20
    trade_outcomes = np.zeros((len(valid_indices), Config.NUM_INVESTABLE_STOCKS))
    
    for i, day_idx in enumerate(valid_indices):
        entry = full_closes[day_idx]
        exit_p = full_closes[day_idx + Config.MIN_HOLDING_PERIOD]
        # % Return
        trade_outcomes[i] = np.where(entry > 0, (exit_p - entry)/entry * 100, 0.0)

    # 4. Grid Search
    thresholds = [0.8, 1.0, 1.2, 1.4, 1.6]
    quorums = range(2, len(members) + 1)
    
    results = []
    print(f"Testing configurations with Bootstrap Resampling...")
    
    for thresh in thresholds:
        for quorum in quorums:
            # Vectorized Trigger Logic
            coefs = all_votes[:, :, :, 0]
            # Vote: Coef > Threshold
            votes = coefs >= thresh
            # Quorum: Sum(Votes) >= Quorum
            triggers = np.sum(votes, axis=1) >= quorum
            
            num_trades = np.sum(triggers)
            
            # Statistical Significance Check (Point 4)
            if num_trades < MIN_TRADES_FOR_SIGNIFICANCE:
                continue
                
            # Get Outcomes
            actual_gains = trade_outcomes[triggers]

            # Deduct Transaction Costs (realistic expectancy)
            cost_per_trade = TRANSACTION_COST_BPS / 100.0  # Convert bps to %
            actual_gains = actual_gains - cost_per_trade

            # Metrics
            wins = actual_gains[actual_gains > 0]
            losses = abs(actual_gains[actual_gains <= 0])
            
            win_rate = len(wins) / len(actual_gains)
            loss_rate = 1.0 - win_rate
            avg_win = np.mean(wins) if len(wins) else 0
            avg_loss = np.mean(losses) if len(losses) else 0
            
            # Expectancy
            expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
            
            # Risk Metrics (Point 6)
            # Convert % to decimal for Sharpe/Sortino
            decimal_returns = actual_gains / 100.0
            
            # Annualize (assuming these are 20-day returns, approx 12 periods/year)
            # Stdev of a series of trades != Time series Stdev, but good proxy for trade consistency
            sharpe = np.mean(decimal_returns) / (np.std(decimal_returns) + 1e-9) * np.sqrt(12)
            
            downside = decimal_returns[decimal_returns < 0]
            if len(downside) > 0:
                sortino = np.mean(decimal_returns) / (np.std(downside) + 1e-9) * np.sqrt(12)
            else:
                sortino = 999.0  # All wins, infinite Sortino
            
            # Drawdown (Simulated equity curve)
            max_dd = calculate_max_drawdown(np.cumsum(decimal_returns))
            
            # Bootstrap Confidence (Point 4)
            ci_low, ci_mean, ci_high = bootstrap_expectancy(actual_gains)
            
            results.append({
                'threshold': thresh,
                'quorum': quorum,
                'trades': num_trades,
                'win_rate': win_rate,
                'expectancy': expectancy,
                'ci_lower': ci_low,
                'ci_upper': ci_high,
                'sharpe': sharpe,
                'sortino': sortino,
                'max_dd': max_dd
            })
            
    # 5. Report
    if not results:
        print("❌ No configurations met the minimum trade count.")
        return

    df = pd.DataFrame(results)
    df.to_csv(CALIBRATION_FILE, index=False)
    
    # Filter: Lower Bound of CI > 0 (95% confident it makes money)
    robust = df[df['ci_lower'] > 0].sort_values('expectancy', ascending=False)
    
    print(f"\nTOP ROBUST CONFIGURATIONS (CI Lower Bound > 0):")
    print(robust[['threshold', 'quorum', 'trades', 'expectancy', 'ci_lower', 'sharpe', 'max_dd']].head(10).to_string(index=False))
    print(f"\n✓ Results saved to {CALIBRATION_FILE}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--draft', action='store_true', help='Run Draft Day to select committee members')
    parser.add_argument('--calibrate', action='store_true', help='Run Calibration to tune thresholds')
    parser.add_argument('--verify-only', action='store_true', help='Only verify data split, do not run')
    args = parser.parse_args()

    if not args.draft and not args.calibrate and not args.verify_only:
        print("Usage: python committee.py [--draft] [--calibrate] [--verify-only]")
        print("\nOptions:")
        print("  --draft        Run Phase 1: Draft committee from Hall of Fame")
        print("  --calibrate    Run Phase 2: Calibrate thresholds and quorum")
        print("  --verify-only  Verify data split without running analysis")
        exit(0)

    print("Initializing Engine...")
    Config.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Config.MIN_HOLDING_PERIOD must be defined, default to 20 if not
    if not hasattr(Config, 'MIN_HOLDING_PERIOD'):
        Config.MIN_HOLDING_PERIOD = 20
        print(f"⚠ MIN_HOLDING_PERIOD not in Config, defaulting to {Config.MIN_HOLDING_PERIOD}")

    # Config.COMMITTEE_HOLDOUT_DAYS must be defined
    if not hasattr(Config, 'COMMITTEE_HOLDOUT_DAYS'):
        print(f"\n❌ ERROR: Config.COMMITTEE_HOLDOUT_DAYS is not defined!")
        print(f"   Add the following to utils/config.py:")
        print(f"")
        print(f"   # Committee Holdout")
        print(f"   COMMITTEE_HOLDOUT_DAYS = 252  # 1 year of holdout data")
        print(f"")
        exit(1)

    loader, stats = load_normalization_stats()

    # CRITICAL: Verify data split before proceeding
    print("\n" + "="*60)
    print("DATA INTEGRITY CHECK")
    print("="*60)
    is_valid, error_msg, holdout_info = verify_data_split(loader)

    if not is_valid:
        print(f"\n❌ Data split verification FAILED!")
        print(f"   Error: {error_msg}")
        print(f"\n   Holdout Configuration:")
        print(f"     Config.COMMITTEE_HOLDOUT_DAYS = {Config.COMMITTEE_HOLDOUT_DAYS}")
        print(f"\n   The script will automatically use the LAST {Config.COMMITTEE_HOLDOUT_DAYS} days")
        print(f"   of your dataset as the holdout period.")
        print(f"\n   IMPORTANT: Ensure your agents were NOT trained on this data!")
        exit(1)

    if args.verify_only:
        print("\n✓ Verification complete. Holdout period is properly configured.")
        exit(0)

    if args.draft:
        run_draft(loader, stats, holdout_info)

    if args.calibrate:
        gc.collect()
        torch.cuda.empty_cache()
        run_calibration(loader, stats, holdout_info)