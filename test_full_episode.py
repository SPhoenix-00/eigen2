"""
Test a full trading episode with positions closing and rewards
"""

import numpy as np
from data.loader import StockDataLoader
from environment.trading_env import TradingEnvironment
from utils.config import Config

print("Testing Full Trading Episode...\n")

# Load data
loader = StockDataLoader()
data_array, stats = loader.load_and_prepare()

# Create environment - run for 60 days (enough for positions to close)
start_idx = Config.CONTEXT_WINDOW_DAYS
end_idx = start_idx + 60

env = TradingEnvironment(
    data_array=data_array,
    dates=loader.dates,
    normalization_stats=stats,
    start_idx=start_idx,
    end_idx=end_idx
)

print(f"Environment: {loader.dates[start_idx]} to {loader.dates[end_idx-1]} ({end_idx-start_idx} days)\n")

# Reset environment
obs, info = env.reset()

# Run full episode with semi-random actions
step_count = 0
max_reward_step = (0, 0.0)
min_reward_step = (0, 0.0)

print("Running episode...")
while True:
    # Generate random action
    # Higher coefficients to ensure positions are taken
    action = np.random.rand(Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM)
    action[:, 0] = action[:, 0] * 10 + 1  # Coefficients in [1, 11]
    action[:, 1] = np.random.uniform(Config.MIN_SALE_TARGET, Config.MAX_SALE_TARGET, Config.NUM_INVESTABLE_STOCKS)
    
    obs, reward, terminated, truncated, info = env.step(action)
    step_count += 1
    
    # Track best and worst reward steps
    if reward > max_reward_step[1]:
        max_reward_step = (step_count, reward)
    if reward < min_reward_step[1]:
        min_reward_step = (step_count, reward)
    
    # Print when something interesting happens
    if reward != 0:
        print(f"  Step {step_count} ({info['day']}): Reward = {reward:+.4f}, "
              f"Open = {info['open_positions']}, Trades = {info['num_trades']}")
    
    if terminated or truncated:
        break

# Get episode summary
summary = env.get_episode_summary()

print("\n" + "="*60)
print("Episode Summary")
print("="*60)
print(f"Total steps: {summary['total_steps']}")
print(f"Actions taken: {summary['actions_taken']}")
print(f"Total trades (closed): {summary['num_trades']}")
print(f"  Wins: {summary['num_wins']}")
print(f"  Losses: {summary['num_losses']}")
print(f"  Win rate: {summary['win_rate']:.1%}")
print(f"\nRewards:")
print(f"  Total: {summary['total_reward']:.4f}")
print(f"  Avg per trade: {summary['avg_reward_per_trade']:.4f}")
print(f"  Best step: {max_reward_step[0]} with {max_reward_step[1]:+.4f}")
print(f"  Worst step: {min_reward_step[0]} with {min_reward_step[1]:+.4f}")

# Analyze action log
print("\n" + "="*60)
print("Action Analysis")
print("="*60)

opens = [a for a in env.episode_actions if a['action'] == 'open']
closes = [a for a in env.episode_actions if a['action'] == 'close']
blocks = [a for a in env.episode_actions if a['action'] == 'blocked']

print(f"Positions opened: {len(opens)}")
print(f"Positions closed: {len(closes)}")
print(f"Actions blocked: {len(blocks)}")

if blocks:
    block_reasons = {}
    for b in blocks:
        reason = b['reason']
        block_reasons[reason] = block_reasons.get(reason, 0) + 1
    print(f"\nBlock reasons:")
    for reason, count in block_reasons.items():
        print(f"  {reason}: {count}")

if closes:
    print(f"\nClosed position details:")
    close_reasons = {}
    gains = []
    losses = []
    
    for c in closes:
        reason = c['reason']
        close_reasons[reason] = close_reasons.get(reason, 0) + 1
        
        if c['gain_pct'] >= 0:
            gains.append(c['gain_pct'])
        else:
            losses.append(c['gain_pct'])
    
    print(f"  Close reasons:")
    for reason, count in close_reasons.items():
        print(f"    {reason}: {count}")
    
    if gains:
        print(f"  Winning trades: avg gain = {np.mean(gains):.2f}%, max = {np.max(gains):.2f}%")
    if losses:
        print(f"  Losing trades: avg loss = {np.mean(losses):.2f}%, max = {np.min(losses):.2f}%")

# Show a few sample closed trades
if closes:
    print(f"\nSample closed trades (first 5):")
    for i, c in enumerate(closes[:5]):
        print(f"\n  Trade {i+1}:")
        print(f"    Stock ID: {c['stock_id']}")
        entry_date = c.get('entry_date', 'N/A')
        print(f"    Entry: ${c['entry_price']:.2f} on {entry_date}")
        print(f"    Exit: ${c['exit_price']:.2f} on {c['day']} (after {c['days_held']} days)")
        print(f"    Gain: {c['gain_pct']:+.2f}%")
        print(f"    Reward: {c['reward']:+.4f}")
        print(f"    Reason: {c['reason']}")

print("\n" + "="*60)
print("✓ Full episode test complete!")
print("="*60)