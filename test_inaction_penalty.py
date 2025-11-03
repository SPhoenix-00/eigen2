"""
Test that inaction penalty is working correctly
"""

import numpy as np
from data.loader import StockDataLoader
from environment.trading_env import TradingEnvironment
from utils.config import Config

print("Testing Inaction Penalty Mechanism...\n")

# Load data
loader = StockDataLoader()
data_array, stats = loader.load_and_prepare()

# Create environment
start_idx = Config.CONTEXT_WINDOW_DAYS
end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

env = TradingEnvironment(
    data_array=data_array,
    dates=loader.dates,
    normalization_stats=stats,
    start_idx=start_idx,
    end_idx=end_idx,
    trading_end_idx=start_idx + Config.TRADING_PERIOD_DAYS
)

print(f"Episode length: {Config.TRADING_PERIOD_DAYS} trading days")
print(f"Inaction penalty: {Config.INACTION_PENALTY} per day")
print(f"Zero trades penalty: {Config.ZERO_TRADES_PENALTY}")
print(f"Expected penalty if NO trades: {-Config.INACTION_PENALTY * Config.TRADING_PERIOD_DAYS - Config.ZERO_TRADES_PENALTY}\n")

# Reset environment
obs, info = env.reset()

# Run episode with NO ACTIONS (all zeros)
cumulative_reward = 0.0
for step in range(Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS):
    # Zero action = do nothing
    action = np.zeros((Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM))
    
    obs, reward, terminated, truncated, info = env.step(action)
    cumulative_reward += reward
    
    if terminated:
        break

# Get final summary
summary = env.get_episode_summary()

print("="*60)
print("Results:")
print("="*60)
print(f"Total trades: {summary['num_trades']}")
print(f"Days without positions: {summary['days_without_positions']}")
print(f"Days with positions: {summary['days_with_positions']}")
print(f"Inaction penalty applied: {summary['inaction_penalty_applied']}")
print(f"Zero trades penalty: {summary['zero_trades_penalty']}")
print(f"Cumulative reward (from loop): {cumulative_reward:.2f}")
print(f"Environment cumulative_reward: {env.cumulative_reward:.2f}")
print(f"Summary total_reward: {summary['total_reward']:.2f}")

# Apply zero trades penalty manually (like trainer does)
final_fitness = summary['total_reward']
if summary['num_trades'] == 0:
    final_fitness -= Config.ZERO_TRADES_PENALTY
    print(f"After zero-trades penalty: {final_fitness:.2f}")

print("\n" + "="*60)
print("Validation:")
print("="*60)

expected_penalty = -(Config.INACTION_PENALTY * (Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS) + Config.ZERO_TRADES_PENALTY)

if abs(final_fitness - expected_penalty) < 1.0:
    print(f"✓ PASS: Final fitness ({final_fitness:.2f}) matches expected penalty ({expected_penalty:.2f})")
else:
    print(f"✗ FAIL: Final fitness ({final_fitness:.2f}) does NOT match expected penalty ({expected_penalty:.2f})")
    print(f"  Difference: {final_fitness - expected_penalty:.2f}")

if final_fitness < -10000:
    print(f"✓ PASS: Inactive agents have catastrophic negative fitness ({final_fitness:.2f})")
else:
    print(f"✗ FAIL: Penalty not strong enough ({final_fitness:.2f})")

print("\n✓ Test complete!")