"""
Test script to verify the new holding period mechanics:
- MIN_HOLDING_PERIOD = 20 days (cannot sell before day 20)
- LIQUIDATION_WINDOW = 10 days (can sell days 21-30)
- MAX_HOLDING_PERIOD = 30 days (forced liquidation at day 30)
"""

import numpy as np
from environment.trading_env import TradingEnvironment, Position
from data.loader import StockDataLoader
from utils.config import Config

def test_holding_period_mechanics():
    """Test the new holding period mechanics."""
    print("="*60)
    print("Testing New Holding Period Mechanics")
    print("="*60)
    print(f"MIN_HOLDING_PERIOD: {Config.MIN_HOLDING_PERIOD} days")
    print(f"LIQUIDATION_WINDOW: {Config.LIQUIDATION_WINDOW} days")
    print(f"MAX_HOLDING_PERIOD: {Config.MAX_HOLDING_PERIOD} days")
    print(f"SETTLEMENT_PERIOD_DAYS: {Config.SETTLEMENT_PERIOD_DAYS} days")
    print("="*60)

    # Load data
    print("\nLoading data...")
    loader = StockDataLoader()
    data_array, stats = loader.load_and_prepare()

    # Create environment
    start_idx = Config.CONTEXT_WINDOW_DAYS
    end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

    print(f"\nCreating environment:")
    print(f"  Start idx: {start_idx}")
    print(f"  End idx: {end_idx}")
    print(f"  Total days: {end_idx - start_idx}")

    env = TradingEnvironment(
        data_array=loader.data_array,
        dates=loader.dates,
        normalization_stats=stats,
        start_idx=start_idx,
        end_idx=end_idx,
        trading_end_idx=start_idx + Config.TRADING_PERIOD_DAYS,
        data_array_full=loader.data_array_full
    )

    # Reset environment
    obs, info = env.reset()

    print("\n" + "="*60)
    print("Test 1: Opening a position and tracking holding period")
    print("="*60)

    # Create an action that will open a position
    # Set high coefficient for first stock and reasonable target
    action = np.zeros((Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM))
    action[0, 0] = 5.0  # High coefficient for stock 0
    action[0, 1] = 20.0  # 20% target gain

    # Step 1: Open position
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"\nStep 1: Action taken")
    print(f"  Open positions: {info['open_positions']}")
    print(f"  Reward: {reward}")

    if info['open_positions'] > 0:
        print(f"  ✓ Position opened successfully")

        # Get the position
        position = list(env.open_positions.values())[0]
        print(f"  Stock ID: {position.stock_id}")
        print(f"  Entry price: ${position.entry_price:.2f}")
        print(f"  Target price: ${position.sale_target_price:.2f}")
        print(f"  Days held: {position.days_held}")

        # Simulate holding for multiple days
        print("\n" + "="*60)
        print("Test 2: Simulating holding period")
        print("="*60)

        days_to_simulate = 35  # Hold past the max holding period
        for day in range(days_to_simulate):
            # No new action (all zeros)
            no_action = np.zeros((Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM))

            obs, reward, terminated, truncated, info = env.step(no_action)

            # Check if position was closed
            if info['open_positions'] == 0 and day > 0:
                print(f"\n  Day {day + 2}: Position CLOSED")
                print(f"    Reward: {reward}")

                # Get last action (should be a close action)
                if env.episode_actions:
                    last_action = env.episode_actions[-1]
                    if last_action.get('action') == 'close':
                        print(f"    Exit reason: {last_action['reason']}")
                        print(f"    Days held: {last_action['days_held']}")
                        print(f"    Exit price: ${last_action['exit_price']:.2f}")
                        print(f"    Gain/Loss: {last_action['gain_pct']:.2f}%")

                        # Verify holding period constraints
                        if last_action['reason'] == 'max_holding_period':
                            if last_action['days_held'] == Config.MAX_HOLDING_PERIOD:
                                print(f"    ✓ PASSED: Position held for exactly {Config.MAX_HOLDING_PERIOD} days")
                            else:
                                print(f"    ✗ FAILED: Expected {Config.MAX_HOLDING_PERIOD} days, got {last_action['days_held']}")
                        elif last_action['reason'] == 'target_hit':
                            if last_action['days_held'] >= Config.MIN_HOLDING_PERIOD:
                                print(f"    ✓ PASSED: Target hit after minimum holding period ({last_action['days_held']} >= {Config.MIN_HOLDING_PERIOD})")
                            else:
                                print(f"    ✗ FAILED: Target hit before minimum holding period ({last_action['days_held']} < {Config.MIN_HOLDING_PERIOD})")
                break
            elif day + 2 <= Config.MIN_HOLDING_PERIOD:
                # Should still be holding
                if info['open_positions'] > 0:
                    position = list(env.open_positions.values())[0]
                    if (day + 2) % 5 == 0:  # Print every 5 days
                        print(f"  Day {day + 2}: Holding (days held: {position.days_held})")

        print("\n" + "="*60)
        print("Test Complete!")
        print("="*60)

        # Summary
        summary = env.get_episode_summary()
        print(f"\nEpisode Summary:")
        print(f"  Total trades: {summary['num_trades']}")
        print(f"  Wins: {summary['num_wins']}")
        print(f"  Losses: {summary['num_losses']}")
        print(f"  Win rate: {summary['win_rate']:.1%}")

        return True
    else:
        print("  ✗ Failed to open position")
        return False

if __name__ == "__main__":
    success = test_holding_period_mechanics()
    if success:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")
