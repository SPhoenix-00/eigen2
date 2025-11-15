"""
Trading Environment for Project Eigen 2
Gym-style environment for stock trading with ERL
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass

from utils.config import Config


@dataclass
class Position:
    """Represents an open position"""
    stock_id: int
    entry_price: float
    coefficient: float
    sale_target_pct: float
    days_held: int = 0
    entry_day_idx: int = 0
    entry_date: str = ""
    
    @property
    def sale_target_price(self) -> float:
        """Calculate absolute sale target price"""
        return self.entry_price * (1 + self.sale_target_pct / 100.0)


class TradingEnvironment(gym.Env):
    """
    Trading environment for stock market simulation.

    Observation: Normalized window of market data [context_days, num_columns, 5_features]
    Features: close, RSI, MACD_signal, TRIX, diff20DMA
    Action: [108, 2] array where each stock has [coefficient, sale_target_pct]
    Reward: Cumulative gains/losses from closed positions
    """

    def __init__(self,
                 data_array: np.ndarray,
                 dates: np.ndarray,
                 normalization_stats: dict,
                 start_idx: int,
                 end_idx: int,
                 trading_end_idx: int = None,
                 data_array_full: np.ndarray = None,
                 is_training: bool = True):
        """
        Initialize trading environment.

        Args:
            data_array: Reduced market data for observations [num_days, num_columns, 5_features]
            dates: Array of date strings
            normalization_stats: Dict with 'mean' and 'std' for normalization
            start_idx: Starting day index (must have context_window_days history)
            end_idx: Ending day index (exclusive) - includes settlement period
            trading_end_idx: Last day new positions can be opened. If None, equals end_idx
            data_array_full: Full market data for reward calculation [num_days, num_columns, 9_features]
                            If None, uses data_array (for backward compatibility)
            is_training: If True, applies observation noise for regularization
        """
        super().__init__()

        self.data_array = data_array  # For observations (5 features)
        self.data_array_full = data_array_full if data_array_full is not None else data_array  # For rewards (9 features)
        self.dates = dates
        self.norm_stats = normalization_stats
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.is_training = is_training  # Flag to control observation noise

        # Trading end is when model stops opening new positions
        # Settlement period allows existing positions to close
        self.trading_end_idx = trading_end_idx if trading_end_idx is not None else end_idx
        
        # Validate indices
        assert start_idx >= Config.CONTEXT_WINDOW_DAYS, \
            f"start_idx must be >= {Config.CONTEXT_WINDOW_DAYS} to have enough context"
        assert end_idx <= len(data_array), \
            f"end_idx must be <= {len(data_array)}"
        assert self.trading_end_idx <= end_idx, \
            f"trading_end_idx must be <= end_idx"
        
        # Current state
        self.current_idx = start_idx
        self.open_positions: Dict[int, Position] = {}  # stock_id -> Position
        self.cumulative_reward = 0.0
        self.episode_rewards: List[float] = []
        self.episode_actions: List[Dict] = []
        
        # Activity tracking
        self.days_with_positions = 0
        self.days_without_positions = 0
        self.total_positions_opened = 0
        
        # Statistics
        self.num_trades = 0
        self.num_wins = 0
        self.num_losses = 0

        # Track max coefficient during episode (for validation gradient)
        self.max_coefficient_during_episode = 0.0
        
        # Define action and observation spaces
        # Action: [108 stocks, 2 values (coefficient, sale_target)]
        self.action_space = spaces.Box(
            low=np.array([[0.0, Config.MIN_SALE_TARGET]] * Config.NUM_INVESTABLE_STOCKS),
            high=np.array([[np.inf, Config.MAX_SALE_TARGET]] * Config.NUM_INVESTABLE_STOCKS),
            shape=(Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM),
            dtype=np.float32
        )
        
        # Observation: [context_window_days, num_columns, 5_features]
        # Values can be any float (normalized), including nan
        obs_shape = (Config.CONTEXT_WINDOW_DAYS,
                    data_array.shape[1],  # num_columns
                    Config.FEATURES_PER_CELL)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_shape,
            dtype=np.float32
        )
    
    def set_training_mode(self, is_training: bool):
        """
        Set whether environment is in training mode.

        Args:
            is_training: If True, applies observation noise for regularization
        """
        self.is_training = is_training

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None,
             start_idx: Optional[int] = None, end_idx: Optional[int] = None,
             trading_end_idx: Optional[int] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to start of episode.

        Args:
            seed: Random seed
            options: Additional options
            start_idx: New starting day index (if provided, re-initializes episode window)
            end_idx: New ending day index (if provided)
            trading_end_idx: New trading end index (if provided)

        Returns:
            Tuple of (observation, info)
        """
        super().reset(seed=seed)

        # CRITICAL FIX: Update episode window if new indices provided
        # This allows reusing single environment object instead of creating new ones
        if start_idx is not None:
            self.start_idx = start_idx
        if end_idx is not None:
            self.end_idx = end_idx
        if trading_end_idx is not None:
            self.trading_end_idx = trading_end_idx

        # Reset to starting position
        self.current_idx = self.start_idx
        self.open_positions = {}
        self.cumulative_reward = 0.0
        self.episode_rewards = []
        self.episode_actions = []

        # Reset statistics
        self.num_trades = 0
        self.num_wins = 0
        self.num_losses = 0

        # Reset tracking variables
        self.total_positions_opened = 0
        self.days_with_positions = 0
        self.days_without_positions = 0

        # Reset max coefficient tracking
        self.max_coefficient_during_episode = 0.0

        # Get initial observation
        obs = self._get_observation()
        info = self._get_info()

        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step.
        
        Args:
            action: Array of shape [108, 2] with [coefficient, sale_target] per stock
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        assert action.shape == (Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM), \
            f"Action shape must be {(Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM)}, got {action.shape}"
        
        step_reward = 0.0
        
        # Track if we have any positions today
        had_positions_today = len(self.open_positions) > 0
        
        # 1. Check existing positions for exits
        step_reward += self._update_positions()
        
        # 2. Process new action (if any)
        position_opened = self._process_action(action)
        step_reward += position_opened['reward'] if isinstance(position_opened, dict) else 0.0
        
        # 3. Apply inaction penalty if no positions held
        has_positions_now = len(self.open_positions) > 0
        if not has_positions_now and not had_positions_today:
            step_reward -= Config.INACTION_PENALTY
            self.days_without_positions += 1
        else:
            self.days_with_positions += 1
        
        # 3. Update cumulative reward
        self.cumulative_reward += step_reward
        self.episode_rewards.append(step_reward)
        
        # 4. Move to next day
        self.current_idx += 1
        
        # 5. Check if episode is done
        terminated = self.current_idx >= self.end_idx
        truncated = False
        
        # 6. Get observation and info (if not terminated)
        if not terminated:
            obs = self._get_observation()
            info = self._get_info()
        else:
            # Episode ended, return final state and info from last valid index
            self.current_idx = self.end_idx - 1  # Go back to last valid index for info
            obs = self._get_observation()
            info = self._get_info()
        
        return obs, step_reward, terminated, truncated, info
    
    def _get_observation(self) -> np.ndarray:
        """
        Get current observation (normalized context window).

        Applies Gaussian noise during training as a regularization technique
        to prevent overfitting. This encourages the agent to learn robust,
        generalized patterns rather than memorizing exact values.

        Returns:
            Array of shape [context_window_days, num_columns, 5_features]
        """
        start_idx = self.current_idx - Config.CONTEXT_WINDOW_DAYS + 1
        end_idx = self.current_idx + 1

        # Extract window
        window = self.data_array[start_idx:end_idx, :, :]

        # Normalize
        normalized = (window - self.norm_stats['mean']) / self.norm_stats['std']

        # Add observation noise for regularization during training
        # This prevents the agent from overfitting to exact values and forces
        # it to learn more robust, "fuzzy" decision rules that generalize better
        if self.is_training:
            noise = np.random.normal(0.0, Config.OBSERVATION_NOISE_STD, normalized.shape)
            normalized = normalized + noise

        return normalized.astype(np.float32)
    
    def _process_action(self, action: np.ndarray) -> float:
        """
        Process the action and open new positions for all qualifying stocks.
        Only opens positions during trading period.

        NEW LOGIC: Evaluates ALL 108 stocks independently. Any stock with coefficient
        >= COEFFICIENT_THRESHOLD will trigger a "buy" action, allowing the agent to
        open multiple positions in a single day and learn true portfolio diversification.

        Args:
            action: Array [108, 2] with [coefficient, sale_target] per stock

        Returns:
            Reward from this action (always 0.0 for opening positions)
        """
        # Check if we're still in trading period
        if self.current_idx >= self.trading_end_idx:
            # Settlement period - no new positions allowed
            self.episode_actions.append({
                'day': self.dates[self.current_idx],
                'action': 'blocked',
                'reason': 'settlement_period'
            })
            return 0.0

        # Extract coefficients and sale targets
        coefficients = action[:, 0]
        sale_targets = action[:, 1]

        # Track max coefficient during episode (for validation gradient)
        # Update with the maximum coefficient from this action
        max_coeff_this_step = float(np.max(coefficients))
        self.max_coefficient_during_episode = max(self.max_coefficient_during_episode, max_coeff_this_step)

        # Track if any positions were opened this step
        positions_opened_this_step = 0

        # Loop through ALL 108 stocks and evaluate each independently
        for stock_id in range(Config.NUM_INVESTABLE_STOCKS):
            coefficient = coefficients[stock_id]
            sale_target = sale_targets[stock_id]

            # Check if this stock's coefficient meets the threshold
            if coefficient < Config.COEFFICIENT_THRESHOLD:
                continue  # Skip this stock, coefficient too low

            # Check if we already have a position in this stock
            if stock_id in self.open_positions:
                # Cannot open duplicate position - log and skip
                self.episode_actions.append({
                    'day': self.dates[self.current_idx],
                    'action': 'blocked',
                    'stock_id': stock_id,
                    'reason': 'position_already_open'
                })
                continue

            # Get stock data for this day
            # Stock ID in action space is relative to investable stocks (0-107)
            # Need to map to actual column index (10-117)
            actual_col_idx = Config.INVESTABLE_START_COL + stock_id
            stock_data = self.data_array_full[self.current_idx, actual_col_idx, :]

            # Check if stock data is valid (not all nan)
            if np.all(np.isnan(stock_data)):
                # Stock doesn't exist on this day - log and skip
                self.episode_actions.append({
                    'day': self.dates[self.current_idx],
                    'action': 'blocked',
                    'stock_id': stock_id,
                    'reason': 'stock_data_invalid'
                })
                continue

            # Extract close price (index 1 in full 9-feature dataset)
            # Full dataset: [Open, Close, High, Low, RSI, MACD, MACD_Signal, Trix, xDiffDMA]
            entry_price = stock_data[1]  # close price

            if np.isnan(entry_price) or entry_price <= 0:
                # Invalid entry price - log and skip
                self.episode_actions.append({
                    'day': self.dates[self.current_idx],
                    'action': 'blocked',
                    'stock_id': stock_id,
                    'reason': 'invalid_entry_price'
                })
                continue

            # Validate and clip sale target
            sale_target = np.clip(sale_target, Config.MIN_SALE_TARGET, Config.MAX_SALE_TARGET)

            # All checks passed - open position for this stock
            position = Position(
                stock_id=stock_id,
                entry_price=entry_price,
                coefficient=coefficient,
                sale_target_pct=sale_target,
                days_held=0,
                entry_day_idx=self.current_idx,
                entry_date=self.dates[self.current_idx]
            )

            self.open_positions[stock_id] = position
            self.total_positions_opened += 1
            positions_opened_this_step += 1

            self.episode_actions.append({
                'day': self.dates[self.current_idx],
                'action': 'open',
                'stock_id': stock_id,
                'entry_price': entry_price,
                'coefficient': coefficient,
                'sale_target_pct': sale_target,
                'sale_target_price': position.sale_target_price
            })

        # If no positions were opened, log that no action was taken
        if positions_opened_this_step == 0:
            self.episode_actions.append({
                'day': self.dates[self.current_idx],
                'action': 'none',
                'reason': 'no_stocks_above_threshold'
            })

        return 0.0  # No immediate reward for opening positions
    
    def _update_positions(self) -> float:
        """
        Update all open positions and close if necessary.

        NEW LOGIC:
        - Agent MUST hold for MIN_HOLDING_PERIOD (20 days) - no selling before day 20
        - After day 20, agent has LIQUIDATION_WINDOW (10 days) to exit (days 21-30)
        - At day 30 (MAX_HOLDING_PERIOD), position is forcibly liquidated

        Returns:
            Total reward from closed positions
        """
        total_reward = 0.0
        positions_to_close = []

        for stock_id, position in self.open_positions.items():
            # Increment days held
            position.days_held += 1

            # Get current stock data (use full 9-feature dataset for accurate prices)
            actual_col_idx = Config.INVESTABLE_START_COL + stock_id
            stock_data = self.data_array_full[self.current_idx, actual_col_idx, :]

            # Check if stock data is valid
            if np.all(np.isnan(stock_data)):
                # Stock delisted or data missing - force close at last known price
                exit_price = position.entry_price
                reason = 'stock_delisted'
                should_close = True
            else:
                # Extract high and close prices
                # Full dataset: [Open, Close, High, Low, RSI, MACD, MACD_Signal, Trix, xDiffDMA]
                day_high = stock_data[2]  # day high (index 2)
                close_price = stock_data[1]  # close price (index 1)

                # NEW LOGIC: Check holding period constraints
                # 1. Before MIN_HOLDING_PERIOD (days 1-20): Cannot sell, continue holding
                if position.days_held < Config.MIN_HOLDING_PERIOD:
                    should_close = False
                    exit_price = None
                    reason = None
                # 2. During liquidation window (days 21-29): Can sell if target hit
                elif position.days_held < Config.MAX_HOLDING_PERIOD:
                    # Check if sale target hit (using day high)
                    if not np.isnan(day_high) and day_high >= position.sale_target_price:
                        exit_price = position.sale_target_price
                        reason = 'target_hit'
                        should_close = True
                    else:
                        should_close = False
                        exit_price = None
                        reason = None
                # 3. At MAX_HOLDING_PERIOD (day 30): Force liquidation
                else:
                    exit_price = close_price if not np.isnan(close_price) else position.entry_price
                    reason = 'max_holding_period'
                    should_close = True
            
            if should_close:
                # Calculate gain/loss
                gain_pct = ((exit_price - position.entry_price) / position.entry_price) * 100.0

                # Calculate base reward (before penalties)
                if gain_pct >= 0:
                    base_reward = position.coefficient * gain_pct
                    self.num_wins += 1
                    loss_penalty = 0.0
                else:
                    base_reward = position.coefficient * gain_pct  # Negative
                    # Apply loss penalty multiplier
                    loss_penalty = (Config.LOSS_PENALTY_MULTIPLIER - 1.0) * position.coefficient * abs(gain_pct)
                    self.num_losses += 1

                # Apply forced exit penalty if exit was due to max_holding_period
                forced_exit_penalty = Config.FORCED_EXIT_PENALTY if reason == 'max_holding_period' else 0.0

                # Final reward = base_reward - loss_penalty - forced_exit_penalty
                reward = base_reward - loss_penalty - forced_exit_penalty

                total_reward += reward
                self.num_trades += 1

                # Log closure with penalty breakdown
                self.episode_actions.append({
                    'day': self.dates[self.current_idx],
                    'action': 'close',
                    'stock_id': stock_id,
                    'entry_date': position.entry_date,
                    'entry_price': position.entry_price,
                    'exit_price': exit_price,
                    'days_held': position.days_held,
                    'gain_pct': gain_pct,
                    'base_reward': base_reward,
                    'loss_penalty': loss_penalty,
                    'forced_exit_penalty': forced_exit_penalty,
                    'reward': reward,
                    'reason': reason
                })
                
                positions_to_close.append(stock_id)
        
        # Remove closed positions
        for stock_id in positions_to_close:
            del self.open_positions[stock_id]
        
        return total_reward
    
    def _get_info(self) -> dict:
        """Get info dictionary for current step."""
        return {
            'day': self.dates[self.current_idx],
            'day_idx': self.current_idx,
            'open_positions': len(self.open_positions),
            'cumulative_reward': self.cumulative_reward,
            'num_trades': self.num_trades,
            'num_wins': self.num_wins,
            'num_losses': self.num_losses,
            'win_rate': self.num_wins / self.num_trades if self.num_trades > 0 else 0.0
        }
    
    def get_episode_summary(self) -> dict:
        """
        Get summary statistics for completed episode.

        Returns:
            Dictionary with episode statistics
        """
        inaction_penalty_total = self.days_without_positions * Config.INACTION_PENALTY

        # Calculate zero trades penalty (will be applied by caller)
        zero_trades_penalty = Config.ZERO_TRADES_PENALTY if self.num_trades == 0 else 0.0

        # Extract closed trades from episode actions (before clearing)
        closed_trades = [
            action for action in self.episode_actions
            if action.get('action') == 'close'
        ]

        summary = {
            'total_reward': self.cumulative_reward,
            'num_trades': self.num_trades,
            'num_wins': self.num_wins,
            'num_losses': self.num_losses,
            'win_rate': self.num_wins / self.num_trades if self.num_trades > 0 else 0.0,
            'avg_reward_per_trade': self.cumulative_reward / self.num_trades if self.num_trades > 0 else 0.0,
            'total_steps': len(self.episode_rewards),
            'actions_taken': self.total_positions_opened,
            'days_with_positions': self.days_with_positions,
            'days_without_positions': self.days_without_positions,
            'inaction_penalty_applied': inaction_penalty_total,
            'zero_trades_penalty': zero_trades_penalty,  # Just report it, don't apply here
            'closed_trades': closed_trades,  # Include all closed trades for analysis
            'max_coefficient_during_episode': self.max_coefficient_during_episode,  # For validation gradient
        }

        # CRITICAL FIX: Clear episode history to prevent memory leak (~15-20GB per generation)
        # These lists accumulate 145 entries per episode × 16 agents × 13+ gens = massive leak
        self.episode_rewards.clear()
        self.episode_actions.clear()

        return summary


# Standalone test
if __name__ == "__main__":
    from data.loader import StockDataLoader
    
    print("Testing TradingEnvironment...\n")
    
    # Load data
    loader = StockDataLoader()
    data_array, stats = loader.load_and_prepare()

    # Create environment on training data
    # Start after context window, run for 252 days (1 year)
    start_idx = Config.CONTEXT_WINDOW_DAYS
    end_idx = start_idx + 252

    env = TradingEnvironment(
        data_array=data_array,
        dates=loader.dates,
        normalization_stats=stats,
        start_idx=start_idx,
        end_idx=end_idx,
        data_array_full=loader.data_array_full
    )
    
    print(f"Environment created:")
    print(f"  Start: {loader.dates[start_idx]}")
    print(f"  End: {loader.dates[end_idx-1]}")
    print(f"  Episodes length: {end_idx - start_idx} days")
    print(f"  Observation shape: {env.observation_space.shape}")
    print(f"  Action shape: {env.action_space.shape}")
    
    # Test random episode
    print("\n--- Testing Random Episode ---")
    obs, info = env.reset()
    print(f"Initial observation shape: {obs.shape}")
    print(f"Initial info: {info}")
    
    # Run 10 random steps
    for step in range(10):
        # Random action: random coefficients and sale targets
        action = np.random.rand(Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM)
        action[:, 0] = action[:, 0] * 5  # Scale coefficients to [0, 5]
        action[:, 1] = action[:, 1] * (Config.MAX_SALE_TARGET - Config.MIN_SALE_TARGET) + Config.MIN_SALE_TARGET
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        print(f"\nStep {step + 1}:")
        print(f"  Reward: {reward:.4f}")
        print(f"  Open positions: {info['open_positions']}")
        print(f"  Cumulative reward: {info['cumulative_reward']:.4f}")
        
        if terminated:
            print("  Episode terminated")
            break
    
    # Get episode summary
    summary = env.get_episode_summary()
    print("\n--- Episode Summary ---")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    print("\n✓ Trading environment test complete!")