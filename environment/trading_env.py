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
                 is_training: bool = True,
                 consistency_mode: bool = False,
                 gauntlet_mode: bool = False,
                 maverick_mode: bool = False):
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
            consistency_mode: If True, applies loss magnification for consistency training (see Config.CONSISTENCY_LOSS_MULTIPLIER)
            gauntlet_mode: If True, uses soft penalty for zero trades (tactical no-trade is acceptable)
            maverick_mode: If True, applies aggressive reward function (FOMO/ROI-First):
                          - Lower hurdle rate (50% of normal)
                          - No forced exit penalty
        """
        super().__init__()

        self.data_array = data_array  # For observations (5 features)
        self.data_array_full = data_array_full if data_array_full is not None else data_array  # For rewards (9 features)
        self.dates = dates
        # Pre-cast normalization stats to float32 to avoid implicit double-precision
        # conversion overhead during vectorized math operations every step
        self.norm_stats = {
            'mean': normalization_stats['mean'].astype(np.float32, copy=False),
            'std': normalization_stats['std'].astype(np.float32, copy=False)
        }
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.is_training = is_training  # Flag to control observation noise
        self.consistency_mode = consistency_mode  # Flag to enable loss magnification
        self.gauntlet_mode = gauntlet_mode  # Flag for soft zero-trades penalty during gauntlet/stabilization
        self.maverick_mode = maverick_mode  # Flag for aggressive FOMO/ROI-First reward function

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

    def set_gauntlet_mode(self, gauntlet_mode: bool):
        """
        Set whether environment is in gauntlet/stabilization mode.

        Args:
            gauntlet_mode: If True, uses soft penalty for zero trades (tactical no-trade is acceptable)
        """
        self.gauntlet_mode = gauntlet_mode

    def set_consistency_mode(self, consistency_mode: bool):
        """
        Set whether environment uses consistency mode evaluation rules.

        Args:
            consistency_mode: If True, applies loss magnification (see Config.CONSISTENCY_LOSS_MULTIPLIER)
        """
        self.consistency_mode = consistency_mode

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

        # Reset raw P&L and investment tracking (for ROI calculation)
        # Uses floored coefficient (integer shares)
        self.raw_pnl = 0.0
        self.total_investment = 0.0  # Legacy: cumulative transaction volume

        # Peak Capital Employed tracking (for accurate ROI)
        # Tracks the maximum capital tied up at any point, not cumulative transaction volume
        self.current_capital_employed = 0.0  # Current value of all open positions
        self.peak_capital_employed = 0.0     # High water mark of capital usage

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
        Get current observation (raw context window).

        Returns RAW nominal data (actual prices, raw MACD, etc.) to the network.
        Instance Normalization is applied inside the FeatureExtractor, making
        the model mathematically scale-invariant.

        Note: norm_stats now contains identity transform (mean=0, std=1) for
        backward compatibility, so the "normalization" step below is effectively
        a pass-through.

        Applies Gaussian noise during training as a regularization technique
        to prevent overfitting. This encourages the agent to learn robust,
        generalized patterns rather than memorizing exact values.

        Returns:
            Array of shape [context_window_days, num_columns, 5_features]
            Contains raw nominal data (actual prices, raw indicators)
        """
        start_idx = self.current_idx - Config.CONTEXT_WINDOW_DAYS + 1
        end_idx = self.current_idx + 1

        # Extract window
        window = self.data_array[start_idx:end_idx, :, :]

        # Legacy normalization step (now identity transform: mean=0, std=1)
        # Instance Normalization is applied in FeatureExtractor instead
        observation = (window - self.norm_stats['mean']) / self.norm_stats['std']

        # Add MULTIPLICATIVE observation noise for regularization during training
        # CRITICAL: With raw data, features have vastly different scales:
        #   - Price: ~150.00
        #   - RSI: ~50.00
        #   - TRIX: ~0.05
        # Additive noise would destroy small-scale signals (TRIX, MACD).
        # Multiplicative noise applies relative perturbation (e.g., ±1%) to all features equally.
        if self.is_training:
            noise_pct = np.random.normal(0.0, Config.OBSERVATION_NOISE_STD, observation.shape)
            observation = observation * (1.0 + noise_pct)

        return observation.astype(np.float32)

    def get_batch_observations(self, start_day_idx: int, count: int) -> np.ndarray:
        """
        ZERO-COPY batch observation fetch using numpy stride tricks.

        Uses sliding_window_view to create 125 observations INSTANTLY
        without a Python loop. This is ~100x faster than looping.

        Args:
            start_day_idx: Starting day index (current_idx at episode start)
            count: Number of days/observations to fetch

        Returns:
            Batch of observations [count, context_window, num_cols, features]
        """
        from numpy.lib.stride_tricks import sliding_window_view

        # 1. Calculate exact data range needed
        first_row = start_day_idx - Config.CONTEXT_WINDOW_DAYS + 1
        last_row = start_day_idx + count  # exclusive

        # 2. Single slice of raw data [total_days_needed, cols, feats]
        data_slice = self.data_array[first_row:last_row]

        # 3. Create sliding windows (ZERO-COPY strided view)
        # Result shape: [count, cols, feats, context_window]
        windows = sliding_window_view(data_slice, window_shape=Config.CONTEXT_WINDOW_DAYS, axis=0)

        # 4. Transpose to [batch, context, cols, feats] and make contiguous
        # This is the ONLY memory copy - one big block instead of 125 small ones
        obs_batch = np.ascontiguousarray(windows.transpose(0, 3, 1, 2), dtype=np.float32)

        # 5. Vectorized normalization (in-place)
        obs_batch -= self.norm_stats['mean']
        obs_batch /= self.norm_stats['std']

        # 6. Vectorized noise (if training)
        if self.is_training:
            noise = np.random.normal(0.0, Config.OBSERVATION_NOISE_STD, obs_batch.shape).astype(np.float32)
            obs_batch *= (1.0 + noise)

        return obs_batch

    def _process_action(self, action: np.ndarray) -> float:
        """
        Process the action and open new positions for qualifying stocks.
        OPTIMIZED: Uses vectorized filtering to avoid looping over all 108 stocks.

        Args:
            action: Array [108, 2] with [coefficient, sale_target] per stock

        Returns:
            Reward from this action (always 0.0 for opening positions)
        """
        # Check if we're still in trading period
        if self.current_idx >= self.trading_end_idx:
            return 0.0  # Settlement period - skip logging for speed

        coefficients = action[:, 0]
        sale_targets = action[:, 1]

        # Track max coefficient (vectorized)
        max_coeff = float(np.max(coefficients))
        if max_coeff > self.max_coefficient_during_episode:
            self.max_coefficient_during_episode = max_coeff

        # VECTORIZED: Find only candidates above threshold (typically 0-5 stocks)
        # This eliminates looping over all 108 stocks every step
        candidate_indices = np.where(coefficients >= Config.COEFFICIENT_THRESHOLD)[0]

        if len(candidate_indices) == 0:
            return 0.0  # Fast exit - nothing to do

        # Get close prices for all investable stocks at once (vectorized)
        # data_array_full shape: [days, cols, features], close price is index 1
        all_close_prices = self.data_array_full[self.current_idx, Config.INVESTABLE_START_COL:Config.INVESTABLE_START_COL + Config.NUM_INVESTABLE_STOCKS, 1]

        positions_opened = 0
        current_date = self.dates[self.current_idx]

        # Only loop over candidates (sparse - typically 0-5 stocks)
        for stock_id in candidate_indices:
            # Skip if position already open
            if stock_id in self.open_positions:
                continue

            entry_price = all_close_prices[stock_id]

            # Skip invalid prices
            if np.isnan(entry_price) or entry_price <= 0:
                continue

            coefficient = coefficients[stock_id]
            sale_target = np.clip(sale_targets[stock_id], Config.MIN_SALE_TARGET, Config.MAX_SALE_TARGET)

            # Open position
            position = Position(
                stock_id=int(stock_id),
                entry_price=float(entry_price),
                coefficient=float(coefficient),
                sale_target_pct=float(sale_target),
                days_held=0,
                entry_day_idx=self.current_idx,
                entry_date=current_date
            )

            self.open_positions[stock_id] = position
            self.total_positions_opened += 1
            positions_opened += 1

            # Track capital employed
            shares = int(coefficient)
            self.current_capital_employed += entry_price * shares
            if self.current_capital_employed > self.peak_capital_employed:
                self.peak_capital_employed = self.current_capital_employed

            # Log action (can be disabled for max speed)
            self.episode_actions.append({
                'day': current_date,
                'action': 'open',
                'stock_id': int(stock_id),
                'entry_price': float(entry_price),
                'coefficient': float(coefficient),
                'sale_target_pct': float(sale_target),
                'sale_target_price': position.sale_target_price
            })

        return 0.0
    
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
                # 1. Calculate Gross Gain
                gain_pct = ((exit_price - position.entry_price) / position.entry_price) * 100.0

                # 2. Update Raw Stats (for reporting)
                shares = int(position.coefficient)
                self.raw_pnl += (exit_price - position.entry_price) * shares
                self.total_investment += position.entry_price * shares  # Legacy cumulative

                # Release capital when closing position (for peak capital tracking)
                cost = position.entry_price * shares
                self.current_capital_employed -= cost

                # 3. SNIPER LOGIC: Apply Hurdle FIRST
                # A trade making 0.5% when hurdle is 0.6% is a LOSS of -0.1%
                # Convert HURDLE_RATE from decimal (0.006) to percentage (0.6%)
                # MAVERICK MODE: Lower hurdle (50% of normal) to encourage more trading
                hurdle_rate = Config.HURDLE_RATE * 0.5 if self.maverick_mode else Config.HURDLE_RATE
                hurdle_pct = hurdle_rate * 100.0
                net_gain_pct = gain_pct - hurdle_pct

                # 4. Conviction Scaling (Keep convex surface)
                scaled_coefficient = position.coefficient ** Config.CONVICTION_SCALING_POWER

                # 5. Calculate Reward with Asymmetric Penalty
                if net_gain_pct >= 0:
                    # WIN: Linear reward
                    base_reward = scaled_coefficient * net_gain_pct
                    self.num_wins += 1
                else:
                    # LOSS: Apply magnification only in consistency mode
                    # Consistency mode: 1.5x magnification (focus on reducing drawdowns)
                    # Normal/Maverick mode: 1.0x (treat losses equally to gains)
                    loss_multiplier = Config.CONSISTENCY_LOSS_MULTIPLIER if self.consistency_mode else 1.0
                    base_reward = scaled_coefficient * net_gain_pct * loss_multiplier
                    self.num_losses += 1

                # 6. Forced Exit Penalty (Lack of decisiveness)
                # MAVERICK MODE: No forced exit penalty (encourages holding for bigger gains)
                if reason == 'max_holding_period' and not self.maverick_mode:
                    forced_exit_penalty = position.entry_price * position.coefficient * Config.FORCED_EXIT_PENALTY_PCT
                else:
                    forced_exit_penalty = 0.0

                # Note: Hurdle is already in net_gain_pct, so we don't subtract it again
                reward = base_reward - forced_exit_penalty

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
                    'coefficient': position.coefficient,
                    'base_reward': base_reward,
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
        # Use mode-specific penalty: soft in gauntlet, lighter in consistency, harsher in normal
        if self.num_trades == 0:
            if self.gauntlet_mode:
                # Soft penalty during stabilization/gauntlet (tactical no-trade is acceptable)
                zero_trades_penalty = Config.ZERO_TRADES_PENALTY_GAUNTLET
            elif self.consistency_mode:
                zero_trades_penalty = Config.ZERO_TRADES_PENALTY_CONSISTENCY
            else:
                zero_trades_penalty = Config.ZERO_TRADES_PENALTY_NORMAL
        else:
            zero_trades_penalty = 0.0

        # Extract closed trades from episode actions (before clearing)
        closed_trades = [
            action for action in self.episode_actions
            if action.get('action') == 'close'
        ]

        # Calculate market return for FOMO penalty (Maverick mode)
        # Uses the market benchmark column (e.g., S&P 500 proxy)
        market_return_pct = 0.0
        if self.maverick_mode:
            try:
                benchmark_col = Config.MAVERICK_MARKET_BENCHMARK_COL
                # Get close prices at start and end of episode (index 1 = close in full dataset)
                start_price = self.data_array_full[self.start_idx, benchmark_col, 1]
                end_price = self.data_array_full[self.current_idx - 1, benchmark_col, 1]
                if not np.isnan(start_price) and not np.isnan(end_price) and start_price > 0:
                    market_return_pct = ((end_price - start_price) / start_price) * 100.0
            except (IndexError, KeyError):
                market_return_pct = 0.0

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
            'raw_pnl': self.raw_pnl,  # Sum of (exit_price - entry_price) * int(coef)
            'total_investment': self.total_investment,  # Legacy: cumulative transaction volume
            'peak_capital_employed': self.peak_capital_employed,  # Max capital tied up at any point
            # ROI now uses peak capital employed (true capital efficiency)
            # This rewards high-turnover strategies that recycle the same capital multiple times
            'roi': (self.raw_pnl / self.peak_capital_employed * 100) if self.peak_capital_employed > 0 else 0.0,
            'market_return_pct': market_return_pct,  # Market benchmark return for FOMO calculation
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