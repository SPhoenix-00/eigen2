"""
Live Trading Engine: daily inference pipeline using the committee.

Loads committee agents, manages a persistent trading episode with locally stored
positions, accepts daily data updates, and outputs actionable buy/sell orders.

All internal day counts are in TRADING DAYS (one row in the data array = one
trading day). Weekends and holidays do not appear as rows.
"""

import json
import numpy as np
import torch
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

from utils.config import Config
from data.loader import StockDataLoader
from committee.agent import CommitteeAgent
from committee.manager import CommitteeManager
from inference.state import (
    TradingState, LivePosition, ClosedTrade, EpisodeStats, DEFAULT_STATE_PATH,
)

# Feature indices from full 9-feature array to reduced 5-feature array
# Full: [Open, Close, High, Low, RSI, MACD, MACD_Signal, Trix, xDiffDMA]
# Reduced: [Close, RSI, MACD_Signal, Trix, xDiffDMA] = indices [1, 4, 6, 7, 8]
REDUCED_FEATURE_INDICES = [1, 4, 6, 7, 8]


class LiveTradingEngine:
    """
    Orchestrates daily inference: load committee, manage positions, produce orders.

    Usage:
        engine = LiveTradingEngine()
        engine.initialize()
        engine.start_episode("2026-02-09")   # once

        # Each trading day:
        report = engine.run_daily("daily_update.json")
        print(report)
    """

    def __init__(self, context_window_days: int = None, state_path: str = None):
        """
        Args:
            context_window_days: Override context window (default: Config.CONTEXT_WINDOW_DAYS)
            state_path: Override path for trading_state.json
        """
        self.context_window_days = context_window_days or Config.CONTEXT_WINDOW_DAYS
        self.state_path = Path(state_path) if state_path else DEFAULT_STATE_PATH

        self.loader: Optional[StockDataLoader] = None
        self.committee: Optional[CommitteeAgent] = None
        self.state: Optional[TradingState] = None
        self.normalization_stats: Optional[dict] = None

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        """Load data, committee, and persisted state."""
        print("=" * 60)
        print("LIVE TRADING ENGINE — Initialization")
        print("=" * 60)

        # 1. Load data
        print("\n[1/3] Loading data...")
        self.loader = StockDataLoader()
        _, self.normalization_stats = self.loader.load_and_prepare()
        total_days = self.loader.data_array.shape[0]
        print(f"  Data loaded: {total_days} trading days")
        print(f"  Date range: {self.loader.dates[0]} to {self.loader.dates[-1]}")

        # 2. Load committee
        print("\n[2/3] Loading committee...")
        manager = CommitteeManager(self.context_window_days)
        roster = manager.load_roster()
        if roster is None:
            raise RuntimeError(
                f"No committee roster found for cw{self.context_window_days}. "
                f"Run: python committee.py --draft"
            )
        self.committee = CommitteeAgent(
            roster['members'], self.context_window_days, track_consensus=True
        )

        # 3. Load persisted state (if any)
        print(f"\n[3/3] Loading state from {self.state_path}...")
        self.state = TradingState.load(self.state_path)
        if self.state is not None:
            print(f"  Resumed episode: started {self.state.episode_start_date}, "
                  f"trading day {self.state.trading_day}, "
                  f"{len(self.state.open_positions)} open positions")
        else:
            print("  No existing state found (use --start to begin an episode)")

        print("\n" + "=" * 60)
        print("Initialization complete")
        print("=" * 60)

    # ------------------------------------------------------------------
    # Episode management
    # ------------------------------------------------------------------

    def start_episode(self, start_date: str = None):
        """
        Start a new trading episode.

        Args:
            start_date: Calendar date string to start from (e.g., "2026-02-09").
                        Defaults to the last available date in the data.
        """
        if self.loader is None:
            raise RuntimeError("Call initialize() first")

        if start_date is None:
            day_idx = len(self.loader.dates) - 1
            start_date = str(self.loader.dates[day_idx])
        else:
            day_idx = self._find_date_index(start_date)

        # Validate we have enough history for the context window
        if day_idx < Config.CONTEXT_WINDOW_DAYS - 1:
            raise ValueError(
                f"Not enough history for context window. Need at least "
                f"{Config.CONTEXT_WINDOW_DAYS} days before start date. "
                f"Start index {day_idx} is too early."
            )

        self.state = TradingState(
            episode_start_date=start_date,
            episode_start_idx=day_idx,
            current_day_idx=day_idx,
            trading_day=0,
            last_processed_date=start_date,
        )
        self.state.save(self.state_path)

        print(f"\nNew episode started:")
        print(f"  Start date: {start_date}")
        print(f"  Data index: {day_idx}")
        print(f"  Max trading days: {Config.TRADING_PERIOD_DAYS}")
        print(f"  State saved to: {self.state_path}")

    # ------------------------------------------------------------------
    # Daily data ingestion
    # ------------------------------------------------------------------

    def append_daily_data(self, daily_file: str):
        """
        Append one trading day's data from a JSON file.

        Expected JSON format:
        {
            "date": "2026-02-14",
            "columns": {
                "SPX_INDEX": [Open, Close, High, Low, RSI, MACD, MACD_Signal, Trix, xDiffDMA],
                "DFAC": [...],
                ...
            }
        }

        Args:
            daily_file: Path to JSON file with the day's data
        """
        if self.loader is None:
            raise RuntimeError("Call initialize() first")

        with open(daily_file, 'r') as f:
            daily = json.load(f)

        date = daily['date']
        columns_data = daily['columns']

        num_columns = self.loader.data_array.shape[1]

        # Build new rows: [1, num_columns, 9] for full, [1, num_columns, 5] for reduced
        new_row_full = np.full((1, num_columns, 9), np.nan, dtype=np.float32)
        new_row = np.full((1, num_columns, Config.FEATURES_PER_CELL), np.nan, dtype=np.float32)

        matched = 0
        for col_idx, col_name in enumerate(self.loader.column_names):
            if col_name in columns_data:
                features_9 = columns_data[col_name]
                if len(features_9) >= 9:
                    new_row_full[0, col_idx, :] = features_9[:9]
                    # Reduced: [Close, RSI, MACD_Signal, Trix, xDiffDMA]
                    new_row[0, col_idx, :] = [features_9[i] for i in REDUCED_FEATURE_INDICES]
                    matched += 1

        # Append to arrays
        self.loader.data_array_full = np.concatenate(
            [self.loader.data_array_full, new_row_full], axis=0
        )
        self.loader.data_array = np.concatenate(
            [self.loader.data_array, new_row], axis=0
        )
        self.loader.dates = np.append(self.loader.dates, date)

        print(f"  Appended data for {date}: {matched}/{num_columns} columns matched")

    # ------------------------------------------------------------------
    # Daily inference
    # ------------------------------------------------------------------

    def run_daily(self, daily_file: str = None) -> str:
        """
        Run one day of live inference.

        Steps:
            1. Append new data (if daily_file provided)
            2. Advance day index
            3. Update existing positions (check exits)
            4. Get committee prediction
            5. Process new buy signals
            6. Generate order report
            7. Save state

        Args:
            daily_file: Optional path to daily update JSON

        Returns:
            Formatted order report string
        """
        if self.loader is None or self.committee is None:
            raise RuntimeError("Call initialize() first")
        if self.state is None:
            raise RuntimeError("No active episode. Use --start to begin one.")

        # 1. Append new data if provided
        if daily_file:
            self.append_daily_data(daily_file)

        # 2. Advance to next trading day
        self.state.current_day_idx += 1
        self.state.trading_day += 1

        current_idx = self.state.current_day_idx
        total_days = len(self.loader.dates)

        if current_idx >= total_days:
            raise RuntimeError(
                f"Data index {current_idx} is beyond available data ({total_days} days). "
                f"Provide a daily update file to extend the data."
            )

        current_date = str(self.loader.dates[current_idx])
        self.state.last_processed_date = current_date

        in_trading_period = (
            self.state.trading_day <= Config.TRADING_PERIOD_DAYS
        )

        # 3. Update existing positions (check exits based on today's prices)
        closed_today, forced_sells_tomorrow = self._update_positions(current_idx, current_date)

        # 4. Get committee prediction (only during trading period)
        buy_orders = []
        if in_trading_period:
            observation = self._get_observation(current_idx)
            held_ids = self.state.get_open_stock_ids()
            action = self.committee.predict_action(observation, held_stock_ids=held_ids)

            # 5. Process new buy signals
            buy_orders = self._process_signals(action, current_idx, current_date)

        # 6. Generate report
        report = self._generate_report(
            current_date=current_date,
            closed_today=closed_today,
            forced_sells=forced_sells_tomorrow,
            buy_orders=buy_orders,
            in_trading_period=in_trading_period,
        )

        # 7. Save state
        self.state.save(self.state_path)

        return report

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------

    def _update_positions(self, current_idx: int, current_date: str
                          ) -> Tuple[List[ClosedTrade], List[LivePosition]]:
        """
        Update all open positions for the current trading day.

        Mirrors TradingEnvironment._update_positions() logic:
        - days_held < MIN_HOLDING_PERIOD: hold (cannot sell)
        - MIN_HOLDING_PERIOD <= days_held < MAX_HOLDING_PERIOD: sell if target hit
        - days_held >= MAX_HOLDING_PERIOD: forced liquidation

        Returns:
            (closed_today, forced_sells_tomorrow):
                closed_today: trades that closed today (target hit)
                forced_sells_tomorrow: positions that reach max holding today,
                    to be sold at next market open (we record them at today's close)
        """
        closed_today = []
        forced_sells_tomorrow = []

        positions_snapshot = list(self.state.open_positions)

        for position in positions_snapshot:
            position.days_held += 1

            # Get today's price data from full 9-feature array
            actual_col_idx = Config.INVESTABLE_START_COL + position.stock_id
            stock_data = self.loader.data_array_full[current_idx, actual_col_idx, :]

            # Handle missing data
            if np.all(np.isnan(stock_data)):
                trade = self.state.close_position(
                    position.stock_id, position.entry_price, current_date, 'stock_delisted'
                )
                if trade:
                    closed_today.append(trade)
                continue

            # Full dataset: [Open, Close, High, Low, RSI, MACD, MACD_Signal, Trix, xDiffDMA]
            day_high = stock_data[2]
            close_price = stock_data[1]

            # Check holding period rules (all in trading days)
            if position.days_held < Config.MIN_HOLDING_PERIOD:
                # Cannot sell yet — continue holding
                continue
            elif position.days_held < Config.MAX_HOLDING_PERIOD:
                # Liquidation window: can sell if target hit
                if not np.isnan(day_high) and day_high >= position.sale_target_price:
                    trade = self.state.close_position(
                        position.stock_id, position.sale_target_price,
                        current_date, 'target_hit'
                    )
                    if trade:
                        closed_today.append(trade)
            else:
                # Max holding period reached — forced liquidation
                exit_price = close_price if not np.isnan(close_price) else position.entry_price
                trade = self.state.close_position(
                    position.stock_id, exit_price, current_date, 'forced_exit'
                )
                if trade:
                    forced_sells_tomorrow.append(trade)

        return closed_today, forced_sells_tomorrow

    def _get_observation(self, current_idx: int) -> np.ndarray:
        """
        Build observation window for the committee.

        Replicates TradingEnvironment._get_observation() without noise.

        Returns:
            Array of shape [CONTEXT_WINDOW_DAYS, num_columns, FEATURES_PER_CELL]
        """
        start_idx = current_idx - Config.CONTEXT_WINDOW_DAYS + 1
        end_idx = current_idx + 1

        window = self.loader.data_array[start_idx:end_idx, :, :]

        # Apply normalization (identity transform: mean=0, std=1)
        observation = (window - self.normalization_stats['mean']) / self.normalization_stats['std']

        # No noise during inference
        return observation.astype(np.float32)

    def _process_signals(self, action: np.ndarray, current_idx: int,
                         current_date: str) -> List[dict]:
        """
        Process committee action to identify new buy signals.

        Args:
            action: [NUM_INVESTABLE_STOCKS, 2] with [coefficient, sale_target] per stock
            current_idx: Current data array index
            current_date: Calendar date string

        Returns:
            List of buy order dicts
        """
        coefficients = action[:, 0]
        sale_targets = action[:, 1]

        buy_orders = []
        held_ids = set(self.state.get_open_stock_ids())

        # Get close prices for entry price estimation
        all_close_prices = self.loader.data_array_full[
            current_idx,
            Config.INVESTABLE_START_COL:Config.INVESTABLE_START_COL + Config.NUM_INVESTABLE_STOCKS,
            1  # Close price index
        ]

        # Find candidates above threshold
        candidate_indices = np.where(coefficients >= Config.COEFFICIENT_THRESHOLD)[0]

        for stock_id in candidate_indices:
            stock_id = int(stock_id)

            # Skip if already holding
            if stock_id in held_ids:
                continue

            entry_price = float(all_close_prices[stock_id])

            # Skip invalid prices
            if np.isnan(entry_price) or entry_price <= 0:
                continue

            coefficient = float(coefficients[stock_id])
            sale_target_pct = float(np.clip(
                sale_targets[stock_id], Config.MIN_SALE_TARGET, Config.MAX_SALE_TARGET
            ))
            sale_target_price = entry_price * (1 + sale_target_pct / 100.0)

            stock_name = self._stock_name(stock_id)

            # Create position
            position = LivePosition(
                stock_id=stock_id,
                stock_name=stock_name,
                entry_price=entry_price,
                coefficient=coefficient,
                sale_target_pct=sale_target_pct,
                sale_target_price=sale_target_price,
                days_held=0,
                entry_date=current_date,
                entry_day_idx=current_idx,
            )

            self.state.open_positions.append(position)

            # Update capital tracking
            shares = int(coefficient)
            self.state.stats.current_capital_employed += entry_price * shares
            if self.state.stats.current_capital_employed > self.state.stats.peak_capital_employed:
                self.state.stats.peak_capital_employed = self.state.stats.current_capital_employed

            buy_orders.append({
                'stock_id': stock_id,
                'stock_name': stock_name,
                'shares': shares,
                'entry_price': entry_price,
                'coefficient': coefficient,
                'sale_target_pct': sale_target_pct,
                'sale_target_price': sale_target_price,
            })

        return buy_orders

    # ------------------------------------------------------------------
    # Report generation
    # ------------------------------------------------------------------

    def _generate_report(self, current_date: str,
                         closed_today: List[ClosedTrade],
                         forced_sells: List[ClosedTrade],
                         buy_orders: List[dict],
                         in_trading_period: bool) -> str:
        """Generate the human-readable daily trading report."""
        lines = []
        sep = "=" * 60

        # Parse current date for day-of-week and next market open
        date_obj = self._parse_date(current_date)
        day_name = date_obj.strftime("%a") if date_obj else ""
        next_open = self._estimate_next_market_open(date_obj) if date_obj else None
        next_open_str = next_open.strftime("%a %Y-%m-%d") if next_open else "next trading day"

        period_label = "Trading Period Active" if in_trading_period else "Settlement Period"

        lines.append(sep)
        lines.append(f"EIGEN 2 DAILY TRADING REPORT — {day_name} {current_date}")
        lines.append(f"Trading Day {self.state.trading_day} of {Config.TRADING_PERIOD_DAYS} | {period_label}")
        lines.append(sep)

        # --- Positions closed today (limit orders that filled) ---
        if closed_today:
            lines.append("")
            lines.append("POSITIONS CLOSED TODAY (limit orders filled):")
            for t in closed_today:
                weeks = self._trading_days_to_weeks(t.days_held)
                sign = "+" if t.gain_pct >= 0 else ""
                lines.append(
                    f"  {t.stock_name}: SOLD {int(t.coefficient)} shares | "
                    f"{sign}{t.gain_pct:.1f}% (${t.entry_price:.2f} -> ${t.exit_price:.2f})"
                )
                lines.append(
                    f"        Held {t.days_held} trading days (~{weeks:.1f} wks) | "
                    f"Reason: {t.reason}"
                )

        # --- Orders for next market open ---
        has_sells = len(forced_sells) > 0
        has_buys = len(buy_orders) > 0

        if has_sells or has_buys:
            lines.append("")
            lines.append(f"ORDERS FOR NEXT MARKET OPEN ({next_open_str}):")

            if has_sells:
                lines.append("  SELL (forced liquidation — max holding period reached):")
                for t in forced_sells:
                    weeks = self._trading_days_to_weeks(t.days_held)
                    sign = "+" if t.gain_pct >= 0 else ""
                    lines.append(
                        f"    {t.stock_name}: Sell {int(t.coefficient)} shares | "
                        f"Entry ${t.entry_price:.2f} | {sign}{t.gain_pct:.1f}% | "
                        f"Held {t.days_held} td (~{weeks:.1f} wks)"
                    )

            if has_buys:
                lines.append("  BUY (committee consensus):")
                for order in buy_orders:
                    lines.append(
                        f"    {order['stock_name']}: Buy {order['shares']} shares "
                        f"at ~${order['entry_price']:.2f} | "
                        f"Target: ${order['sale_target_price']:.2f} "
                        f"(+{order['sale_target_pct']:.1f}%) | "
                        f"Coeff: {order['coefficient']:.1f}"
                    )
        else:
            if not closed_today:
                lines.append("")
                lines.append("No orders for next market open.")

        # --- Current holdings ---
        if self.state.open_positions:
            lines.append("")
            lines.append("CURRENT HOLDINGS (after today's closes):")

            current_idx = self.state.current_day_idx
            for pos in sorted(self.state.open_positions, key=lambda p: p.days_held, reverse=True):
                # Get current price for unrealized P&L
                actual_col = Config.INVESTABLE_START_COL + pos.stock_id
                current_close = self.loader.data_array_full[current_idx, actual_col, 1]

                if not np.isnan(current_close):
                    unrealized_pct = ((current_close - pos.entry_price) / pos.entry_price) * 100.0
                    sign = "+" if unrealized_pct >= 0 else ""
                    price_str = f"Now ~${current_close:.2f} ({sign}{unrealized_pct:.1f}%)"
                else:
                    price_str = "Price N/A"

                remaining_td = Config.MAX_HOLDING_PERIOD - pos.days_held
                remaining_wks = self._trading_days_to_weeks(max(0, remaining_td))

                lines.append(
                    f"  {pos.stock_name:>5}: {int(pos.coefficient)} shares | "
                    f"Day {pos.days_held:>2}/{Config.MAX_HOLDING_PERIOD} td | "
                    f"Entry ${pos.entry_price:.2f} | "
                    f"Target ${pos.sale_target_price:.2f} (+{pos.sale_target_pct:.1f}%)"
                )
                lines.append(
                    f"         {price_str} | "
                    f"~{remaining_wks:.1f} wks remaining"
                )
        else:
            lines.append("")
            lines.append("CURRENT HOLDINGS: None")

        # --- Summary ---
        stats = self.state.stats
        win_rate = (stats.wins / stats.total_trades * 100.0) if stats.total_trades > 0 else 0.0
        sign = "+" if stats.raw_pnl >= 0 else ""

        lines.append("")
        lines.append(
            f"SUMMARY: {len(self.state.open_positions)} open | "
            f"{stats.total_trades} trades | "
            f"{stats.wins}W/{stats.losses}L ({win_rate:.1f}%) | "
            f"P&L: {sign}${stats.raw_pnl:.2f}"
        )
        if stats.peak_capital_employed > 0:
            roi = (stats.raw_pnl / stats.peak_capital_employed) * 100.0
            lines.append(
                f"         Peak capital: ${stats.peak_capital_employed:.2f} | "
                f"ROI: {sign}{roi:.1f}%"
            )
        lines.append(sep)
        lines.append("")
        lines.append("td = trading days (excludes weekends/holidays)")
        lines.append("Next market open is estimated (verify against exchange holiday calendar)")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Status and history (read-only)
    # ------------------------------------------------------------------

    def get_status(self) -> str:
        """Return current portfolio status without running inference."""
        if self.state is None:
            return "No active trading episode. Use --start to begin one."

        lines = []
        sep = "=" * 60
        lines.append(sep)
        lines.append("EIGEN 2 PORTFOLIO STATUS")
        lines.append(sep)
        lines.append(f"  Episode started: {self.state.episode_start_date}")
        lines.append(f"  Last processed:  {self.state.last_processed_date}")
        lines.append(f"  Trading day:     {self.state.trading_day} of {Config.TRADING_PERIOD_DAYS}")

        period = "Trading" if self.state.trading_day <= Config.TRADING_PERIOD_DAYS else "Settlement"
        lines.append(f"  Period:          {period}")

        lines.append(f"\n  Open positions:  {len(self.state.open_positions)}")
        for pos in sorted(self.state.open_positions, key=lambda p: p.days_held, reverse=True):
            remaining_td = Config.MAX_HOLDING_PERIOD - pos.days_held
            lines.append(
                f"    {pos.stock_name:>5}: {int(pos.coefficient)} shares | "
                f"Day {pos.days_held:>2}/{Config.MAX_HOLDING_PERIOD} td | "
                f"Entry ${pos.entry_price:.2f} | "
                f"Target ${pos.sale_target_price:.2f} (+{pos.sale_target_pct:.1f}%) | "
                f"{remaining_td} td remaining"
            )

        stats = self.state.stats
        win_rate = (stats.wins / stats.total_trades * 100.0) if stats.total_trades > 0 else 0.0
        sign = "+" if stats.raw_pnl >= 0 else ""

        lines.append(f"\n  Closed trades:   {stats.total_trades}")
        lines.append(f"  Win/Loss:        {stats.wins}W / {stats.losses}L ({win_rate:.1f}%)")
        lines.append(f"  Raw P&L:         {sign}${stats.raw_pnl:.2f}")
        lines.append(f"  Peak capital:    ${stats.peak_capital_employed:.2f}")
        lines.append(f"  Current capital: ${stats.current_capital_employed:.2f}")

        if stats.peak_capital_employed > 0:
            roi = (stats.raw_pnl / stats.peak_capital_employed) * 100.0
            lines.append(f"  ROI:             {sign}{roi:.1f}%")

        lines.append(sep)
        return "\n".join(lines)

    def get_history(self) -> str:
        """Return closed trade history."""
        if self.state is None:
            return "No active trading episode. Use --start to begin one."

        if not self.state.closed_trades:
            return "No closed trades yet."

        lines = []
        sep = "=" * 60
        lines.append(sep)
        lines.append("EIGEN 2 TRADE HISTORY")
        lines.append(sep)

        lines.append(
            f"\n{'#':>3}  {'Stock':>5}  {'Entry Date':>10}  {'Exit Date':>10}  "
            f"{'Entry$':>8}  {'Exit$':>8}  {'Gain%':>7}  {'Days':>4}  {'Reason':<12}"
        )
        lines.append("-" * 85)

        for i, t in enumerate(self.state.closed_trades, 1):
            sign = "+" if t.gain_pct >= 0 else ""
            lines.append(
                f"{i:>3}  {t.stock_name:>5}  {t.entry_date:>10}  {t.exit_date:>10}  "
                f"${t.entry_price:>7.2f}  ${t.exit_price:>7.2f}  "
                f"{sign}{t.gain_pct:>6.1f}%  {t.days_held:>4}td  {t.reason:<12}"
            )

        stats = self.state.stats
        win_rate = (stats.wins / stats.total_trades * 100.0) if stats.total_trades > 0 else 0.0
        sign = "+" if stats.raw_pnl >= 0 else ""

        lines.append("-" * 85)
        lines.append(
            f"Total: {stats.total_trades} trades | "
            f"{stats.wins}W/{stats.losses}L ({win_rate:.1f}%) | "
            f"P&L: {sign}${stats.raw_pnl:.2f}"
        )
        lines.append(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _stock_name(self, stock_id: int) -> str:
        """Map stock_id (0-107) to ticker symbol."""
        col_idx = Config.INVESTABLE_START_COL + stock_id
        if self.loader.column_names and col_idx < len(self.loader.column_names):
            return self.loader.column_names[col_idx]
        return f"STOCK_{stock_id}"

    def _find_date_index(self, date_str: str) -> int:
        """Find the data array index for a given date string."""
        dates = self.loader.dates
        for i, d in enumerate(dates):
            if str(d) == date_str or str(d).startswith(date_str):
                return i
        raise ValueError(
            f"Date '{date_str}' not found in data. "
            f"Available range: {dates[0]} to {dates[-1]}"
        )

    @staticmethod
    def _parse_date(date_str: str) -> Optional[datetime]:
        """Try to parse a date string into a datetime object."""
        for fmt in ('%Y-%m-%d', '%m/%d/%Y', '%Y-%m-%d %H:%M:%S'):
            try:
                return datetime.strptime(str(date_str).strip(), fmt)
            except ValueError:
                continue
        return None

    @staticmethod
    def _estimate_next_market_open(current_date: datetime) -> datetime:
        """
        Estimate the next market open date by skipping weekends.

        Does NOT account for exchange holidays — the user should verify
        against their exchange's holiday calendar.
        """
        next_day = current_date + timedelta(days=1)
        # Skip Saturday (5) and Sunday (6)
        while next_day.weekday() >= 5:
            next_day += timedelta(days=1)
        return next_day

    @staticmethod
    def _trading_days_to_weeks(trading_days: int) -> float:
        """Convert trading days to approximate calendar weeks (5 td = 1 week)."""
        return trading_days / 5.0
