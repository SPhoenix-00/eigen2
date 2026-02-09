"""
Inference state persistence: LivePosition and TradingState.

Tracks open/closed positions and episode metadata across daily inference runs.
All day counts (days_held, trading_day, etc.) are in TRADING DAYS, not calendar days.
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Dict, Optional

# Default state file location
DEFAULT_STATE_PATH = Path(__file__).parent / "trading_state.json"


@dataclass
class LivePosition:
    """
    An open position tracked by the inference engine.

    All day counts are in trading days (1 row in data = 1 trading day).
    Weekends/holidays are not counted.
    """
    stock_id: int               # 0-107 (investable stock index)
    stock_name: str             # Ticker symbol (e.g., "AAPL")
    entry_price: float          # Close price on entry day
    coefficient: float          # Position size multiplier from committee
    sale_target_pct: float      # Target gain percentage (10-50%)
    sale_target_price: float    # Absolute target price
    days_held: int              # Trading days since entry (incremented each run_daily)
    entry_date: str             # Calendar date string of entry
    entry_day_idx: int          # Index into data array on entry day

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'LivePosition':
        return cls(**d)


@dataclass
class ClosedTrade:
    """A completed trade with entry/exit details."""
    stock_id: int
    stock_name: str
    entry_price: float
    exit_price: float
    coefficient: float
    sale_target_pct: float
    sale_target_price: float
    gain_pct: float             # ((exit - entry) / entry) * 100
    days_held: int              # Trading days held
    entry_date: str
    exit_date: str
    reason: str                 # "target_hit", "forced_exit", "manual"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'ClosedTrade':
        return cls(**d)


@dataclass
class EpisodeStats:
    """Running statistics for the current episode."""
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    raw_pnl: float = 0.0
    peak_capital_employed: float = 0.0
    current_capital_employed: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'EpisodeStats':
        return cls(**d)


@dataclass
class TradingState:
    """
    Persistent state for a live trading episode.

    Serialized to/from JSON between daily inference runs.
    All day counts are in trading days unless noted otherwise.
    """
    # Episode metadata
    episode_start_date: str = ""        # Calendar date when episode started
    episode_start_idx: int = 0          # Data array index at episode start
    current_day_idx: int = 0            # Current position in data array
    trading_day: int = 0                # Count of run_daily() calls (trading days elapsed)
    last_processed_date: str = ""       # Calendar date of last processed day

    # Positions
    open_positions: List[LivePosition] = field(default_factory=list)
    closed_trades: List[ClosedTrade] = field(default_factory=list)

    # Running stats
    stats: EpisodeStats = field(default_factory=EpisodeStats)

    def save(self, path: Optional[Path] = None):
        """Save state to JSON file."""
        path = Path(path) if path else DEFAULT_STATE_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'episode_start_date': self.episode_start_date,
            'episode_start_idx': self.episode_start_idx,
            'current_day_idx': self.current_day_idx,
            'trading_day': self.trading_day,
            'last_processed_date': self.last_processed_date,
            'open_positions': [p.to_dict() for p in self.open_positions],
            'closed_trades': [t.to_dict() for t in self.closed_trades],
            'stats': self.stats.to_dict(),
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> Optional['TradingState']:
        """Load state from JSON file. Returns None if file doesn't exist."""
        path = Path(path) if path else DEFAULT_STATE_PATH

        if not path.exists():
            return None

        with open(path, 'r') as f:
            data = json.load(f)

        state = cls()
        state.episode_start_date = data.get('episode_start_date', '')
        state.episode_start_idx = data.get('episode_start_idx', 0)
        state.current_day_idx = data.get('current_day_idx', 0)
        state.trading_day = data.get('trading_day', 0)
        state.last_processed_date = data.get('last_processed_date', '')

        state.open_positions = [
            LivePosition.from_dict(p) for p in data.get('open_positions', [])
        ]
        state.closed_trades = [
            ClosedTrade.from_dict(t) for t in data.get('closed_trades', [])
        ]
        state.stats = EpisodeStats.from_dict(data.get('stats', {}))

        return state

    def get_open_stock_ids(self) -> List[int]:
        """Return list of stock IDs with open positions."""
        return [p.stock_id for p in self.open_positions]

    def get_position_by_stock(self, stock_id: int) -> Optional[LivePosition]:
        """Find open position for a given stock ID."""
        for p in self.open_positions:
            if p.stock_id == stock_id:
                return p
        return None

    def close_position(self, stock_id: int, exit_price: float, exit_date: str,
                       reason: str) -> Optional[ClosedTrade]:
        """
        Close an open position and move it to closed_trades.

        Args:
            stock_id: Stock to close
            exit_price: Price at exit
            exit_date: Calendar date of exit
            reason: "target_hit", "forced_exit", or "manual"

        Returns:
            ClosedTrade if position was found and closed, None otherwise
        """
        position = self.get_position_by_stock(stock_id)
        if position is None:
            return None

        gain_pct = ((exit_price - position.entry_price) / position.entry_price) * 100.0

        trade = ClosedTrade(
            stock_id=position.stock_id,
            stock_name=position.stock_name,
            entry_price=position.entry_price,
            exit_price=exit_price,
            coefficient=position.coefficient,
            sale_target_pct=position.sale_target_pct,
            sale_target_price=position.sale_target_price,
            gain_pct=gain_pct,
            days_held=position.days_held,
            entry_date=position.entry_date,
            exit_date=exit_date,
            reason=reason,
        )

        # Remove from open positions
        self.open_positions = [p for p in self.open_positions if p.stock_id != stock_id]

        # Add to closed trades
        self.closed_trades.append(trade)

        # Update stats
        shares = int(position.coefficient)
        pnl = (exit_price - position.entry_price) * shares
        self.stats.total_trades += 1
        self.stats.raw_pnl += pnl
        self.stats.current_capital_employed -= position.entry_price * shares

        if gain_pct > 0:
            self.stats.wins += 1
        else:
            self.stats.losses += 1

        return trade
