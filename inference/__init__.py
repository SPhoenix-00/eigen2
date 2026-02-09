"""
Inference package: live trading with committee consensus.

Provides daily inference pipeline that loads the committee, manages persistent
trading positions, accepts daily data updates, and outputs buy/sell orders.
"""

from inference.engine import LiveTradingEngine
from inference.state import TradingState, LivePosition, ClosedTrade

__all__ = ['LiveTradingEngine', 'TradingState', 'LivePosition', 'ClosedTrade']
