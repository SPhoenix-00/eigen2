"""
Inference CLI — Daily live trading with committee consensus.

Usage:
    python inference.py --start [DATE]          Start a new trading episode
    python inference.py --daily UPDATE.json     Run daily inference with new data
    python inference.py --catchup [DATE]        Run inference up to DATE using pickle data
    python inference.py --status                Show current portfolio status
    python inference.py --history               Show closed trade history

Examples:
    # Start an episode from the last available date in the pickle:
    python inference.py --start

    # Start from a specific date:
    python inference.py --start 2026-02-09

    # Feed today's data and get orders:
    python inference.py --daily daily_update_20260210.json

    # Run inference on the last day in the pickle (no JSON needed):
    python inference.py --catchup

    # Run inference up to a specific date already in the pickle:
    python inference.py --catchup 2026-02-20

    # Check portfolio without running inference:
    python inference.py --status

    # View trade history:
    python inference.py --history

Daily update JSON format:
    {
        "date": "2026-02-10",
        "columns": {
            "SPX_INDEX": [Open, Close, High, Low, RSI, MACD, MACD_Signal, Trix, xDiffDMA],
            "DFAC": [25.10, 25.30, 25.45, 25.05, 62.1, 0.15, 0.12, 0.003, 0.25],
            ...
        }
    }

Notes:
    - All day counts (holding periods, trading days) are in TRADING DAYS,
      not calendar days. Weekends/holidays do not count.
    - "Next market open" estimates skip weekends but do NOT account for
      exchange holidays. Verify against your exchange calendar.
    - State is persisted in inference/trading_state.json between runs.
"""

import argparse
import os
import sys
from pathlib import Path

# Fix Windows console encoding for Unicode characters (checkmarks, etc.)
if sys.platform == 'win32':
    os.environ.setdefault('PYTHONIOENCODING', 'utf-8')
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, OSError):
        pass  # Fallback for older Python or non-standard streams

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent))

from inference.engine import LiveTradingEngine


def main():
    parser = argparse.ArgumentParser(
        description="Eigen 2 Live Inference — Daily trading with committee consensus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        '--start',
        nargs='?',
        const='__latest__',
        default=None,
        metavar='DATE',
        help='Start a new trading episode. Optional DATE (e.g., "2026-02-09") '
             'defaults to the last available date in the dataset.'
    )
    parser.add_argument(
        '--daily',
        type=str,
        default=None,
        metavar='UPDATE_FILE',
        help='Path to daily update JSON file. Runs one day of inference '
             'and outputs actionable buy/sell orders.'
    )
    parser.add_argument(
        '--catchup',
        nargs='?',
        const='__latest__',
        default=None,
        metavar='DATE',
        help='Run inference using data already in the pickle. Starts a fresh '
             'episode and runs through to DATE (or the last available date). '
             'No JSON file needed.'
    )
    parser.add_argument(
        '--status',
        action='store_true',
        help='Show current portfolio status and open positions.'
    )
    parser.add_argument(
        '--history',
        action='store_true',
        help='Show closed trade history for the current episode.'
    )
    parser.add_argument(
        '--state-file',
        type=str,
        default=None,
        metavar='PATH',
        help='Override path for trading state file '
             '(default: inference/trading_state.json).'
    )

    args = parser.parse_args()

    # Validate: at least one action required
    has_action = (
        args.start is not None or args.daily is not None
        or args.catchup is not None or args.status or args.history
    )
    if not has_action:
        parser.print_help()
        print("\nError: specify one of --start, --daily, --catchup, --status, or --history")
        sys.exit(1)

    # Validate --daily file exists
    if args.daily and not Path(args.daily).exists():
        print(f"Error: daily update file not found: {args.daily}")
        sys.exit(1)

    # Create engine
    engine = LiveTradingEngine(state_path=args.state_file)

    # --- Status/History: load state only (no data/committee needed) ---
    standalone_query = (
        args.start is None and args.daily is None and args.catchup is None
    )
    if args.status and standalone_query:
        from inference.state import TradingState, DEFAULT_STATE_PATH
        state_path = Path(args.state_file) if args.state_file else DEFAULT_STATE_PATH
        engine.state = TradingState.load(state_path)
        print(engine.get_status())
        return

    if args.history and standalone_query:
        from inference.state import TradingState, DEFAULT_STATE_PATH
        state_path = Path(args.state_file) if args.state_file else DEFAULT_STATE_PATH
        engine.state = TradingState.load(state_path)
        print(engine.get_history())
        return

    # --- Actions that need full initialization ---
    engine.initialize()

    # Start a new episode
    if args.start is not None:
        start_date = None if args.start == '__latest__' else args.start
        engine.start_episode(start_date)
        print("\nEpisode ready. Use --daily to feed data and run inference.")

    # Run daily inference
    if args.daily is not None:
        report = engine.run_daily(args.daily)
        print(report)

    # Catch-up: run inference over pickle data up to a target date
    if args.catchup is not None:
        target = None if args.catchup == '__latest__' else args.catchup
        report = engine.run_catchup(target_date=target)
        print(report)

    # Show status after other operations
    if args.status and not standalone_query:
        print(engine.get_status())

    if args.history and not standalone_query:
        print(engine.get_history())

    # Cleanup committee GPU memory
    if engine.committee:
        engine.committee.cleanup()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
