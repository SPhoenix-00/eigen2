"""
API Compatibility Tests for committee/ package refactoring.

Verifies that the refactored committee/ package exposes the exact same
public API as the original monolithic committee.py, ensuring no breakage
for external consumers (erl_trainer.py, main.py, etc.).

Run: python test_committee_api.py
"""

import sys
import subprocess
import importlib


def test_backward_compatible_imports():
    """Imports used by erl_trainer.py and main.py must still work."""
    from committee import get_agent_filepath
    from committee import CommitteeManager
    from committee import convert_numpy_types
    from committee import calculate_conviction_threshold

    assert callable(get_agent_filepath), "get_agent_filepath not callable"
    assert callable(convert_numpy_types), "convert_numpy_types not callable"
    assert callable(calculate_conviction_threshold), "calculate_conviction_threshold not callable"
    assert hasattr(CommitteeManager, "save_roster"), "CommitteeManager missing save_roster"
    assert hasattr(CommitteeManager, "load_roster"), "CommitteeManager missing load_roster"
    assert hasattr(CommitteeManager, "check_mirror_status"), "CommitteeManager missing check_mirror_status"
    print("  PASS: backward_compatible_imports")


def test_full_public_api():
    """All symbols from the old committee.py must be importable."""
    from committee import (
        # Classes
        CommitteeManager, CommitteeAgent,
        # Optimization
        committee_objective, optimize_committee,
        interactive_correlation_refinement, automatic_correlation_refinement,
        calculate_coefficient_correlations,
        find_highest_correlation_pair, find_best_swap_candidate,
        # Validation
        evaluate_agent_on_slice, evaluate_committee_on_slice,
        run_validation, run_validation_sweep,
        run_quorum_sweep, run_conviction_sweep, run_combined_sweep,
        # Data
        verify_data_split, load_normalization_stats,
        get_holdout_data, get_validation_data,
        load_global50_candidates, load_agent_actor_only,
        get_agent_filepath,
        # Agent stats
        calculate_conviction_threshold, recalculate_conviction_thresholds,
        # Utilities
        calculate_expectancy, calculate_max_drawdown,
        parse_date_input, parse_date_flexible, find_date_index,
        sanitize_date_for_filename, convert_numpy_types,
        format_quality_ratio,
        # DRY helpers (new but part of public API)
        build_member_data, build_metrics_result,
        aggregate_slice_metrics, aggregate_consensus_stats,
        enrich_closed_trades, generate_validation_slices,
        # Constants
        GLOBAL50_BASE_DIR, ROSTER_FILENAME, CORRELATION_FILENAME,
    )
    print("  PASS: full_public_api")


def test_committee_manager_interface():
    """CommitteeManager must have all expected methods and attributes."""
    from committee import CommitteeManager
    expected_methods = [
        "save_roster", "load_roster", "check_mirror_status",
        "update_maverick_flags",
    ]
    expected_attrs = [
        "context_window_days", "context_window_id",
        "cloud_sync",
        "local_base", "local_committee_dir",
        "local_roster_path", "local_correlation_path",
        "cloud_base", "cloud_committee_base",
        "cloud_roster_path", "cloud_correlation_path",
    ]
    for method in expected_methods:
        assert hasattr(CommitteeManager, method), f"CommitteeManager missing method: {method}"

    # Instantiate to check attributes
    mgr = CommitteeManager(151)
    for attr in expected_attrs:
        assert hasattr(mgr, attr), f"CommitteeManager instance missing attr: {attr}"
    print("  PASS: committee_manager_interface")


def test_committee_agent_interface():
    """CommitteeAgent must have the same interface methods."""
    from committee import CommitteeAgent
    expected_methods = [
        "predict_action", "_apply_consensus",
        "get_consensus_summary", "reset_episode", "cleanup",
    ]
    for method in expected_methods:
        assert hasattr(CommitteeAgent, method), f"CommitteeAgent missing method: {method}"
    print("  PASS: committee_agent_interface")


def test_utility_functions_behavior():
    """Spot-check that utility functions produce correct output."""
    import numpy as np
    from committee import (
        convert_numpy_types, format_quality_ratio,
        sanitize_date_for_filename, calculate_expectancy,
        generate_validation_slices, aggregate_slice_metrics,
        aggregate_consensus_stats, build_member_data,
    )

    # convert_numpy_types
    assert convert_numpy_types(np.int64(42)) == 42
    assert convert_numpy_types(np.float32(3.14)) == float(np.float32(3.14))
    assert convert_numpy_types({"a": np.array([1, 2])}) == {"a": [1, 2]}

    # format_quality_ratio
    assert format_quality_ratio(float("inf")) == "inf"
    assert format_quality_ratio(float("nan")) == "nan"
    assert format_quality_ratio(2.5) == "2.500"

    # sanitize_date_for_filename
    assert "/" not in sanitize_date_for_filename("10/7/2022")
    assert " " not in sanitize_date_for_filename("2022-01-01 12:00:00")

    # calculate_expectancy
    assert calculate_expectancy([]) == 0.0
    trades = [{"gain_pct": 10.0}, {"gain_pct": -5.0}, {"gain_pct": 8.0}]
    exp = calculate_expectancy(trades)
    assert isinstance(exp, float)

    # generate_validation_slices
    holdout_info = {
        "val_start": 200, "val_end": 800,
        "holdout_start": 801, "holdout_end": 1100,
    }
    slices = generate_validation_slices(holdout_info, 155)
    assert len(slices) == 5, f"Expected 5 slices, got {len(slices)}"
    assert sum(1 for s in slices if s["type"] == "validation") == 3
    assert sum(1 for s in slices if s["type"] == "holdout") == 2
    for s in slices:
        assert s["end"] - s["start"] == 155, f"Slice length mismatch: {s}"

    # aggregate_slice_metrics
    mock_slices = [
        {"fitness": 100, "num_wins": 8, "num_losses": 2, "num_trades": 10,
         "raw_pnl": 500, "peak_capital_employed": 5000, "expectancy": 5.0},
        {"fitness": 200, "num_wins": 6, "num_losses": 4, "num_trades": 10,
         "raw_pnl": 800, "peak_capital_employed": 8000, "expectancy": 3.0},
    ]
    agg = aggregate_slice_metrics(mock_slices)
    assert agg["mean_fitness"] == 150.0
    assert agg["total_trades"] == 20
    assert agg["mean_win_rate"] == 14 / 20
    assert agg["mean_quality_ratio"] == 14 / 6

    # aggregate_consensus_stats
    empty_result = aggregate_consensus_stats([{"note": "disabled"}])
    assert "note" in empty_result

    valid_stats = [
        {"unanimity_pct": 10, "avg_consensus_votes": 3.0,
         "trades_by_quorum": 5, "trades_by_conviction_only": 2},
        {"unanimity_pct": 20, "avg_consensus_votes": 4.0,
         "trades_by_quorum": 8, "trades_by_conviction_only": 3},
    ]
    cs = aggregate_consensus_stats(valid_stats)
    assert cs["avg_unanimity_pct"] == 15.0
    assert cs["total_trades_by_quorum"] == 13
    assert cs["total_trades_by_conviction_only"] == 5

    # build_member_data
    entry = {
        "run_name": "test-run", "agent_id": 42,
        "gauntlet_score": 100.0, "roi": 5.0,
        "expectancy": 2.0, "quality_ratio": 3.0,
        "win_ratio": 0.7, "is_maverick": True,
    }
    member = build_member_data(entry, 1.5)
    assert member["filename"] == "test-run_42.pth"
    assert member["is_maverick"] is True
    assert member["stats"]["conviction_threshold"] == 1.5

    print("  PASS: utility_functions_behavior")


def test_cli_entry_points():
    """Both CLI invocation methods must work."""
    r1 = subprocess.run(
        [sys.executable, "committee.py", "--help"],
        capture_output=True, text=True, timeout=30
    )
    assert r1.returncode == 0, f"committee.py --help failed: {r1.stderr}"
    assert "--draft" in r1.stdout, "committee.py --help missing --draft"
    assert "--validate" in r1.stdout, "committee.py --help missing --validate"
    assert "--mirror" in r1.stdout, "committee.py --help missing --mirror"
    assert "--sweep-both" in r1.stdout, "committee.py --help missing --sweep-both"
    assert "--swap-agent" in r1.stdout, "committee.py --help missing --swap-agent"

    r2 = subprocess.run(
        [sys.executable, "-m", "committee", "--help"],
        capture_output=True, text=True, timeout=30
    )
    assert r2.returncode == 0, f"python -m committee --help failed: {r2.stderr}"
    assert "--draft" in r2.stdout, "python -m committee --help missing --draft"

    print("  PASS: cli_entry_points")


def test_no_circular_imports():
    """Importing individual submodules must not cause circular import errors."""
    modules = [
        "committee",
        "committee.utils",
        "committee.manager",
        "committee.agent",
        "committee.optimization",
        "committee.validation",
    ]
    for mod_name in modules:
        if mod_name in sys.modules:
            del sys.modules[mod_name]

    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
        except ImportError as e:
            raise AssertionError(f"Failed to import {mod_name}: {e}")

    print("  PASS: no_circular_imports")


if __name__ == "__main__":
    print()
    print("=" * 60)
    print("COMMITTEE PACKAGE API COMPATIBILITY TESTS")
    print("=" * 60)

    tests = [
        test_backward_compatible_imports,
        test_full_public_api,
        test_committee_manager_interface,
        test_committee_agent_interface,
        test_utility_functions_behavior,
        test_cli_entry_points,
        test_no_circular_imports,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {test_fn.__name__}: {e}")
            failed += 1

    print()
    print("=" * 60)
    if failed == 0:
        print(f"ALL {passed} TESTS PASSED")
    else:
        print(f"{failed} FAILED, {passed} passed")
    print("=" * 60)

    sys.exit(1 if failed > 0 else 0)
