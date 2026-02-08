"""
Tests for Global Hall of Fame functionality.
Converted from root-level test_global_hof.py to pytest format.
"""

import json
import shutil
from pathlib import Path
from erl.global_hof import GlobalHallOfFame, LeagueRules, GlobalHoFEntry
from utils.cloud_sync import CloudSync


def test_basic_functionality(tmp_path):
    """Test basic Global HoF operations."""
    cloud_sync = CloudSync(provider="local")
    league_rules = LeagueRules(context_window_days=504)

    test_checkpoint_dir = tmp_path / "checkpoints"
    test_checkpoint_dir.mkdir()

    global_hof = GlobalHallOfFame(
        cloud_sync=cloud_sync,
        run_name="test-run-001",
        league_rules=league_rules,
        checkpoint_dir=test_checkpoint_dir,
        disable_global50=False
    )

    assert hasattr(global_hof, 'enabled')
    assert hasattr(global_hof, 'league_compatible')
    assert hasattr(global_hof, 'entry_threshold')

    # Test entry creation
    entry = GlobalHoFEntry(
        agent_id=123,
        run_name="test-run-001",
        gauntlet_score=1234.56,
        generation=10,
        roi=15.5,
        expectancy=2.3,
        quality_ratio=0.67,
        win_ratio=0.75,
        total_trades=75
    )

    filename = entry.get_filename()
    assert filename == "test-run-001_123.pth"

    # Test serialization round-trip
    entry_dict = entry.to_dict()
    entry_restored = GlobalHoFEntry.from_dict(entry_dict)
    assert entry_restored.gauntlet_score == entry.gauntlet_score
    assert entry_restored.run_name == entry.run_name


def test_league_rules_matching():
    """League rules matching logic."""
    rules1 = LeagueRules(context_window_days=504)
    rules2 = LeagueRules(context_window_days=504)
    rules3 = LeagueRules(context_window_days=252)

    assert rules1.matches(rules2), "Same rules should match"
    assert not rules1.matches(rules3), "Different rules should not match"


def test_promotion_when_disabled(tmp_path):
    """When disabled (local mode), should always return False."""
    cloud_sync = CloudSync(provider="local")
    league_rules = LeagueRules(context_window_days=504)

    global_hof = GlobalHallOfFame(
        cloud_sync=cloud_sync,
        run_name="test-run-001",
        league_rules=league_rules,
        checkpoint_dir=tmp_path,
        disable_global50=False
    )

    assert global_hof.should_promote(9999.0, roi=10.0, expectancy=1.0) == False


def test_stats_generation(tmp_path):
    """Stats generation returns expected keys."""
    cloud_sync = CloudSync(provider="local")
    league_rules = LeagueRules(context_window_days=504)

    global_hof = GlobalHallOfFame(
        cloud_sync=cloud_sync,
        run_name="test-run-001",
        league_rules=league_rules,
        checkpoint_dir=tmp_path,
        disable_global50=False
    )

    stats = global_hof.get_stats()
    assert 'enabled' in stats
    assert 'size' in stats
    assert 'entry_threshold' in stats


def test_json_ledger_format():
    """Test the global50.json ledger serialization format."""
    league_rules = LeagueRules(context_window_days=504)

    entries = [
        GlobalHoFEntry(
            agent_id=1, run_name="azure-thunder-123",
            gauntlet_score=1500.0, generation=25,
            roi=18.5, expectancy=2.8, quality_ratio=0.75,
            win_ratio=0.80, total_trades=80
        ),
        GlobalHoFEntry(
            agent_id=2, run_name="crimson-wave-456",
            gauntlet_score=1450.0, generation=30,
            roi=16.2, expectancy=2.5, quality_ratio=0.73,
            win_ratio=0.77, total_trades=75
        )
    ]

    ledger_data = {
        'league_rules': league_rules.to_dict(),
        'entries': [e.to_dict() for e in entries],
        'capacity': 50,
        'version': '1.0'
    }

    # Serialization round-trip
    json_str = json.dumps(ledger_data, indent=2)
    restored_data = json.loads(json_str)

    restored_rules = LeagueRules.from_dict(restored_data['league_rules'])
    assert restored_rules.context_window_days == 504

    restored_entries = [GlobalHoFEntry.from_dict(e) for e in restored_data['entries']]
    assert len(restored_entries) == 2
    assert restored_entries[0].gauntlet_score == 1500.0
