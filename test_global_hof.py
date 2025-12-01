"""
Simple test script for Global Hall of Fame functionality.
Tests the core logic without requiring a full training run.
"""

import sys
import json
import shutil
from pathlib import Path
from erl.global_hof import GlobalHallOfFame, LeagueRules, GlobalHoFEntry
from utils.cloud_sync import CloudSync


def test_basic_functionality():
    """Test basic Global HoF operations."""

    print("\n" + "="*60)
    print("Testing Global Hall of Fame - Basic Functionality")
    print("="*60)

    # Create a local-only CloudSync for testing
    cloud_sync = CloudSync(provider="local")

    # Define league rules
    league_rules = LeagueRules(context_window_days=504)

    # Create test checkpoint directory
    test_checkpoint_dir = Path("test_checkpoints")
    test_checkpoint_dir.mkdir(exist_ok=True)

    # Initialize Global HoF
    print("\n1. Initializing Global Hall of Fame...")
    global_hof = GlobalHallOfFame(
        cloud_sync=cloud_sync,
        run_name="test-run-001",
        league_rules=league_rules,
        checkpoint_dir=test_checkpoint_dir,
        disable_global50=False
    )

    print(f"   Enabled: {global_hof.enabled}")
    print(f"   League Compatible: {global_hof.league_compatible}")
    print(f"   Entry Threshold: {global_hof.entry_threshold}")

    # Test entry creation
    print("\n2. Testing GlobalHoFEntry...")
    entry = GlobalHoFEntry(
        agent_id=123,
        run_name="test-run-001",
        gauntlet_score=1234.56,
        generation=10,
        roi=15.5,
        expectancy=2.3,
        quality_count=50,
        total_trades=75
    )

    filename = entry.get_filename()
    print(f"   Entry filename: {filename}")
    assert filename == "1234.56_test-run-001_123.pth", f"Expected '1234.56_test-run-001_123.pth', got '{filename}'"

    # Test serialization
    print("\n3. Testing entry serialization...")
    entry_dict = entry.to_dict()
    print(f"   Serialized keys: {list(entry_dict.keys())}")

    entry_restored = GlobalHoFEntry.from_dict(entry_dict)
    assert entry_restored.gauntlet_score == entry.gauntlet_score
    assert entry_restored.run_name == entry.run_name
    print("   [PASS] Serialization works correctly")

    # Test league rules matching
    print("\n4. Testing league rules validation...")
    rules1 = LeagueRules(context_window_days=504)
    rules2 = LeagueRules(context_window_days=504)
    rules3 = LeagueRules(context_window_days=252)

    assert rules1.matches(rules2), "Same rules should match"
    assert not rules1.matches(rules3), "Different rules should not match"
    print("   [PASS] League rules validation works correctly")

    # Test should_promote logic
    print("\n5. Testing promotion threshold logic...")
    # When disabled (local mode), should always return False
    assert global_hof.should_promote(9999.0) == False, "Should not promote when disabled"
    print("   [PASS] Correctly rejects promotion when disabled")

    # Test get_stats
    print("\n6. Testing stats generation...")
    stats = global_hof.get_stats()
    print(f"   Stats keys: {list(stats.keys())}")
    assert 'enabled' in stats
    assert 'size' in stats
    assert 'entry_threshold' in stats
    print("   [PASS] Stats generation works correctly")

    # Cleanup
    print("\n7. Cleaning up test files...")
    if test_checkpoint_dir.exists():
        shutil.rmtree(test_checkpoint_dir)
    if global_hof.local_dir.exists():
        shutil.rmtree(global_hof.local_dir)
    print("   [PASS] Cleanup complete")

    print("\n" + "="*60)
    print("[PASS] ALL TESTS PASSED")
    print("="*60)


def test_json_ledger_format():
    """Test the global50.json ledger format."""

    print("\n" + "="*60)
    print("Testing Global50.json Ledger Format")
    print("="*60)

    # Create test data
    league_rules = LeagueRules(context_window_days=504)

    entries = [
        GlobalHoFEntry(
            agent_id=1,
            run_name="azure-thunder-123",
            gauntlet_score=1500.0,
            generation=25,
            roi=18.5,
            expectancy=2.8,
            quality_count=60,
            total_trades=80
        ),
        GlobalHoFEntry(
            agent_id=2,
            run_name="crimson-wave-456",
            gauntlet_score=1450.0,
            generation=30,
            roi=16.2,
            expectancy=2.5,
            quality_count=55,
            total_trades=75
        )
    ]

    # Create ledger data structure
    ledger_data = {
        'league_rules': league_rules.to_dict(),
        'entries': [e.to_dict() for e in entries],
        'capacity': 50,
        'version': '1.0'
    }

    # Test serialization
    print("\n1. Serializing ledger to JSON...")
    json_str = json.dumps(ledger_data, indent=2)
    print(f"   JSON length: {len(json_str)} characters")

    # Test deserialization
    print("\n2. Deserializing ledger from JSON...")
    restored_data = json.loads(json_str)

    restored_rules = LeagueRules.from_dict(restored_data['league_rules'])
    assert restored_rules.context_window_days == 504

    restored_entries = [GlobalHoFEntry.from_dict(e) for e in restored_data['entries']]
    assert len(restored_entries) == 2
    assert restored_entries[0].gauntlet_score == 1500.0
    print("   [PASS] Deserialization works correctly")

    # Print sample JSON
    print("\n3. Sample global50.json format:")
    print("-" * 60)
    print(json_str)
    print("-" * 60)

    print("\n" + "="*60)
    print("[PASS] LEDGER FORMAT TEST PASSED")
    print("="*60)


if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_json_ledger_format()

        print("\n" + "="*60)
        print("[SUCCESS] ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\nThe Global Hall of Fame system is ready to use!")
        print("\nNext steps:")
        print("1. Set up GCP credentials (GOOGLE_APPLICATION_CREDENTIALS)")
        print("2. Set CLOUD_PROVIDER=gcs and CLOUD_BUCKET=<your-bucket>")
        print("3. Run training with --consistency or normal mode")
        print("4. Agents passing the gauntlet will auto-promote to Global 50")

    except Exception as e:
        print(f"\n[FAIL] TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
