"""
Test script for Hall of Fame implementation
"""

import torch
import numpy as np
from pathlib import Path
import tempfile
import shutil

from erl.hall_of_fame import HallOfFame
from models.ddpg_agent import DDPGAgent


def test_hall_of_fame():
    """Test basic Hall of Fame functionality."""
    print("="*60)
    print("Testing Hall of Fame Implementation")
    print("="*60)

    # Create temporary directory for testing
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\nTest directory: {temp_dir}")

    try:
        # 1. Test initialization
        print("\n1. Testing initialization...")
        hof = HallOfFame(capacity=5, checkpoint_dir=temp_dir)
        print(f"   [OK] Created HoF with capacity {hof.capacity}")
        assert len(hof) == 0, "HoF should be empty on initialization"
        assert not hof.is_full(), "HoF should not be full initially"
        print("   [OK] Initial state correct")

        # 2. Test adding agents
        print("\n2. Testing agent addition...")
        agents = []
        for i in range(7):  # Create 7 agents (more than capacity)
            agent = DDPGAgent(agent_id=i)
            agents.append(agent)

            # Add to HoF with increasing validation scores
            val_score = 100.0 + i * 10  # Scores: 100, 110, 120, ...
            was_added = hof.add(agent, val_score, generation=i)

            if i < 5:
                assert was_added, f"Agent {i} should be added (HoF not full)"
                print(f"   [OK] Added agent {i} with score {val_score:.2f}")
            else:
                # These should replace worse agents
                if was_added:
                    print(f"   [OK] Agent {i} (score {val_score:.2f}) replaced worse agent")
                else:
                    print(f"   [SKIP] Agent {i} (score {val_score:.2f}) NOT added (too low)")

        # Check final state
        print(f"\n   Final HoF size: {len(hof)}/{hof.capacity}")
        assert len(hof) == 5, "HoF should be at capacity"
        assert hof.is_full(), "HoF should be full"

        # 3. Test statistics
        print("\n3. Testing statistics...")
        stats = hof.get_stats()
        print(f"   Size: {stats['size']}")
        print(f"   Best score: {stats['best_score']:.2f}")
        print(f"   Worst score: {stats['worst_score']:.2f}")
        print(f"   Mean score: {stats['mean_score']:.2f}")
        print(f"   Std score: {stats['std_score']:.2f}")
        assert stats['size'] == 5, "Stats should show size of 5"
        assert stats['best_score'] == 160.0, "Best score should be 160"
        assert stats['worst_score'] == 120.0, "Worst score should be 120"
        print("   [OK] Statistics correct")

        # 4. Test sampling
        print("\n4. Testing random sampling...")
        sampled = hof.sample_random(k=3)
        assert len(sampled) == 3, "Should sample 3 agents"
        print(f"   [OK] Sampled {len(sampled)} agents")

        # Verify sampled agents are valid DDPGAgent instances
        for i, agent in enumerate(sampled):
            assert isinstance(agent, DDPGAgent), "Sampled agent should be DDPGAgent"
            print(f"   [OK] Agent {i} is valid DDPGAgent")

        # 5. Test save/load
        print("\n5. Testing save/load...")
        hof.save()
        print("   [OK] Saved HoF to disk")

        # Create new HoF and load
        hof2 = HallOfFame(capacity=5, checkpoint_dir=temp_dir)
        hof2.load()
        print("   [OK] Loaded HoF from disk")

        stats2 = hof2.get_stats()
        assert stats2['size'] == stats['size'], "Loaded HoF should have same size"
        assert stats2['best_score'] == stats['best_score'], "Loaded HoF should have same best score"
        print(f"   [OK] Loaded HoF matches (size: {stats2['size']}, best: {stats2['best_score']:.2f})")

        # 6. Test getting best agent
        print("\n6. Testing get_best()...")
        best_agent, best_score = hof.get_best()
        assert best_score == 160.0, "Best score should be 160"
        assert isinstance(best_agent, DDPGAgent), "Best agent should be DDPGAgent"
        print(f"   [OK] Retrieved best agent with score {best_score:.2f}")

        # 7. Test should_admit
        print("\n7. Testing should_admit()...")
        assert hof.should_admit(130.0), "Score of 130 should be admitted"
        assert not hof.should_admit(110.0), "Score of 110 should NOT be admitted"
        print("   [OK] Admission logic correct")

        print("\n" + "="*60)
        print("[SUCCESS] All tests passed!")
        print("="*60)

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"\n[OK] Cleaned up test directory")


def test_batch_admission():
    """Test the new batch admission with cascading swaps."""
    print("\n" + "="*60)
    print("Testing Batch Admission (update_from_generation)")
    print("="*60)

    # Create temporary directory for testing
    temp_dir = Path(tempfile.mkdtemp())
    print(f"\nTest directory: {temp_dir}")

    try:
        # 1. Test Phase 1: Initial filling with positive scores only
        print("\n1. Testing Phase 1 (Initial Filling)...")
        hof = HallOfFame(capacity=10, checkpoint_dir=temp_dir)

        # Create candidates with mixed scores (4 positive, 1 negative)
        candidates = []
        scores = [2041.77, 917.25, 805.23, 636.97, -53.44]  # From user's example
        for i, score in enumerate(scores):
            agent = DDPGAgent(agent_id=i + 20)
            candidates.append((agent, score, i + 20))

        # HoF is empty (0/10), should admit all positive (4 agents)
        results = hof.update_from_generation(candidates, generation=5)

        admitted = [r for r in results if r[2] == 'admitted']
        rejected_negative = [r for r in results if r[2] == 'rejected_negative']

        print(f"   Candidates: {len(candidates)}")
        print(f"   Admitted: {len(admitted)}")
        print(f"   Rejected (negative): {len(rejected_negative)}")

        assert len(admitted) == 4, f"Should admit 4 positive-score agents, got {len(admitted)}"
        assert len(rejected_negative) == 1, f"Should reject 1 negative-score agent, got {len(rejected_negative)}"
        assert len(hof) == 4, f"HoF should have 4 entries, got {len(hof)}"
        print("   [OK] Phase 1 correctly admits all positive-score agents")

        # 2. Test Phase 2: Cascading swaps when full
        print("\n2. Testing Phase 2 (Cascading Swaps)...")

        # First fill the HoF to capacity with lower-scoring agents
        hof2 = HallOfFame(capacity=5, checkpoint_dir=temp_dir / "hof2")
        (temp_dir / "hof2").mkdir(exist_ok=True)

        # Add 5 agents with scores 100-140
        initial_candidates = []
        for i in range(5):
            agent = DDPGAgent(agent_id=i)
            initial_candidates.append((agent, 100.0 + i * 10, i))
        hof2.update_from_generation(initial_candidates, generation=1)

        print(f"   Initial HoF: {[e.validation_score for e in sorted(hof2.entries, key=lambda x: x.validation_score)]}")
        assert len(hof2) == 5, "HoF should be full"

        # Now add new candidates that should cascade-replace the worst entries
        # Scores: 155, 125, 115, 85
        # Expected: 155 replaces 100 (HoF: [110,120,130,140,155])
        #           125 replaces 110 (HoF: [120,125,130,140,155])
        #           115 < 120 (new worst), rejected
        #           85 < 120, rejected
        new_candidates = []
        new_scores = [155.0, 125.0, 115.0, 85.0]
        for i, score in enumerate(new_scores):
            agent = DDPGAgent(agent_id=i + 10)
            new_candidates.append((agent, score, i + 10))

        results2 = hof2.update_from_generation(new_candidates, generation=2)

        replaced = [r for r in results2 if r[2].startswith('replaced_')]
        rejected = [r for r in results2 if r[2] == 'rejected_not_better']

        print(f"   New candidates: {new_scores}")
        print(f"   Replaced: {len(replaced)}")
        print(f"   Rejected: {len(rejected)}")

        assert len(replaced) == 2, f"Should replace 2 agents, got {len(replaced)}"
        assert len(rejected) == 2, f"Should reject 2 agents, got {len(rejected)}"

        final_scores = sorted([e.validation_score for e in hof2.entries])
        expected_final = [120.0, 125.0, 130.0, 140.0, 155.0]
        print(f"   Final HoF scores: {final_scores}")
        print(f"   Expected: {expected_final}")
        assert final_scores == expected_final, f"Final scores mismatch: {final_scores} != {expected_final}"
        print("   [OK] Cascading swaps work correctly")

        # 3. Test mixed scenario: partial fill then swaps
        print("\n3. Testing mixed scenario (fill + swap in one call)...")
        hof3 = HallOfFame(capacity=5, checkpoint_dir=temp_dir / "hof3")
        (temp_dir / "hof3").mkdir(exist_ok=True)

        # Add 3 agents (HoF has room for 2 more)
        for i in range(3):
            agent = DDPGAgent(agent_id=i)
            hof3.add(agent, 100.0 + i * 10, generation=1)

        print(f"   Initial HoF (3/5): {[e.validation_score for e in sorted(hof3.entries, key=lambda x: x.validation_score)]}")

        # Add 4 new candidates: 2 should fill, 1 should swap, 1 should be rejected
        # Current worst is 100, scores in HoF: [100, 110, 120]
        # New scores: 200, 150, 90, 50
        # Expected: 200 fills, 150 fills (now full), 90 rejected (worse than 100), 50 rejected
        mixed_candidates = []
        mixed_scores = [200.0, 150.0, 90.0, 50.0]
        for i, score in enumerate(mixed_scores):
            agent = DDPGAgent(agent_id=i + 100)
            mixed_candidates.append((agent, score, i + 100))

        results3 = hof3.update_from_generation(mixed_candidates, generation=2)

        admitted3 = [r for r in results3 if r[2] == 'admitted']
        rejected3 = [r for r in results3 if r[2] == 'rejected_not_better']

        print(f"   New candidates: {mixed_scores}")
        print(f"   Admitted: {len(admitted3)}")
        print(f"   Rejected: {len(rejected3)}")

        final_scores3 = sorted([e.validation_score for e in hof3.entries])
        print(f"   Final HoF scores: {final_scores3}")

        assert len(hof3) == 5, f"HoF should be full, got {len(hof3)}"
        assert 200.0 in final_scores3 and 150.0 in final_scores3, "Top candidates should be in HoF"
        print("   [OK] Mixed scenario works correctly")

        print("\n" + "="*60)
        print("[SUCCESS] All batch admission tests passed!")
        print("="*60)

    finally:
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print(f"\n[OK] Cleaned up test directory")


if __name__ == "__main__":
    test_hall_of_fame()
    test_batch_admission()
