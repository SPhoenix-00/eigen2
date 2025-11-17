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


if __name__ == "__main__":
    test_hall_of_fame()
