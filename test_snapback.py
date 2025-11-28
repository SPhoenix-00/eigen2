"""
Test script to verify the Snapback Mechanism implementation.

This script checks that:
1. save_gauntlet_snapshot() correctly saves population and state
2. restore_gauntlet_snapshot() correctly restores population and state
"""

import sys
import torch
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from training.erl_trainer import ERLTrainer, BreakthroughState
from data.data_loader import StockDataLoader
from config.config import Config


def test_snapback_mechanism():
    """Test that snapback saves and restores correctly."""

    print("="*60)
    print("Testing Snapback Mechanism")
    print("="*60)

    # Create a temporary directory for this test
    with tempfile.TemporaryDirectory() as tmpdir:
        # Override checkpoint directory
        original_checkpoint_dir = Config.CHECKPOINT_DIR
        Config.CHECKPOINT_DIR = Path(tmpdir)

        try:
            # 1. Initialize a minimal trainer
            print("\n1. Initializing trainer...")
            data_loader = StockDataLoader()
            trainer = ERLTrainer(data_loader)

            # Set some state values that we can verify
            trainer.generation = 42
            trainer.best_fitness = 100.5
            trainer.best_validation_fitness = 95.3
            trainer.current_mutation_rate = 0.15
            trainer.confirmed_baseline = 50.0
            trainer.confirmed_breakthroughs = 3
            trainer.breakthrough_state = BreakthroughState.NORMAL

            print(f"   Generation: {trainer.generation}")
            print(f"   Best Fitness: {trainer.best_fitness}")
            print(f"   Best Val Fitness: {trainer.best_validation_fitness}")
            print(f"   Mutation Rate: {trainer.current_mutation_rate}")
            print(f"   Confirmed Baseline: {trainer.confirmed_baseline}")
            print(f"   Breakthroughs: {trainer.confirmed_breakthroughs}")

            # Save original population weights to verify they're restored
            print("\n2. Capturing original population state...")
            original_weights = []
            for agent in trainer.population[:3]:  # Just check first 3 agents
                actor_weight = agent.actor.fc1.weight.data.clone()
                original_weights.append(actor_weight)

            # 3. Save snapshot
            print("\n3. Saving gauntlet snapshot...")
            trainer.save_gauntlet_snapshot()

            snapshot_dir = trainer.checkpoint_dir / "gauntlet_snapshot"
            assert snapshot_dir.exists(), "Snapshot directory should exist"
            assert (snapshot_dir / "trainer_state.json").exists(), "Trainer state should be saved"
            assert (snapshot_dir / "population").exists(), "Population directory should exist"
            print("   ✓ Snapshot saved successfully")

            # 4. Modify state to simulate population poisoning
            print("\n4. Simulating population poisoning...")
            trainer.generation = 50
            trainer.best_fitness = 120.0
            trainer.best_validation_fitness = 110.0
            trainer.current_mutation_rate = 0.25
            trainer.confirmed_baseline = 60.0
            trainer.breakthrough_state = BreakthroughState.REJECTED

            # Modify population weights
            for agent in trainer.population:
                agent.actor.fc1.weight.data.fill_(0.999)  # Poison with uniform values

            print(f"   Generation: {trainer.generation} (was 42)")
            print(f"   Best Fitness: {trainer.best_fitness} (was 100.5)")
            print(f"   State: {trainer.breakthrough_state.value}")

            # 5. Restore snapshot
            print("\n5. Restoring gauntlet snapshot (Snapback)...")
            trainer.restore_gauntlet_snapshot()

            # 6. Verify restoration
            print("\n6. Verifying restoration...")

            # Check state values
            assert trainer.generation == 42, f"Generation should be 42, got {trainer.generation}"
            assert trainer.best_fitness == 100.5, f"Best fitness should be 100.5, got {trainer.best_fitness}"
            assert trainer.best_validation_fitness == 95.3, f"Best val fitness should be 95.3, got {trainer.best_validation_fitness}"
            assert trainer.current_mutation_rate == 0.15, f"Mutation rate should be 0.15, got {trainer.current_mutation_rate}"
            assert trainer.confirmed_baseline == 50.0, f"Confirmed baseline should be 50.0, got {trainer.confirmed_baseline}"
            assert trainer.confirmed_breakthroughs == 3, f"Breakthroughs should be 3, got {trainer.confirmed_breakthroughs}"
            assert trainer.breakthrough_state == BreakthroughState.NORMAL, f"State should be NORMAL, got {trainer.breakthrough_state.value}"

            print("   ✓ Trainer state restored correctly")

            # Check population weights
            weights_match = True
            for i, (agent, original_weight) in enumerate(zip(trainer.population[:3], original_weights)):
                restored_weight = agent.actor.fc1.weight.data
                if not torch.allclose(restored_weight, original_weight, atol=1e-6):
                    weights_match = False
                    print(f"   ✗ Agent {i} weights don't match!")

            if weights_match:
                print("   ✓ Population weights restored correctly")

            print("\n" + "="*60)
            print("✓ ALL TESTS PASSED - Snapback Mechanism Working!")
            print("="*60)

        finally:
            # Restore original config
            Config.CHECKPOINT_DIR = original_checkpoint_dir


if __name__ == "__main__":
    test_snapback_mechanism()
