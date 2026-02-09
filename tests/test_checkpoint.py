"""
Unit tests for training/checkpoint.py — CheckpointManager.
"""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from training.checkpoint import CheckpointManager


class TestCheckpointManager:
    def test_init(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "checkpoints")
        assert mgr.checkpoint_dir == tmp_path / "checkpoints"
        assert mgr.cloud_sync is None

    def test_checkpoint_exists_false(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "nonexistent")
        assert mgr.checkpoint_exists() is False

    def test_checkpoint_exists_true(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        (ckpt_dir / "dummy.txt").write_text("test")
        mgr = CheckpointManager(ckpt_dir)
        assert mgr.checkpoint_exists() is True

    def test_gauntlet_snapshot_exists(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        mgr = CheckpointManager(ckpt_dir)
        assert mgr.gauntlet_snapshot_exists() is False

        (ckpt_dir / "gauntlet_snapshot").mkdir()
        assert mgr.gauntlet_snapshot_exists() is True


class TestTrainerStateIO:
    def test_save_and_load_round_trip(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        mgr = CheckpointManager(ckpt_dir)

        state = {
            'generation': 42,
            'best_fitness': 100.5,
            'best_validation_fitness': 95.3,
            'consistency_mode': True,
            'normalization_stats': {
                'mean': [0.0, 1.0, 2.0],
                'std': [1.0, 1.0, 1.0]
            }
        }
        mgr.save_trainer_state(state)

        loaded = mgr.load_trainer_state()
        assert loaded is not None
        assert loaded['generation'] == 42
        assert loaded['best_fitness'] == 100.5
        assert loaded['consistency_mode'] is True

    def test_load_nonexistent(self, tmp_path):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        mgr = CheckpointManager(ckpt_dir)
        assert mgr.load_trainer_state() is None

    def test_atomic_write(self, tmp_path):
        """Verify no .tmp file is left behind after save."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        mgr = CheckpointManager(ckpt_dir)

        mgr.save_trainer_state({'generation': 1})
        assert (ckpt_dir / "trainer_state.json").exists()
        assert not (ckpt_dir / "trainer_state.tmp").exists()


class TestLastRunFile:
    def test_write_last_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        mgr = CheckpointManager(tmp_path / "ckpt")

        mgr.write_last_run_file("azure-thunder-42", "abc123")

        last_run = tmp_path / "last_run.json"
        assert last_run.exists()
        data = json.loads(last_run.read_text())
        assert data['run_name'] == "azure-thunder-42"
        assert data['run_id'] == "abc123"
        assert data['project'] == "eigen2-self"
        assert 'timestamp' in data


class TestAgentIO:
    def test_save_and_load_agents(self, tmp_path, mock_agent):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        mgr = CheckpointManager(ckpt_dir)

        # Save
        population = [mock_agent]
        mgr.save_agents(population, best_agent=mock_agent)

        assert (ckpt_dir / "best_agent.pth").exists()
        assert (ckpt_dir / "population" / f"agent_{mock_agent.agent_id}.pth").exists()

        # Load
        from models.ddpg_agent import DDPGAgent
        new_population = [DDPGAgent(agent_id=0)]
        success = mgr.load_agents(new_population)
        assert success is True

    def test_load_agents_no_dir(self, tmp_path):
        mgr = CheckpointManager(tmp_path / "checkpoints")
        assert mgr.load_agents([]) is False

    def test_load_best_agent(self, tmp_path, mock_agent):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        mgr = CheckpointManager(ckpt_dir)

        # Save best agent
        mgr.save_agents([], best_agent=mock_agent)

        # Load
        from models.ddpg_agent import DDPGAgent
        loaded = mgr.load_best_agent(DDPGAgent)
        assert loaded is not None


class TestGauntletSnapshotIO:
    def test_save_and_load_snapshot(self, tmp_path, mock_agent):
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        mgr = CheckpointManager(ckpt_dir)

        state_dict = {
            'generation': 10,
            'best_fitness': 200.0,
            'breakthrough_state': 'normal',
            'confirmed_baseline': 50.0,
        }

        mgr.save_gauntlet_snapshot(
            population=[mock_agent],
            state_dict=state_dict,
            generation=10,
        )

        assert mgr.gauntlet_snapshot_exists()

        # Load state
        loaded_state = mgr.load_gauntlet_snapshot_state()
        assert loaded_state is not None
        assert loaded_state['generation'] == 10
        assert loaded_state['best_fitness'] == 200.0

        # Load agents
        from models.ddpg_agent import DDPGAgent
        new_pop = [DDPGAgent(agent_id=0)]
        success = mgr.load_gauntlet_snapshot_agents(new_pop)
        assert success is True


class TestCloudSync:
    def test_sync_with_no_cloud(self, tmp_path):
        """Should not raise if cloud_sync is None."""
        mgr = CheckpointManager(tmp_path, cloud_sync=None)
        mgr.sync_to_cloud()  # Should be a no-op

    def test_sync_calls_cloud(self, tmp_path):
        mock_sync = MagicMock()
        mgr = CheckpointManager(tmp_path, cloud_sync=mock_sync)
        mgr.sync_to_cloud()
        mock_sync.sync_checkpoints.assert_called_once()

    def test_download_calls_cloud(self, tmp_path):
        mock_sync = MagicMock()
        ckpt_dir = tmp_path / "checkpoints"  # doesn't exist
        mgr = CheckpointManager(ckpt_dir, cloud_sync=mock_sync)
        mgr.download_from_cloud()
        mock_sync.download_checkpoints.assert_called_once()
