"""
Checkpoint management for Project Eigen 2.

Handles all file I/O for training state persistence:
- Agent population save/load
- Replay buffer save/load
- Trainer state JSON save/load
- Hall of Fame save/load
- Cloud sync
- Gauntlet snapshots

ERLTrainer provides the state dict; CheckpointManager handles the disk.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Dict, Optional, Any, List

from training.fitness import NumpyEncoder


class CheckpointManager:
    """
    Manages checkpoint persistence for ERLTrainer.

    Separates file I/O concerns from training logic. ERLTrainer builds
    state dicts; CheckpointManager writes/reads them to/from disk.
    """

    def __init__(self, checkpoint_dir: Path, cloud_sync=None):
        """
        Args:
            checkpoint_dir: Path to checkpoint directory
            cloud_sync: Optional cloud sync instance for remote backup
        """
        self.checkpoint_dir = checkpoint_dir
        self.cloud_sync = cloud_sync

    # ── Agent I/O ────────────────────────────────────────────────────────

    def save_agents(self, population: list, best_agent=None):
        """Save population and best agent to checkpoint directory."""
        if best_agent is not None:
            best_path = self.checkpoint_dir / "best_agent.pth"
            best_agent.save(str(best_path))

        pop_dir = self.checkpoint_dir / "population"
        pop_dir.mkdir(exist_ok=True)
        for agent in population:
            agent_path = pop_dir / f"agent_{agent.agent_id}.pth"
            agent.save(str(agent_path))

    def load_agents(self, population: list) -> bool:
        """
        Load population agents from checkpoint directory.

        Args:
            population: List of DDPGAgent instances to load into (modifies in-place)

        Returns:
            True if population was loaded successfully
        """
        pop_dir = self.checkpoint_dir / "population"
        if not pop_dir.exists():
            return False

        try:
            for agent in population:
                agent_path = pop_dir / f"agent_{agent.agent_id}.pth"
                if agent_path.exists():
                    agent.load(str(agent_path))
            print(f"✓ Loaded {len(population)} agents")
            return True
        except Exception as e:
            print(f"❌ Error loading agents: {e}")
            return False

    def load_best_agent(self, agent_class, agent_id='best'):
        """
        Load best agent from checkpoint.

        Args:
            agent_class: DDPGAgent class to instantiate
            agent_id: Agent ID for the best agent

        Returns:
            Loaded agent or None
        """
        best_path = self.checkpoint_dir / "best_agent.pth"
        if not best_path.exists():
            return None
        try:
            agent = agent_class(agent_id=agent_id)
            agent.load(str(best_path))
            print("✓ Loaded best agent")
            return agent
        except Exception as e:
            print(f"❌ Error loading best agent: {e}")
            return None

    # ── Replay Buffer I/O ────────────────────────────────────────────────

    def save_buffer(self, replay_buffer, generation: int):
        """Save replay buffer metadata (every 5 generations)."""
        if (generation + 1) % 5 == 0:
            buffer_path = self.checkpoint_dir / "replay_buffer.pkl"
            print(f"  Saving on-disk buffer metadata ({len(replay_buffer)} paths)...")
            replay_buffer.save(str(buffer_path))

    def load_buffer(self, buffer_class, local_mode: bool, config):
        """
        Load or reconstruct replay buffer from checkpoint.

        Args:
            buffer_class: OnDiskReplayBuffer class
            local_mode: Whether running in local mode
            config: Config class with BUFFER_SIZE / LOCAL_BUFFER_SIZE

        Returns:
            Tuple of (replay_buffer, buffer_loaded: bool)
        """
        buffer_path = self.checkpoint_dir / "replay_buffer.pkl"
        buffer_storage_path = str(self.checkpoint_dir / "buffer_storage")
        target_capacity = config.LOCAL_BUFFER_SIZE if local_mode else config.BUFFER_SIZE

        if buffer_path.exists():
            try:
                replay_buffer = buffer_class.load(
                    str(buffer_path),
                    storage_path_override=buffer_storage_path
                )
                print(f"✓ Loaded on-disk replay buffer metadata ({len(replay_buffer)} transitions)")

                # Resize if config changed
                if replay_buffer.capacity != target_capacity:
                    print(f"⚠️ Resizing buffer: {replay_buffer.capacity:,} -> {target_capacity:,}")
                    from collections import deque
                    new_deque = deque(replay_buffer.buffer, maxlen=target_capacity)
                    replay_buffer.buffer = new_deque
                    replay_buffer.capacity = target_capacity
                    print(f"✓ Buffer resized ({len(replay_buffer):,} transitions preserved)")

                return replay_buffer, True
            except Exception as e:
                print(f"❌ Error loading buffer: {e}")
                print("  Creating new empty buffer...")
                return buffer_class(capacity=target_capacity, storage_path=buffer_storage_path), True

        # Try to reconstruct from transition files on disk
        storage_path_obj = Path(buffer_storage_path)
        if storage_path_obj.exists():
            transition_files = sorted(
                storage_path_obj.glob("transition_*.pkl.gz"),
                key=lambda p: int(p.stem.split('_')[1]) if p.stem.split('_')[1].isdigit() else 0
            )
            chunk_files = sorted(
                storage_path_obj.glob("chunk_*.pkl.gz"),
                key=lambda p: int(p.stem.split('_')[1]) if p.stem.split('_')[1].isdigit() else 0
            )
            all_files = list(transition_files) + list(chunk_files)

            if all_files:
                print(f"  Found {len(all_files)} existing transition files on disk.")
                print(f"  Reconstructing buffer metadata...")
                try:
                    replay_buffer = buffer_class(
                        capacity=target_capacity,
                        storage_path=buffer_storage_path
                    )
                    from collections import deque
                    files_to_add = all_files[-target_capacity:] if len(all_files) > target_capacity else all_files
                    file_paths = [str(f) for f in files_to_add]
                    replay_buffer.buffer = deque(file_paths, maxlen=target_capacity)

                    # Estimate transition count
                    total_transitions = 0
                    for fp in file_paths:
                        if 'chunk_' in fp:
                            total_transitions += 64  # Default estimate
                        else:
                            total_transitions += 1
                    replay_buffer.total_transitions = total_transitions
                    replay_buffer.total_added = len(all_files)

                    print(f"  ✓ Reconstructed: {len(replay_buffer.buffer)} files, ~{total_transitions} transitions")
                    return replay_buffer, True
                except Exception as e:
                    print(f"  ⚠️ Error reconstructing buffer: {e}")

        print("! No replay buffer checkpoint found.")
        return None, False

    # ── Trainer State I/O ────────────────────────────────────────────────

    def save_trainer_state(self, state_dict: Dict[str, Any]):
        """
        Save trainer state dict to JSON with atomic write.

        Args:
            state_dict: Complete trainer state dictionary
        """
        state_path = self.checkpoint_dir / "trainer_state.json"
        temp_path = self.checkpoint_dir / "trainer_state.tmp"
        try:
            with open(temp_path, 'w') as f:
                json.dump(state_dict, f, indent=4, cls=NumpyEncoder)
            temp_path.replace(state_path)
        except Exception as e:
            print(f"⚠ Failed to save trainer state: {e}")

    def load_trainer_state(self) -> Optional[Dict[str, Any]]:
        """
        Load trainer state dict from JSON.

        Returns:
            State dict or None if not found
        """
        state_path = self.checkpoint_dir / "trainer_state.json"
        if not state_path.exists():
            return None
        try:
            with open(state_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading state: {e}")
            return None

    # ── Hall of Fame I/O ────────────────────────────────────────────────

    def save_hof(self, hall_of_fame):
        """Save Hall of Fame if it has entries."""
        if hall_of_fame is not None and len(hall_of_fame) > 0:
            hall_of_fame.save()
            hof_stats = hall_of_fame.get_stats()
            print(f"  Saved Hall of Fame ({hof_stats['size']} champions)")

    def load_hof(self, hall_of_fame):
        """Load Hall of Fame from checkpoint."""
        try:
            hall_of_fame.load()
            hof_stats = hall_of_fame.get_stats()
            if hof_stats['size'] > 0:
                print(f"✓ Loaded Hall of Fame: {hof_stats['size']} champions")
                print(f"  Best HoF score: {hof_stats['best_score']:.2f}, "
                      f"Range: {hof_stats['worst_score']:.2f}-{hof_stats['best_score']:.2f}")
            else:
                print("! No Hall of Fame found (starting fresh)")
        except Exception as e:
            print(f"⚠ Could not load Hall of Fame: {e}")

    # ── Cloud Sync ───────────────────────────────────────────────────────

    def sync_to_cloud(self):
        """Sync checkpoint directory to cloud storage (non-blocking)."""
        if self.cloud_sync:
            self.cloud_sync.sync_checkpoints(
                str(self.checkpoint_dir), background=True,
                exclude_patterns=["buffer_storage"]
            )
            print(f"✓ Saved & syncing to cloud")

    def download_from_cloud(self):
        """Download checkpoint from cloud if local directory is empty."""
        if not self.checkpoint_dir.exists() or len(list(self.checkpoint_dir.glob('*'))) == 0:
            if self.cloud_sync:
                print("Downloading from cloud...")
                self.cloud_sync.download_checkpoints(str(self.checkpoint_dir))

    # ── Gauntlet Snapshots ───────────────────────────────────────────────

    def save_gauntlet_snapshot(self, population: list, state_dict: Dict[str, Any],
                               hall_of_fame=None, generation: int = 0):
        """
        Save a snapshot of trainer state before entering the Gauntlet.

        Args:
            population: List of agents to snapshot
            state_dict: Trainer state dict (from _build_gauntlet_snapshot_dict)
            hall_of_fame: Optional HoF to snapshot
            generation: Current generation number
        """
        try:
            snapshot_dir = self.checkpoint_dir / "gauntlet_snapshot"
            snapshot_dir.mkdir(parents=True, exist_ok=True)

            print(f"\n📸 Saving Gauntlet Snapshot (Gen {generation})...")

            # 1. Save population
            pop_dir = snapshot_dir / "population"
            pop_dir.mkdir(exist_ok=True)
            for agent in population:
                agent_path = pop_dir / f"agent_{agent.agent_id}.pth"
                agent.save(str(agent_path))

            # 2. Save trainer state
            state_path = snapshot_dir / "trainer_state.json"
            with open(state_path, 'w') as f:
                json.dump(state_dict, f, indent=4, cls=NumpyEncoder)

            # 3. Save Hall of Fame
            if hall_of_fame is not None and len(hall_of_fame) > 0:
                hof_snapshot_path = snapshot_dir / "hall_of_fame.json"
                hof_source = hall_of_fame.hof_dir / "hall_of_fame.json"
                if hof_source.exists():
                    shutil.copy(hof_source, hof_snapshot_path)

                hof_agents_dir = snapshot_dir / "hof_agents"
                hof_agents_dir.mkdir(exist_ok=True)
                for entry in hall_of_fame.entries:
                    agent_path = hall_of_fame.hof_dir / f"hof_agent_{entry.agent_id}.pth"
                    if agent_path.exists():
                        shutil.copy(agent_path, hof_agents_dir / agent_path.name)

            print(f"✓ Snapshot saved to {snapshot_dir}")
        except Exception as e:
            print(f"⚠ FAILED TO SAVE GAUNTLET SNAPSHOT: {e}")
            print(f"  Continuing training anyway...")

    def load_gauntlet_snapshot_state(self) -> Optional[Dict[str, Any]]:
        """Load gauntlet snapshot trainer state."""
        snapshot_dir = self.checkpoint_dir / "gauntlet_snapshot"
        state_path = snapshot_dir / "trainer_state.json"
        if not state_path.exists():
            return None
        try:
            with open(state_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"❌ Error loading snapshot state: {e}")
            return None

    def load_gauntlet_snapshot_agents(self, population: list) -> bool:
        """Load population agents from gauntlet snapshot."""
        snapshot_dir = self.checkpoint_dir / "gauntlet_snapshot"
        pop_dir = snapshot_dir / "population"
        if not pop_dir.exists():
            return False
        try:
            for agent in population:
                agent_path = pop_dir / f"agent_{agent.agent_id}.pth"
                if agent_path.exists():
                    agent.load(str(agent_path))
            print(f"✓ Restored {len(population)} agents to pre-Gauntlet state")
            return True
        except Exception as e:
            print(f"❌ Error restoring population: {e}")
            return False

    def restore_gauntlet_hof(self, hall_of_fame) -> bool:
        """Restore Hall of Fame from gauntlet snapshot."""
        snapshot_dir = self.checkpoint_dir / "gauntlet_snapshot"
        hof_snapshot_path = snapshot_dir / "hall_of_fame.json"
        hof_agents_dir = snapshot_dir / "hof_agents"

        if not hof_snapshot_path.exists() or not hof_agents_dir.exists():
            return False

        try:
            hof_dest = hall_of_fame.hof_dir / "hall_of_fame.json"
            shutil.copy(hof_snapshot_path, hof_dest)

            for agent_file in hof_agents_dir.glob("hof_agent_*.pth"):
                shutil.copy(agent_file, hall_of_fame.hof_dir / agent_file.name)

            hall_of_fame.load()
            print(f"✓ Restored Hall of Fame ({len(hall_of_fame.entries)} champions)")
            return True
        except Exception as e:
            print(f"❌ Error restoring Hall of Fame: {e}")
            return False

    # ── Convenience ──────────────────────────────────────────────────────

    def write_last_run_file(self, run_name: str, run_id: str, project: str = 'eigen2-self'):
        """Write last_run.json to root directory for easy resume."""
        last_run_info = {
            'run_name': run_name,
            'run_id': run_id,
            'project': project,
            'timestamp': time.time()
        }
        last_run_file = Path("last_run.json")
        try:
            with open(last_run_file, 'w') as f:
                json.dump(last_run_info, f, indent=2)
            print(f"✓ Wrote run info to {last_run_file}")
        except Exception as e:
            print(f"⚠ Could not write last_run.json: {e}")

    def checkpoint_exists(self) -> bool:
        """Check if checkpoint directory exists and has content."""
        return self.checkpoint_dir.exists() and len(list(self.checkpoint_dir.glob('*'))) > 0

    def gauntlet_snapshot_exists(self) -> bool:
        """Check if gauntlet snapshot exists."""
        return (self.checkpoint_dir / "gauntlet_snapshot").exists()
