"""
Hall of Fame - Active Archive for Evolutionary RL
Stores and manages the best agents based on validation scores across all generations.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict
from models.ddpg_agent import DDPGAgent


class HallOfFameEntry:
    """
    A single entry in the Hall of Fame.

    Stores the agent's validation score, generation of admission,
    and the path to the agent's saved weights.
    """

    def __init__(self, agent_id: int, validation_score: float, generation: int):
        """
        Initialize a Hall of Fame entry.

        Args:
            agent_id: Unique identifier for this HoF entry
            validation_score: Validation fitness that qualified this agent
            generation: Generation number when this agent was admitted
        """
        self.agent_id = agent_id
        self.validation_score = validation_score
        self.generation = generation

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'agent_id': self.agent_id,
            'validation_score': self.validation_score,
            'generation': self.generation
        }

    @staticmethod
    def from_dict(data: dict) -> 'HallOfFameEntry':
        """Create entry from dictionary."""
        return HallOfFameEntry(
            agent_id=data['agent_id'],
            validation_score=data['validation_score'],
            generation=data['generation']
        )


class HallOfFame:
    """
    Hall of Fame - Active archive of the best agents by validation score.

    Maintains a persistent list of top N agents. When full, new entries
    replace the worst agent if they have a higher validation score.
    """

    def __init__(self, capacity: int = 10, checkpoint_dir: Optional[Path] = None):
        """
        Initialize Hall of Fame.

        Args:
            capacity: Maximum number of agents to store (default: 10)
            checkpoint_dir: Directory to save HoF data and agent weights
        """
        self.capacity = capacity
        self.checkpoint_dir = checkpoint_dir
        self.entries: List[HallOfFameEntry] = []

        # Create HoF subdirectory if checkpoint_dir provided
        if self.checkpoint_dir:
            self.hof_dir = self.checkpoint_dir / "hall_of_fame"
            self.hof_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        """Return current number of entries."""
        return len(self.entries)

    def is_full(self) -> bool:
        """Check if Hall of Fame is at capacity."""
        return len(self.entries) >= self.capacity

    def get_worst_score(self) -> float:
        """
        Get the validation score of the worst agent in the Hall.

        Returns:
            Worst validation score, or -infinity if Hall is empty
        """
        if len(self.entries) == 0:
            return float('-inf')
        return min(entry.validation_score for entry in self.entries)

    def should_admit(self, validation_score: float) -> bool:
        """
        Check if an agent with given validation score should be admitted.

        Args:
            validation_score: Candidate agent's validation fitness

        Returns:
            True if agent qualifies for admission, False otherwise
        """
        if not self.is_full():
            return True
        return validation_score > self.get_worst_score()

    def add(self, agent: DDPGAgent, validation_score: float, generation: int) -> bool:
        """
        Add an agent to the Hall of Fame.

        If Hall is full, replaces the worst agent if the new agent is better.

        Args:
            agent: DDPGAgent to add
            validation_score: Validation fitness score
            generation: Current generation number

        Returns:
            True if agent was added, False otherwise
        """
        if not self.should_admit(validation_score):
            return False

        # If full, remove the worst entry
        if self.is_full():
            worst_entry = min(self.entries, key=lambda e: e.validation_score)
            self.entries.remove(worst_entry)

            # Delete the old agent file
            if self.checkpoint_dir:
                old_path = self.hof_dir / f"hof_agent_{worst_entry.agent_id}.pth"
                if old_path.exists():
                    old_path.unlink()

        # Create new entry with unique ID
        new_id = self._get_next_id()
        entry = HallOfFameEntry(
            agent_id=new_id,
            validation_score=validation_score,
            generation=generation
        )
        self.entries.append(entry)

        # Save agent weights
        if self.checkpoint_dir:
            agent_path = self.hof_dir / f"hof_agent_{new_id}.pth"
            agent.save(str(agent_path))

        return True

    def _get_next_id(self) -> int:
        """Generate next unique agent ID for Hall of Fame."""
        if len(self.entries) == 0:
            return 0
        return max(entry.agent_id for entry in self.entries) + 1

    def sample_random(self, k: int = 1) -> List[DDPGAgent]:
        """
        Sample k random agents from the Hall of Fame.

        Args:
            k: Number of agents to sample

        Returns:
            List of k cloned DDPGAgent instances
        """
        if len(self.entries) == 0:
            return []

        # Sample without replacement (or with if k > len)
        sample_size = min(k, len(self.entries))
        sampled_entries = np.random.choice(self.entries, size=sample_size, replace=False)

        agents = []
        for entry in sampled_entries:
            agent_path = self.hof_dir / f"hof_agent_{entry.agent_id}.pth"
            if agent_path.exists():
                # Create new agent and load weights
                agent = DDPGAgent(agent_id=entry.agent_id)
                agent.load(str(agent_path))
                agents.append(agent)

        return agents

    def get_best(self) -> Optional[Tuple[DDPGAgent, float]]:
        """
        Get the best agent from the Hall of Fame.

        Returns:
            Tuple of (agent, validation_score), or None if Hall is empty
        """
        if len(self.entries) == 0:
            return None

        best_entry = max(self.entries, key=lambda e: e.validation_score)
        agent_path = self.hof_dir / f"hof_agent_{best_entry.agent_id}.pth"

        if agent_path.exists():
            agent = DDPGAgent(agent_id=best_entry.agent_id)
            agent.load(str(agent_path))
            return agent, best_entry.validation_score

        return None

    def get_top_k(self, k: int = 5) -> List[DDPGAgent]:
        """
        Get the top k agents from the Hall of Fame sorted by validation score.

        Args:
            k: Number of top agents to retrieve

        Returns:
            List of DDPGAgent instances (best first)
        """
        if len(self.entries) == 0:
            return []

        # Sort entries by validation score (descending)
        sorted_entries = sorted(self.entries, key=lambda e: e.validation_score, reverse=True)

        # Take top k
        top_entries = sorted_entries[:min(k, len(sorted_entries))]

        agents = []
        for entry in top_entries:
            agent_path = self.hof_dir / f"hof_agent_{entry.agent_id}.pth"
            if agent_path.exists():
                agent = DDPGAgent(agent_id=entry.agent_id)
                agent.load(str(agent_path))
                agents.append(agent)

        return agents

    def get_stats(self) -> Dict:
        """
        Get Hall of Fame statistics.

        Returns:
            Dictionary with HoF statistics
        """
        if len(self.entries) == 0:
            return {
                'size': 0,
                'best_score': 0.0,
                'worst_score': 0.0,
                'mean_score': 0.0,
                'std_score': 0.0,
                'oldest_generation': 0,
                'newest_generation': 0
            }

        scores = [e.validation_score for e in self.entries]
        generations = [e.generation for e in self.entries]

        return {
            'size': len(self.entries),
            'best_score': max(scores),
            'worst_score': min(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'oldest_generation': min(generations),
            'newest_generation': max(generations)
        }

    def save(self):
        """Save Hall of Fame metadata to disk."""
        if not self.checkpoint_dir:
            return

        metadata = {
            'capacity': self.capacity,
            'entries': [entry.to_dict() for entry in self.entries]
        }

        metadata_path = self.hof_dir / "hall_of_fame.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

    def load(self):
        """Load Hall of Fame metadata from disk."""
        if not self.checkpoint_dir:
            return

        metadata_path = self.hof_dir / "hall_of_fame.json"
        if not metadata_path.exists():
            return

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.capacity = metadata['capacity']
        self.entries = [HallOfFameEntry.from_dict(e) for e in metadata['entries']]
