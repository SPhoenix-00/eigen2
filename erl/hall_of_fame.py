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

    def __init__(self, agent_id: int, validation_score: float, generation: int, roi: float = 0.0):
        """
        Initialize a Hall of Fame entry.

        Args:
            agent_id: Unique identifier for this HoF entry
            validation_score: Validation fitness that qualified this agent
            generation: Generation number when this agent was admitted
            roi: Return on Investment percentage for this agent
        """
        self.agent_id = agent_id
        self.validation_score = validation_score
        self.generation = generation
        self.roi = roi

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'agent_id': self.agent_id,
            'validation_score': float(self.validation_score),
            'generation': self.generation,
            'roi': float(self.roi)
        }

    @staticmethod
    def from_dict(data: dict) -> 'HallOfFameEntry':
        """Create entry from dictionary."""
        return HallOfFameEntry(
            agent_id=data['agent_id'],
            validation_score=data['validation_score'],
            generation=data['generation'],
            roi=data.get('roi', 0.0)  # Default for backwards compatibility
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

    def update_from_generation(self, candidates: List[Tuple[DDPGAgent, float, int, float]], generation: int) -> List[Tuple[int, float, str]]:
        """
        Update Hall of Fame from a generation of candidates with aggressive admission.

        Implements two-phase admission:
        1. Initial Filling Phase (HoF < capacity): Admit all candidates with Combined > 0
        2. Maintenance Phase: Cascading swaps - replace worst HoF agents with best candidates

        Args:
            candidates: List of (agent, combined_score, agent_idx, roi) tuples
            generation: Current generation number

        Returns:
            List of (agent_idx, score, action) tuples describing what happened
            action is one of: 'admitted', 'replaced_agent_X', 'rejected'
        """
        results = []

        if not candidates:
            return results

        # Sort candidates by combined score (descending - best first)
        sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

        # Phase 1: Initial Filling - admit all positive-score candidates if not full
        if not self.is_full():
            for agent, score, agent_idx, roi in sorted_candidates:
                if self.is_full():
                    break

                # Only admit agents with positive Combined score during filling phase
                if score > 0:
                    new_id = self._get_next_id()
                    entry = HallOfFameEntry(
                        agent_id=new_id,
                        validation_score=score,
                        generation=generation,
                        roi=roi
                    )
                    self.entries.append(entry)

                    # Save agent weights
                    if self.checkpoint_dir:
                        agent_path = self.hof_dir / f"hof_agent_{new_id}.pth"
                        agent.save(str(agent_path))

                    results.append((agent_idx, score, 'admitted'))
                else:
                    results.append((agent_idx, score, 'rejected_negative'))

        # Phase 2: Cascading Swaps - even if we just filled some slots, check for replacements
        # Get remaining candidates that weren't admitted in Phase 1
        admitted_indices = {r[0] for r in results if r[2] == 'admitted'}
        remaining_candidates = [(a, s, idx, roi) for a, s, idx, roi in sorted_candidates
                                if idx not in admitted_indices]

        if remaining_candidates and self.is_full():
            # Sort HoF entries by score (ascending - worst first)
            sorted_hof = sorted(self.entries, key=lambda e: e.validation_score)

            # Cascading swap: iterate through worst HoF agents and best candidates
            swaps_made = 0
            for candidate_agent, candidate_score, agent_idx, candidate_roi in remaining_candidates:
                if swaps_made >= len(sorted_hof):
                    break

                # Find the current worst HoF agent (accounting for previous swaps)
                current_hof_sorted = sorted(self.entries, key=lambda e: e.validation_score)
                if not current_hof_sorted:
                    break

                worst_entry = current_hof_sorted[0]

                # Check if candidate beats the worst HoF agent
                if candidate_score > worst_entry.validation_score:
                    old_score = worst_entry.validation_score
                    old_id = worst_entry.agent_id

                    # Remove worst entry
                    self.entries.remove(worst_entry)

                    # Delete old agent file
                    if self.checkpoint_dir:
                        old_path = self.hof_dir / f"hof_agent_{old_id}.pth"
                        if old_path.exists():
                            old_path.unlink()

                    # Add new entry
                    new_id = self._get_next_id()
                    new_entry = HallOfFameEntry(
                        agent_id=new_id,
                        validation_score=candidate_score,
                        generation=generation,
                        roi=candidate_roi
                    )
                    self.entries.append(new_entry)

                    # Save new agent weights
                    if self.checkpoint_dir:
                        agent_path = self.hof_dir / f"hof_agent_{new_id}.pth"
                        candidate_agent.save(str(agent_path))

                    results.append((agent_idx, candidate_score, f'replaced_{old_score:.2f}'))
                    swaps_made += 1
                else:
                    results.append((agent_idx, candidate_score, 'rejected_not_better'))

        return results

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

    def get_median_roi(self) -> float:
        """
        Get the median ROI of all agents in the Hall of Fame.

        Returns:
            Median ROI percentage, or 0.0 if Hall is empty
        """
        if len(self.entries) == 0:
            return 0.0
        roi_values = [e.roi for e in self.entries]
        return float(np.median(roi_values))

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
                'newest_generation': 0,
                'median_roi': 0.0
            }

        scores = [e.validation_score for e in self.entries]
        generations = [e.generation for e in self.entries]
        roi_values = [e.roi for e in self.entries]

        return {
            'size': len(self.entries),
            'best_score': max(scores),
            'worst_score': min(scores),
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'oldest_generation': min(generations),
            'newest_generation': max(generations),
            'median_roi': float(np.median(roi_values))
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
