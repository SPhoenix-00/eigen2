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

    def __init__(self, agent_id: int, validation_score: float, generation: int, roi: float = 0.0, expectancy: float = 0.0,
                 train_fitness: float = 0.0, quality_count: int = 0, total_trades: int = 0,
                 val_fitness: float = 0.0, base_combined_fitness: float = 0.0):
        """
        Initialize a Hall of Fame entry.

        Args:
            agent_id: Unique identifier for this HoF entry
            validation_score: Combined fitness (with ROI adjustment) that qualified this agent
            generation: Generation number when this agent was admitted
            roi: Return on Investment percentage for this agent
            expectancy: Expectancy metric for this agent
            train_fitness: Training fitness score (for re-computing combined score)
            quality_count: Number of quality trades (for re-computing confidence factor)
            total_trades: Total trades across all slices (for re-computing confidence factor)
            val_fitness: Raw validation fitness before ROI adjustment
            base_combined_fitness: Base combined fitness before ROI adjustment
        """
        self.agent_id = agent_id
        self.validation_score = validation_score  # This is the combined_fitness (with ROI adjustment)
        self.generation = generation
        self.roi = roi
        self.expectancy = expectancy
        # Store raw metrics for re-evaluation with EMA erosion
        self.train_fitness = train_fitness
        self.quality_count = quality_count
        self.total_trades = total_trades
        self.val_fitness = val_fitness
        self.base_combined_fitness = base_combined_fitness

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'agent_id': self.agent_id,
            'validation_score': float(self.validation_score),
            'generation': self.generation,
            'roi': float(self.roi),
            'expectancy': float(self.expectancy),
            'train_fitness': float(self.train_fitness),
            'quality_count': int(self.quality_count),
            'total_trades': int(self.total_trades),
            'val_fitness': float(self.val_fitness),
            'base_combined_fitness': float(self.base_combined_fitness)
        }

    @staticmethod
    def from_dict(data: dict) -> 'HallOfFameEntry':
        """Create entry from dictionary."""
        return HallOfFameEntry(
            agent_id=data['agent_id'],
            validation_score=data['validation_score'],
            generation=data['generation'],
            roi=data.get('roi', 0.0),  # Default for backwards compatibility
            expectancy=data.get('expectancy', 0.0),  # Default for backwards compatibility
            train_fitness=data.get('train_fitness', 0.0),
            quality_count=data.get('quality_count', 0),
            total_trades=data.get('total_trades', 0),
            val_fitness=data.get('val_fitness', 0.0),
            base_combined_fitness=data.get('base_combined_fitness', 0.0)
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

    def update_from_generation(self, candidates: List[Tuple[DDPGAgent, float, int, float, float, float, int, int, float, float]], generation: int) -> List[Tuple[int, float, str]]:
        """
        Update Hall of Fame from a generation of candidates with aggressive admission.

        Implements two-phase admission:
        1. Initial Filling Phase (HoF < capacity): Admit all candidates with Combined > 0
        2. Maintenance Phase: Cascading swaps - replace worst HoF agents with best candidates

        Args:
            candidates: List of (agent, combined_score, agent_idx, roi, expectancy,
                       train_fitness, quality_count, total_trades, val_fitness, base_combined_fitness) tuples
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
            for agent, score, agent_idx, roi, expectancy, train_fitness, quality_count, total_trades, val_fitness, base_combined_fitness in sorted_candidates:
                if self.is_full():
                    break

                # Only admit agents with positive Combined score during filling phase
                if score > 0:
                    new_id = self._get_next_id()
                    entry = HallOfFameEntry(
                        agent_id=new_id,
                        validation_score=score,
                        generation=generation,
                        roi=roi,
                        expectancy=expectancy,
                        train_fitness=train_fitness,
                        quality_count=quality_count,
                        total_trades=total_trades,
                        val_fitness=val_fitness,
                        base_combined_fitness=base_combined_fitness
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
        remaining_candidates = [(a, s, idx, roi, exp, tf, qc, tt, vf, bcf) for a, s, idx, roi, exp, tf, qc, tt, vf, bcf in sorted_candidates
                                if idx not in admitted_indices]

        if remaining_candidates and self.is_full():
            # Sort HoF entries by score (ascending - worst first)
            sorted_hof = sorted(self.entries, key=lambda e: e.validation_score)

            # Cascading swap: iterate through worst HoF agents and best candidates
            swaps_made = 0
            for candidate_agent, candidate_score, agent_idx, candidate_roi, candidate_expectancy, candidate_train_fitness, candidate_quality_count, candidate_total_trades, candidate_val_fitness, candidate_base_combined in remaining_candidates:
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
                        roi=candidate_roi,
                        expectancy=candidate_expectancy,
                        train_fitness=candidate_train_fitness,
                        quality_count=candidate_quality_count,
                        total_trades=candidate_total_trades,
                        val_fitness=candidate_val_fitness,
                        base_combined_fitness=candidate_base_combined
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
            List of k cloned DDPGAgent instances with fresh optimizers

        Note:
            Uses load_weights_only() to avoid loading stale optimizer states.
            This prevents 60% training slowdown caused by mismatched Adam
            momentum from previous training contexts.
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
                # Create new agent with fresh optimizers
                agent = DDPGAgent(agent_id=entry.agent_id)
                # Load only network weights, not optimizer states
                agent.load_weights_only(str(agent_path))
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

    def recompute_all_scores(self, current_median_roi: float, roi_adjustment_multiplier: float,
                            min_trades_threshold: int, erosion_alpha: float = 0.33,
                            consistency_mode: bool = False) -> int:
        """
        Re-evaluate HoF entries' combined fitness using the current median ROI with gradual erosion.

        This implements the "erosion" mechanism: as the median rises over time,
        historical agents' relative ROI advantage diminishes, and their combined
        fitness scores are adjusted accordingly. This ensures newer agents compete
        on equal footing and prevents early agents from having unfair advantages.

        Erosion is gradual using an EMA (Exponential Moving Average) approach:
        - Each generation, we compute what the score SHOULD be with current median
        - We blend the old score with the new target using EMA: score = (1-α)*old + α*new
        - Default α=0.33 means full adjustment takes ~3 generations (convergence at ~95% in 3 steps)

        This prevents sudden HoF turnover and reduces noise while still implementing erosion.

        Args:
            current_median_roi: Current HoF median ROI to use as benchmark
            roi_adjustment_multiplier: Multiplier for ROI adjustment (Config.ROI_ADJUSTMENT_MULTIPLIER)
            min_trades_threshold: Minimum quality trades for full confidence (Config.ROI_CONFIDENCE_MIN_TRADES)
            erosion_alpha: EMA smoothing factor (0-1). Higher = faster erosion. Default 0.33 ≈ 3 generations
            consistency_mode: If True, skip ROI adjustment (consistency mode)

        Returns:
            Number of entries that had their scores updated
        """
        if len(self.entries) == 0:
            return 0

        updated_count = 0

        for entry in self.entries:
            # Store original score for comparison
            old_score = entry.validation_score

            # Compute target combined fitness using stored raw metrics and CURRENT median
            if consistency_mode:
                # Consistency mode: validation fitness only (no ROI adjustment)
                target_combined_fitness = entry.val_fitness
            else:
                # Normal mode: recalculate with current median
                # Base combined: val_fitness + min(0, train_fitness)
                base_combined = entry.val_fitness + min(0.0, entry.train_fitness)

                # Confidence factor based on quality trades
                confidence_factor = min(1.0, entry.quality_count / min_trades_threshold) if min_trades_threshold > 0 else 0.0

                # ROI adjustment: (|base| × multiplier × (agent_roi - current_median) / 100) × confidence
                roi_adjustment = abs(base_combined) * roi_adjustment_multiplier * (entry.roi - current_median_roi) / 100.0
                roi_adjustment = roi_adjustment * confidence_factor

                target_combined_fitness = base_combined + roi_adjustment

            # Apply EMA smoothing: blend old score toward target
            # new_score = (1 - alpha) * old_score + alpha * target_score
            # This creates gradual erosion over multiple generations
            new_combined_fitness = (1.0 - erosion_alpha) * old_score + erosion_alpha * target_combined_fitness

            # Update the validation_score (which represents combined fitness)
            entry.validation_score = new_combined_fitness

            if abs(new_combined_fitness - old_score) > 0.01:  # Track meaningful changes
                updated_count += 1

        return updated_count

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
            'best_score': float(max(scores)),
            'worst_score': float(min(scores)),
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
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
