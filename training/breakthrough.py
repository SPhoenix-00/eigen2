"""
Gauntlet state machine types and breakthrough tracking for Project Eigen 2.

Contains the BreakthroughState enum, BreakthroughCandidate dataclass,
and BreakthroughTracker class that encapsulates all breakthrough-related state.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, List, Dict, Any


class BreakthroughState(Enum):
    """
    State machine for Gauntlet Mode breakthrough validation.

    Transitions:
        NORMAL -> DETECTION (spike detected)
        DETECTION -> STABILIZATION (candidate locked in)
        STABILIZATION -> GAUNTLET (stabilization complete)
        GAUNTLET -> CONFIRMED (passed stress test) -> NORMAL (ratchet applied, search for next)
        GAUNTLET -> REJECTED (failed stress test) -> NORMAL (continue searching)
    """
    NORMAL = "normal"
    DETECTION = "detection"
    STABILIZATION = "stabilization"
    GAUNTLET = "gauntlet"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


@dataclass
class BreakthroughCandidate:
    """
    Represents a candidate agent (or agents) for breakthrough validation.
    """
    agent: object  # DDPGAgent
    agents: list
    agent_idx: int
    agent_indices: list
    spike_score: float
    spike_scores: list
    detection_generation: int
    stabilization_start_gen: Optional[int] = None
    gauntlet_score: Optional[float] = None


class BreakthroughTracker:
    """
    Encapsulates all breakthrough/gauntlet state machine variables.

    Provides to_dict()/from_dict() for checkpoint serialization without
    requiring the full ERLTrainer instance.
    """

    def __init__(
        self,
        gauntlet_mode_enabled: bool = True,
        consistency_mode: bool = False,
        use_candidate_queue: bool = False,
        breakthrough_threshold: float = 0.05,
        breakthrough_quorum: int = 1,
        target_breakthroughs: int = 3,
    ):
        self.gauntlet_mode_enabled = gauntlet_mode_enabled

        # State machine
        self.state = BreakthroughState.NORMAL
        self.candidate: Optional[BreakthroughCandidate] = None
        self.confirmed_baseline = 0.0
        self.confirmed_breakthroughs = 0
        self.history: List[Dict] = []
        self.stabilization_generations_elapsed = 0

        # Configuration
        self.threshold = breakthrough_threshold
        self.quorum = breakthrough_quorum
        self.target_breakthroughs = target_breakthroughs

        # Candidate queue (heroes+consistency mode)
        self.use_candidate_queue = use_candidate_queue
        self.candidate_queue: List = []
        self.tested_candidate_indices: set = set()
        self.pending_baseline_update = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize breakthrough state for checkpointing."""
        candidate_data = None
        if self.candidate is not None:
            candidate_data = {
                'agent_idx': self.candidate.agent_idx,
                'agent_indices': self.candidate.agent_indices,
                'spike_score': self.candidate.spike_score,
                'spike_scores': self.candidate.spike_scores,
                'detection_generation': self.candidate.detection_generation,
                'stabilization_start_gen': self.candidate.stabilization_start_gen,
                'gauntlet_score': self.candidate.gauntlet_score,
            }

        return {
            'breakthrough_state': self.state.value if self.gauntlet_mode_enabled else None,
            'confirmed_baseline': self.confirmed_baseline,
            'confirmed_breakthroughs': self.confirmed_breakthroughs,
            'breakthrough_history': self.history,
            'stabilization_generations_elapsed': self.stabilization_generations_elapsed,
            'breakthrough_candidate': candidate_data,
            'use_candidate_queue': self.use_candidate_queue,
            'candidate_queue': self.candidate_queue,
            'tested_candidate_indices': list(self.tested_candidate_indices),
            'pending_baseline_update': self.pending_baseline_update,
        }

    def from_dict(self, d: Dict[str, Any]):
        """Restore breakthrough state from checkpoint dict."""
        state_str = d.get('breakthrough_state')
        if state_str and self.gauntlet_mode_enabled:
            self.state = BreakthroughState(state_str)
        self.confirmed_baseline = d.get('confirmed_baseline', 0.0)
        self.confirmed_breakthroughs = d.get('confirmed_breakthroughs', 0)
        self.history = d.get('breakthrough_history', [])
        self.stabilization_generations_elapsed = d.get('stabilization_generations_elapsed', 0)

        # Note: candidate agent objects cannot be serialized, only metadata.
        # The actual agent must be re-loaded from population after checkpoint restore.
        candidate_data = d.get('breakthrough_candidate')
        if candidate_data is not None:
            self.candidate = BreakthroughCandidate(
                agent=None,  # Must be re-loaded
                agents=[],
                agent_idx=candidate_data.get('agent_idx', 0),
                agent_indices=candidate_data.get('agent_indices', []),
                spike_score=candidate_data.get('spike_score', 0.0),
                spike_scores=candidate_data.get('spike_scores', []),
                detection_generation=candidate_data.get('detection_generation', 0),
                stabilization_start_gen=candidate_data.get('stabilization_start_gen'),
                gauntlet_score=candidate_data.get('gauntlet_score'),
            )
        else:
            self.candidate = None

        self.use_candidate_queue = d.get('use_candidate_queue', False)
        self.candidate_queue = d.get('candidate_queue', [])
        tested_indices = d.get('tested_candidate_indices', [])
        self.tested_candidate_indices = set(tested_indices) if isinstance(tested_indices, list) else tested_indices
        self.pending_baseline_update = d.get('pending_baseline_update')
