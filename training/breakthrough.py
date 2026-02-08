"""
Gauntlet state machine types and breakthrough tracking for Project Eigen 2.

Contains the BreakthroughState enum and BreakthroughCandidate dataclass.
The full BreakthroughTracker class will be added in Phase 2b.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


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
