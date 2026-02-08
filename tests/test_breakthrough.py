"""
Unit tests for training/breakthrough.py — BreakthroughTracker state machine.
"""

import pytest
from training.breakthrough import (
    BreakthroughState,
    BreakthroughCandidate,
    BreakthroughTracker,
)


class TestBreakthroughState:
    def test_all_states_exist(self):
        assert BreakthroughState.NORMAL.value == "normal"
        assert BreakthroughState.DETECTION.value == "detection"
        assert BreakthroughState.STABILIZATION.value == "stabilization"
        assert BreakthroughState.GAUNTLET.value == "gauntlet"
        assert BreakthroughState.CONFIRMED.value == "confirmed"
        assert BreakthroughState.REJECTED.value == "rejected"

    def test_state_from_string(self):
        assert BreakthroughState("normal") == BreakthroughState.NORMAL
        assert BreakthroughState("gauntlet") == BreakthroughState.GAUNTLET


class TestBreakthroughTracker:
    def test_default_initialization(self):
        tracker = BreakthroughTracker()
        assert tracker.state == BreakthroughState.NORMAL
        assert tracker.candidate is None
        assert tracker.confirmed_baseline == 0.0
        assert tracker.confirmed_breakthroughs == 0
        assert tracker.history == []
        assert tracker.stabilization_generations_elapsed == 0

    def test_custom_initialization(self):
        tracker = BreakthroughTracker(
            consistency_mode=True,
            breakthrough_threshold=0.03,
            breakthrough_quorum=2,
            target_breakthroughs=5,
        )
        assert tracker.threshold == 0.03
        assert tracker.quorum == 2
        assert tracker.target_breakthroughs == 5

    def test_to_dict_empty(self):
        tracker = BreakthroughTracker()
        d = tracker.to_dict()
        assert d['breakthrough_state'] == 'normal'
        assert d['confirmed_baseline'] == 0.0
        assert d['confirmed_breakthroughs'] == 0
        assert d['breakthrough_candidate'] is None
        assert d['candidate_queue'] == []
        assert d['tested_candidate_indices'] == []

    def test_to_dict_with_candidate(self):
        tracker = BreakthroughTracker()
        tracker.state = BreakthroughState.DETECTION
        tracker.candidate = BreakthroughCandidate(
            agent=None, agents=[], agent_idx=5,
            agent_indices=[5, 8], spike_score=150.0,
            spike_scores=[150.0, 140.0], detection_generation=42,
        )
        d = tracker.to_dict()
        assert d['breakthrough_state'] == 'detection'
        assert d['breakthrough_candidate']['agent_idx'] == 5
        assert d['breakthrough_candidate']['spike_score'] == 150.0
        assert d['breakthrough_candidate']['detection_generation'] == 42

    def test_serialization_round_trip(self):
        """to_dict -> from_dict should preserve all state."""
        tracker = BreakthroughTracker(
            breakthrough_threshold=0.05,
            breakthrough_quorum=3,
            target_breakthroughs=4,
        )
        tracker.state = BreakthroughState.STABILIZATION
        tracker.confirmed_baseline = 200.0
        tracker.confirmed_breakthroughs = 2
        tracker.history = [{'gen': 10, 'score': 150.0}, {'gen': 25, 'score': 200.0}]
        tracker.stabilization_generations_elapsed = 5
        tracker.use_candidate_queue = True
        tracker.tested_candidate_indices = {1, 3, 7}

        d = tracker.to_dict()

        restored = BreakthroughTracker()
        restored.from_dict(d)

        assert restored.state == BreakthroughState.STABILIZATION
        assert restored.confirmed_baseline == 200.0
        assert restored.confirmed_breakthroughs == 2
        assert len(restored.history) == 2
        assert restored.stabilization_generations_elapsed == 5
        assert restored.use_candidate_queue is True
        assert restored.tested_candidate_indices == {1, 3, 7}

    def test_baseline_never_decreases_pattern(self):
        """Demonstrate the ratcheting baseline pattern."""
        tracker = BreakthroughTracker()
        tracker.confirmed_baseline = 100.0

        # Ratchet up
        new_baseline = 150.0
        if new_baseline > tracker.confirmed_baseline:
            tracker.confirmed_baseline = new_baseline
        assert tracker.confirmed_baseline == 150.0

        # Attempt to decrease (should not change if caller uses pattern correctly)
        lower_baseline = 80.0
        if lower_baseline > tracker.confirmed_baseline:
            tracker.confirmed_baseline = lower_baseline
        assert tracker.confirmed_baseline == 150.0

    def test_disabled_gauntlet_mode(self):
        tracker = BreakthroughTracker(gauntlet_mode_enabled=False)
        d = tracker.to_dict()
        assert d['breakthrough_state'] is None
