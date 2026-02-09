"""
Unit tests for training/multi_agent.py — MultiAgentOrchestrator.
"""

import pytest
from training.multi_agent import MultiAgentOrchestrator


class TestMultiAgentOrchestrator:
    def test_default_initialization(self):
        orch = MultiAgentOrchestrator()
        assert orch.num_committee_members == 9
        assert len(orch.member_breakthroughs) == 9
        assert all(b == 0 for b in orch.member_breakthroughs)
        assert orch.turnovers_completed == 0
        assert orch.current_member_idx == 0
        assert orch.phase == 'non_maverick'
        assert orch.gens_since_improvement == 0

    def test_custom_initialization(self):
        orch = MultiAgentOrchestrator(num_committee_members=5)
        assert len(orch.member_breakthroughs) == 5
        assert len(orch.member_baselines) == 5


class TestSerialization:
    def test_round_trip(self):
        orch = MultiAgentOrchestrator(num_committee_members=9)
        orch.current_member_idx = 3
        orch.turnovers_completed = 2
        orch.member_breakthroughs = [3, 3, 3, 2, 2, 3, 3, 3, 2]
        orch.member_baselines = [100.0, 200.0, 150.0, 120.0, 180.0, 160.0, 140.0, 170.0, 130.0]
        orch.phase = 'maverick'
        orch.non_maverick_members = [0, 1, 2, 3, 4]
        orch.maverick_members = [5, 6, 7, 8]
        orch.gens_since_improvement = 15
        orch.best_score_for_member = 250.0

        d = orch.to_dict()

        restored = MultiAgentOrchestrator(num_committee_members=9)
        restored.from_dict(d)

        assert restored.current_member_idx == 3
        assert restored.turnovers_completed == 2
        assert restored.member_breakthroughs == [3, 3, 3, 2, 2, 3, 3, 3, 2]
        assert restored.member_baselines == [100.0, 200.0, 150.0, 120.0, 180.0, 160.0, 140.0, 170.0, 130.0]
        assert restored.phase == 'maverick'
        assert restored.non_maverick_members == [0, 1, 2, 3, 4]
        assert restored.maverick_members == [5, 6, 7, 8]
        assert restored.gens_since_improvement == 15
        assert restored.best_score_for_member == 250.0

    def test_round_trip_empty(self):
        orch = MultiAgentOrchestrator()
        d = orch.to_dict()
        restored = MultiAgentOrchestrator()
        restored.from_dict(d)
        assert restored.current_member_idx == 0
        assert restored.turnovers_completed == 0


class TestTurnoverDetection:
    def test_no_turnover(self):
        orch = MultiAgentOrchestrator(num_committee_members=3)
        orch.member_breakthroughs = [1, 0, 1]
        assert orch.check_turnover() is False

    def test_first_turnover(self):
        orch = MultiAgentOrchestrator(num_committee_members=3)
        orch.member_breakthroughs = [1, 1, 1]
        assert orch.check_turnover() is True
        assert orch.turnovers_completed == 1

    def test_second_turnover(self):
        orch = MultiAgentOrchestrator(num_committee_members=3)
        orch.turnovers_completed = 1
        orch.member_breakthroughs = [2, 2, 2]
        assert orch.check_turnover() is True
        assert orch.turnovers_completed == 2

    def test_partial_second_turnover(self):
        orch = MultiAgentOrchestrator(num_committee_members=3)
        orch.turnovers_completed = 1
        orch.member_breakthroughs = [2, 1, 2]
        assert orch.check_turnover() is False


class TestImprovement:
    def test_record_improvement(self):
        orch = MultiAgentOrchestrator(num_committee_members=3)
        orch.gens_since_improvement = 10
        orch.record_improvement(member_idx=1, new_baseline=150.0, new_roi=8.5)

        assert orch.member_baselines[1] == 150.0
        assert orch.member_starting_rois[1] == 8.5
        assert orch.gens_since_improvement == 0
        assert orch.best_score_for_member == 150.0


class TestBreakthrough:
    def test_record_breakthrough_non_maverick(self):
        orch = MultiAgentOrchestrator(num_committee_members=3)
        orch.phase = 'non_maverick'
        orch.non_maverick_members = [0, 1, 2]
        orch.non_maverick_turnovers_per_agent = [0, 0, 0]

        orch.record_breakthrough(member_idx=1)

        assert orch.member_breakthroughs[1] == 1
        assert orch.non_maverick_turnovers_per_agent[1] == 1

    def test_record_breakthrough_maverick(self):
        orch = MultiAgentOrchestrator(num_committee_members=5)
        orch.phase = 'maverick'
        orch.maverick_members = [3, 4]
        orch.maverick_turnovers_per_agent = [0, 0]

        orch.record_breakthrough(member_idx=4)

        assert orch.member_breakthroughs[4] == 1
        assert orch.maverick_turnovers_per_agent[1] == 1


class TestGetCurrentMember:
    def test_with_roster(self):
        roster = {
            'members': [
                {'run_name': 'run-a', 'agent_id': 0},
                {'run_name': 'run-b', 'agent_id': 1},
            ]
        }
        orch = MultiAgentOrchestrator(num_committee_members=2, roster=roster)
        orch.current_member_idx = 1

        member = orch.get_current_member()
        assert member['run_name'] == 'run-b'

    def test_without_roster(self):
        orch = MultiAgentOrchestrator()
        assert orch.get_current_member() is None
