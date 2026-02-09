"""
Multi-Agent Committee Training Orchestrator for Project Eigen 2.

Encapsulates all multi2-mode state: member tracking, breakthrough counts,
phase management (non-maverick/maverick), stuck detection, and turnover tracking.

Heavy methods that need deep trainer access (environment creation, agent evaluation,
Global50 updates) remain as ERLTrainer delegates. This class owns the STATE.
"""

from typing import List, Dict, Any, Optional


class MultiAgentOrchestrator:
    """
    Encapsulates all multi-agent (multi2) committee training state.

    Provides to_dict()/from_dict() for checkpoint serialization.
    """

    def __init__(
        self,
        num_committee_members: int = 9,
        roster: Optional[Dict] = None,
    ):
        self.num_committee_members = num_committee_members
        self.roster = roster

        # Per-member tracking
        self.member_breakthroughs: List[int] = [0] * num_committee_members
        self.member_baselines: List[float] = [0.0] * num_committee_members
        self.member_starting_rois: List[float] = [0.0] * num_committee_members
        self.turnovers_completed: int = 0
        self.current_member_idx: int = 0
        self.member_training_start_gen: int = 0
        self.generation_offset: int = 0

        # Stuck detection
        self.parent_agent = None  # Reference to parent agent for mutant injection
        self.gens_since_improvement: int = 0
        self.best_score_for_member: float = float('-inf')

        # Phase management (non-maverick / maverick)
        self.phase: str = 'non_maverick'
        self.non_maverick_members: List[int] = []
        self.maverick_members: List[int] = []
        self.current_non_maverick_idx: int = 0
        self.current_maverick_idx: int = 0
        self.non_maverick_turnovers_per_agent: List[int] = []
        self.maverick_turnovers_per_agent: List[int] = []

    def to_dict(self) -> Dict[str, Any]:
        """Serialize multi-agent state for checkpointing."""
        return {
            'current_member_idx': self.current_member_idx,
            'turnovers_completed': self.turnovers_completed,
            'member_breakthroughs': self.member_breakthroughs,
            'member_training_start_gen': self.member_training_start_gen,
            'member_baselines': self.member_baselines,
            'member_starting_rois': self.member_starting_rois,
            'multi_gens_since_improvement': self.gens_since_improvement,
            'multi_best_score_for_member': self.best_score_for_member,
            'multi_phase': self.phase,
            'non_maverick_members': self.non_maverick_members,
            'maverick_members': self.maverick_members,
            'current_non_maverick_idx': self.current_non_maverick_idx,
            'current_maverick_idx': self.current_maverick_idx,
            'non_maverick_turnovers_per_agent': self.non_maverick_turnovers_per_agent,
            'maverick_turnovers_per_agent': self.maverick_turnovers_per_agent,
        }

    def from_dict(self, d: Dict[str, Any]):
        """Restore multi-agent state from checkpoint dict."""
        self.current_member_idx = d.get('current_member_idx', 0)
        self.turnovers_completed = d.get('turnovers_completed', 0)
        self.member_breakthroughs = d.get('member_breakthroughs', [0] * self.num_committee_members)
        self.member_training_start_gen = d.get('member_training_start_gen', 0)

        saved_baselines = d.get('member_baselines')
        if saved_baselines:
            self.member_baselines = saved_baselines

        saved_rois = d.get('member_starting_rois')
        if saved_rois:
            self.member_starting_rois = saved_rois

        self.gens_since_improvement = d.get('multi_gens_since_improvement', 0)
        self.best_score_for_member = d.get('multi_best_score_for_member', float('-inf'))

        self.phase = d.get('multi_phase', 'non_maverick')
        self.non_maverick_members = d.get('non_maverick_members', [])
        self.maverick_members = d.get('maverick_members', [])
        self.current_non_maverick_idx = d.get('current_non_maverick_idx', 0)
        self.current_maverick_idx = d.get('current_maverick_idx', 0)
        self.non_maverick_turnovers_per_agent = d.get('non_maverick_turnovers_per_agent', [])
        self.maverick_turnovers_per_agent = d.get('maverick_turnovers_per_agent', [])

    def check_turnover(self) -> bool:
        """
        Check if all members have achieved a new turnover level.

        Returns:
            True if a new turnover was achieved
        """
        min_breakthroughs = min(self.member_breakthroughs)
        if min_breakthroughs > self.turnovers_completed:
            self.turnovers_completed = min_breakthroughs
            return True
        return False

    def get_current_member(self) -> Optional[Dict]:
        """Get the current member from roster."""
        if self.roster and 'members' in self.roster:
            return self.roster['members'][self.current_member_idx]
        return None

    def record_improvement(self, member_idx: int, new_baseline: float, new_roi: float):
        """
        Record an improvement for a member (updates baseline, resets stuck counter).

        Args:
            member_idx: Index of the member
            new_baseline: New baseline score
            new_roi: New ROI value
        """
        self.member_baselines[member_idx] = new_baseline
        self.member_starting_rois[member_idx] = new_roi
        self.gens_since_improvement = 0
        self.best_score_for_member = new_baseline

    def record_breakthrough(self, member_idx: int):
        """
        Record a breakthrough for a member.

        Args:
            member_idx: Index of the member
        """
        self.member_breakthroughs[member_idx] += 1

        # Update per-agent turnover tracking
        if self.phase == 'non_maverick' and member_idx in self.non_maverick_members:
            list_idx = self.non_maverick_members.index(member_idx)
            if list_idx < len(self.non_maverick_turnovers_per_agent):
                self.non_maverick_turnovers_per_agent[list_idx] = self.member_breakthroughs[member_idx]
        elif self.phase == 'maverick' and member_idx in self.maverick_members:
            list_idx = self.maverick_members.index(member_idx)
            if list_idx < len(self.maverick_turnovers_per_agent):
                self.maverick_turnovers_per_agent[list_idx] = self.member_breakthroughs[member_idx]
