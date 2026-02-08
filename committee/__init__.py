"""
Committee Production Engine for Project Eigen 2 (Global50 Edition)

- Phase 1: Draft Day (Global50 Selection with Coefficient Correlation Optimization)
- Phase 2: Validation (3-Slice Holdout Testing)
- Cloud Sync: Committee roster is mirrored to cloud in global50/cw{N}/committee/

USAGE:
  python -m committee --verify-only   # Check data split only
  python -m committee --draft         # Phase 1: Select committee from Global50
  python -m committee --validate      # Phase 2: Validate on holdout slices
  python -m committee --mirror        # Sync roster, correlation, and agent files
"""

# Public API - backward compatible re-exports
# These ensure `from committee import X` still works for external consumers.
from committee.utils import (
    GLOBAL50_BASE_DIR,
    COMMITTEE_DIR,
    ROSTER_FILENAME,
    CORRELATION_FILENAME,
    sanitize_date_for_filename,
    convert_numpy_types,
    calculate_max_drawdown,
    parse_date_input,
    parse_date_flexible,
    find_date_index,
    format_quality_ratio,
    load_agent_actor_only,
    get_agent_filepath,
    load_global50_candidates,
    verify_data_split,
    load_normalization_stats,
    get_holdout_data,
    get_validation_data,
    calculate_expectancy,
    generate_validation_slices,
    aggregate_slice_metrics,
    aggregate_consensus_stats,
    build_metrics_result,
    build_member_data,
    enrich_closed_trades,
)

from committee.manager import CommitteeManager

from committee.agent import (
    CommitteeAgent,
    calculate_agent_stats_vectorized,
    recalculate_conviction_thresholds,
)

from committee.optimization import (
    calculate_coefficient_correlations,
    committee_objective,
    find_highest_correlation_pair,
    find_best_swap_candidate,
    optimize_committee,
    interactive_correlation_refinement,
    automatic_correlation_refinement,
)

from committee.validation import (
    evaluate_agent_on_slice,
    evaluate_committee_on_slice,
    run_validation,
    run_validation_sweep,
    run_quorum_sweep,
    run_conviction_sweep,
    run_combined_sweep,
)

