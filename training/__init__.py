"""
Training package for Project Eigen 2.

Public API re-exports for all extracted modules.
"""

from training.erl_trainer import ERLTrainer
from training.breakthrough import BreakthroughState, BreakthroughCandidate, BreakthroughTracker
from training.fitness import (
    NumpyEncoder,
    calculate_triad_fitness,
    calculate_holographic_fitness,
    calculate_pessimistic_fitness,
    calculate_penalized_median_fitness,
    aggregate_agent_stats,
    aggregate_population_stats,
    calculate_expectancy,
    hash_agent,
    hash_validation_slices,
)
from training.episode import run_episode, run_episode_batched
from training.validation_slices import generate_validation_slices, generate_gauntlet_slices
from training.workers import (
    SharedMemoryManager,
    _init_worker,
    _run_episode_worker,
    _run_validation_worker,
)
from training.checkpoint import CheckpointManager
from training.multi_agent import MultiAgentOrchestrator
