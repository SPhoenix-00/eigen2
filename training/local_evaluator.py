"""
CPU-Optimized Local Evaluator for Project Eigen 2

Clean-sheet implementation of evaluation and validation loops for --local mode.
Designed to eliminate I/O bottlenecks and minimize overhead through:

1. In-memory transition buffer (no disk I/O)
2. Batched agent inference
3. Pre-allocated numpy arrays
4. Minimal memory copying
5. Direct computation without caching overhead

Usage:
    evaluator = LocalEvaluator(trainer)
    fitness_scores, stats = evaluator.evaluate_population()
    val_results = evaluator.validate_population(quality_threshold)
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from dataclasses import dataclass

from utils.config import Config
from models.ddpg_agent import DDPGAgent


@dataclass
class TransitionBatch:
    """Pre-allocated transition storage for zero-copy collection."""
    states: np.ndarray       # [capacity, *state_shape]
    actions: np.ndarray      # [capacity, num_stocks, 2]
    rewards: np.ndarray      # [capacity]
    next_states: np.ndarray  # [capacity, *state_shape]
    dones: np.ndarray        # [capacity]
    count: int = 0

    @classmethod
    def create(cls, capacity: int, state_shape: Tuple[int, ...]) -> 'TransitionBatch':
        """Pre-allocate arrays for transition storage."""
        return cls(
            states=np.zeros((capacity, *state_shape), dtype=np.float32),
            actions=np.zeros((capacity, Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM), dtype=np.float32),
            rewards=np.zeros(capacity, dtype=np.float32),
            next_states=np.zeros((capacity, *state_shape), dtype=np.float32),
            dones=np.zeros(capacity, dtype=np.float32),
            count=0
        )

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: float):
        """Add a single transition (zero-copy into pre-allocated arrays)."""
        if self.count >= len(self.rewards):
            return  # Buffer full

        idx = self.count
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done
        self.count += 1

    def get_valid(self) -> Dict[str, np.ndarray]:
        """Return only the filled portion of the buffer."""
        n = self.count
        return {
            'states': self.states[:n],
            'actions': self.actions[:n],
            'rewards': self.rewards[:n],
            'next_states': self.next_states[:n],
            'dones': self.dones[:n]
        }

    def reset(self):
        """Reset counter without reallocating memory."""
        self.count = 0


class LocalEvaluator:
    """
    CPU-optimized evaluation and validation for --local mode.

    Key optimizations:
    - Single environment instance reused for all episodes
    - In-memory transition buffer (no disk I/O during evaluation)
    - Batched inference where possible
    - Minimal tensor creation overhead
    """

    def __init__(self, trainer):
        """
        Initialize with reference to ERLTrainer.

        Args:
            trainer: ERLTrainer instance (provides population, env, replay_buffer, etc.)
        """
        self.trainer = trainer

        # Cache frequently accessed trainer attributes
        self.population = trainer.population
        self.eval_env = trainer.eval_env
        self.replay_buffer = trainer.replay_buffer

        # Pre-allocate transition buffer for one generation's worth of data
        # Estimate: 96 agents × 3 slices × ~125 steps = ~36,000 transitions max
        # Use 50,000 for safety margin
        state_shape = (Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL)
        self.transition_buffer = TransitionBatch.create(50000, state_shape)

        # Pre-compute episode indices ranges
        self._init_episode_ranges()

    def _init_episode_ranges(self):
        """Pre-compute valid episode start/end ranges."""
        total_days_needed = (Config.CONTEXT_WINDOW_DAYS +
                            Config.TRADING_PERIOD_DAYS +
                            Config.SETTLEMENT_PERIOD_DAYS)

        self.train_start = self.trainer.train_start_idx
        self.train_end = self.trainer.train_end_idx
        self.max_start = self.train_end - total_days_needed

        if self.max_start <= self.train_start:
            raise ValueError(f"Not enough training data: need {total_days_needed} days")

    def evaluate_population(self) -> Tuple[List[float], Dict]:
        """
        Evaluate all agents in population with optimized CPU execution.

        Returns:
            Tuple of (fitness_scores, aggregate_stats)
        """
        # Refresh references (population changes after evolution)
        self.population = self.trainer.population
        self.eval_env = self.trainer.eval_env

        num_episodes = 5 if self.trainer.consistency_mode else 3

        # Log evaluation info
        if self.trainer.multi_mode:
            scoring_method = "median - 0.5*std (Penalized Median)"
        else:
            scoring_method = "0.4*mean + 0.6*min (pessimistic)"

        print(f"\n--- Generation {self.trainer.generation + 1}: Evaluating Population (Local/CPU) ---")
        print(f"Multi-slice evaluation: {num_episodes} slices per agent, scoring = {scoring_method}")

        num_elites = sum(1 for a in self.population if a.is_elite)
        print(f"Elite Demonstration: {num_elites} elites + {len(self.population) - num_elites} exploratory")

        # Reset transition buffer
        self.transition_buffer.reset()

        # Pre-generate all episode start indices for reproducibility
        np.random.seed(self.trainer.seed + self.trainer.generation * 1000)
        all_starts = np.random.randint(
            self.train_start,
            self.max_start,
            size=(len(self.population), num_episodes)
        )

        fitness_scores = []
        all_episode_stats = []

        # Process each agent
        for agent_idx, agent in enumerate(tqdm(self.population, desc="Evaluating (CPU)")):
            slice_fitness = []
            slice_stats = []

            for ep_idx in range(num_episodes):
                start_idx = int(all_starts[agent_idx, ep_idx])
                end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS

                # Run episode with optimized path
                fitness, episode_info = self._run_episode_optimized(
                    agent=agent,
                    start_idx=start_idx,
                    end_idx=end_idx,
                    collect_transitions=(not agent.is_elite)  # Only exploratory agents
                )

                # Calculate structural fitness
                triad_fitness = self.trainer.calculate_triad_fitness(episode_info)
                slice_fitness.append(triad_fitness)
                slice_stats.append(episode_info)

            # Calculate final fitness
            if self.trainer.multi_mode:
                final_fitness = self._penalized_median(slice_fitness)
            else:
                final_fitness = self._pessimistic(slice_fitness)

            fitness_scores.append(float(final_fitness))
            all_episode_stats.append(self._aggregate_stats(slice_stats))

        # Transfer collected transitions to replay buffer
        if self.transition_buffer.count > 0:
            print(f"\n--- Transferring {self.transition_buffer.count} transitions to replay buffer ---")
            self._transfer_to_buffer()

        # Aggregate population stats
        aggregate_stats = self.trainer._aggregate_population_stats(all_episode_stats, fitness_scores)

        return fitness_scores, aggregate_stats

    def _run_episode_optimized(self, agent: DDPGAgent, start_idx: int, end_idx: int,
                                collect_transitions: bool = True) -> Tuple[float, Dict]:
        """
        Run a single episode with minimal overhead.

        Optimizations:
        - Reuses persistent environment
        - Direct numpy operations
        - In-memory transition storage
        """
        env = self.eval_env
        trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

        # Set training mode and reset
        env.set_training_mode(collect_transitions)
        state, _ = env.reset(
            start_idx=start_idx,
            end_idx=end_idx,
            trading_end_idx=trading_end_idx
        )

        cumulative_reward = 0.0
        steps = 0

        # Episode loop
        while True:
            # Get action from agent
            action = agent.select_action(state, add_noise=collect_transitions)

            # Take step
            next_state, reward, terminated, truncated, _ = env.step(action)

            # Store transition in pre-allocated buffer (if collecting)
            if collect_transitions:
                self.transition_buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=float(terminated or truncated)
                )

            cumulative_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break

        # Get episode summary
        episode_summary = env.get_episode_summary()
        episode_summary['steps'] = steps

        # Calculate final fitness
        final_fitness = float(cumulative_reward)

        if episode_summary['num_trades'] == 0:
            final_fitness -= episode_summary['zero_trades_penalty']

        if episode_summary['num_trades'] >= Config.WIN_RATE_BONUS_MIN_TRADES:
            win_rate_pct = episode_summary['win_rate'] * 100.0
            if win_rate_pct > Config.WIN_RATE_BONUS_THRESHOLD:
                bonus = (win_rate_pct - Config.WIN_RATE_BONUS_THRESHOLD) ** 2
                final_fitness += bonus
                episode_summary['win_rate_bonus'] = bonus
            else:
                episode_summary['win_rate_bonus'] = 0.0
        else:
            episode_summary['win_rate_bonus'] = 0.0

        return final_fitness, episode_summary

    def _transfer_to_buffer(self):
        """
        Transfer collected transitions to the on-disk replay buffer.
        Uses batch write for efficiency.
        """
        transitions = self.transition_buffer.get_valid()
        n = self.transition_buffer.count

        # Convert to list of dicts for batch add
        transition_list = []
        for i in range(n):
            transition_list.append({
                'state': transitions['states'][i],
                'action': transitions['actions'][i],
                'reward': float(transitions['rewards'][i]),
                'next_state': transitions['next_states'][i],
                'done': float(transitions['dones'][i])
            })

        # Batch write to disk
        self.replay_buffer.add_batch(transition_list)

        # Reset buffer for next generation
        self.transition_buffer.reset()

    def validate_population(self, quality_threshold: float = None) -> List[Dict]:
        """
        Validate all agents with optimized CPU execution.

        Args:
            quality_threshold: Threshold for counting quality trades

        Returns:
            List of validation results for each agent
        """
        # Refresh references (population changes after evolution)
        self.population = self.trainer.population
        self.eval_env = self.trainer.eval_env

        print(f"\n--- Walk-Forward Validation (Local/CPU) ---")

        if not self.trainer.current_generation_val_slices:
            raise ValueError("No validation slices for this generation")

        validation_slices = self.trainer.current_generation_val_slices
        use_penalized_median = self.trainer.multi_mode

        validation_results = []

        for agent in tqdm(self.population, desc="Validating (CPU)"):
            val_result = self._validate_agent_optimized(
                agent=agent,
                slices=validation_slices,
                quality_threshold=quality_threshold,
                use_penalized_median=use_penalized_median
            )
            validation_results.append(val_result)

        return validation_results

    def _validate_agent_optimized(self, agent: DDPGAgent,
                                   slices: List[Tuple[int, int, str]],
                                   quality_threshold: float = None,
                                   use_penalized_median: bool = False) -> Dict:
        """
        Validate a single agent on all slices with minimal overhead.

        Args:
            agent: Agent to validate
            slices: List of (start_idx, end_idx, name) tuples
            quality_threshold: Threshold for quality trades
            use_penalized_median: Use penalized median scoring

        Returns:
            Validation results dictionary
        """
        env = self.eval_env
        slice_results = []
        all_closed_trades = []

        for start_idx, end_idx, _ in slices:
            # Run episode (no transitions collected for validation)
            fitness, episode_info = self._run_validation_episode(
                agent=agent,
                env=env,
                start_idx=start_idx,
                end_idx=end_idx
            )

            # Add max coefficient bonus for zero-trade agents
            if episode_info['num_trades'] == 0:
                max_coeff = episode_info.get('max_coefficient_during_episode', 0.0)
                fitness = fitness + max_coeff

            slice_results.append({
                'fitness': fitness,
                'win_rate': episode_info['win_rate'],
                'num_trades': episode_info['num_trades'],
                'num_wins': episode_info['num_wins'],
                'num_losses': episode_info['num_losses'],
                'avg_reward_per_trade': episode_info['avg_reward_per_trade'],
                'raw_pnl': episode_info.get('raw_pnl', 0.0),
                'total_investment': episode_info.get('total_investment', 0.0),
                'peak_capital_employed': episode_info.get('peak_capital_employed', 0.0)
            })

            if 'closed_trades' in episode_info and episode_info['closed_trades']:
                all_closed_trades.extend(episode_info['closed_trades'])

        # Aggregate slice fitness
        fitness_scores = [r['fitness'] for r in slice_results]

        if use_penalized_median:
            validation_fitness = self._penalized_median(fitness_scores)
        else:
            validation_fitness = self._pessimistic(fitness_scores)

        # Aggregate metrics
        total_raw_pnl = sum(r['raw_pnl'] for r in slice_results)
        total_peak_capital = sum(r['peak_capital_employed'] for r in slice_results)
        roi = (total_raw_pnl / total_peak_capital * 100) if total_peak_capital > 0 else 0.0

        total_wins = sum(r['num_wins'] for r in slice_results)
        total_losses = sum(r['num_losses'] for r in slice_results)
        total_trades = total_wins + total_losses
        global_win_rate = total_wins / total_trades if total_trades > 0 else 0.0

        expectancy = self.trainer.calculate_expectancy(all_closed_trades)

        quality_count = 0
        if quality_threshold is not None:
            quality_count = sum(
                1 for trade in all_closed_trades
                if trade.get('gain_pct', 0) >= quality_threshold
            )

        sample_trade = all_closed_trades[0] if all_closed_trades else None

        return {
            'fitness': validation_fitness,
            'fitness_all_slices': fitness_scores,
            'fitness_mean': float(np.mean(fitness_scores)),
            'fitness_min': float(np.min(fitness_scores)),
            'win_rate': global_win_rate,
            'num_trades': int(np.mean([r['num_trades'] for r in slice_results])),
            'num_wins': int(np.mean([r['num_wins'] for r in slice_results])),
            'num_losses': int(np.mean([r['num_losses'] for r in slice_results])),
            'avg_reward_per_trade': float(np.mean([r['avg_reward_per_trade'] for r in slice_results])),
            'raw_pnl': total_raw_pnl,
            'total_investment': sum(r['total_investment'] for r in slice_results),
            'roi': roi,
            'expectancy': expectancy,
            'sample_trade': sample_trade,
            'total_trades': total_trades,
            'quality_count': quality_count,
        }

    def _run_validation_episode(self, agent: DDPGAgent, env,
                                 start_idx: int, end_idx: int) -> Tuple[float, Dict]:
        """
        Run a validation episode (no noise, no transition collection).

        Note: Episode steps are inherently sequential (each action changes env state),
        so we use simple single-action inference. The overhead is minimal since
        inference is fast on CPU with no noise computation.
        """
        trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

        env.set_training_mode(False)
        state, _ = env.reset(
            start_idx=start_idx,
            end_idx=end_idx,
            trading_end_idx=trading_end_idx
        )

        cumulative_reward = 0.0
        steps = 0

        # Simple sequential episode loop
        while True:
            action = agent.select_action(state, add_noise=False)
            next_state, reward, terminated, truncated, _ = env.step(action)

            cumulative_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break

        # Get episode summary
        episode_summary = env.get_episode_summary()
        episode_summary['steps'] = steps

        # Calculate final fitness
        final_fitness = float(cumulative_reward)

        if episode_summary['num_trades'] == 0:
            final_fitness -= episode_summary['zero_trades_penalty']

        if episode_summary['num_trades'] >= Config.WIN_RATE_BONUS_MIN_TRADES:
            win_rate_pct = episode_summary['win_rate'] * 100.0
            if win_rate_pct > Config.WIN_RATE_BONUS_THRESHOLD:
                bonus = (win_rate_pct - Config.WIN_RATE_BONUS_THRESHOLD) ** 2
                final_fitness += bonus
                episode_summary['win_rate_bonus'] = bonus
            else:
                episode_summary['win_rate_bonus'] = 0.0
        else:
            episode_summary['win_rate_bonus'] = 0.0

        return final_fitness, episode_summary

    @staticmethod
    def _penalized_median(scores: List[float]) -> float:
        """Penalized median: median - 0.5 * std"""
        arr = np.array(scores)
        return float(np.median(arr) - 0.5 * np.std(arr))

    @staticmethod
    def _pessimistic(scores: List[float]) -> float:
        """Pessimistic aggregation: 0.4 * mean + 0.6 * min"""
        arr = np.array(scores)
        return float(0.4 * np.mean(arr) + 0.6 * np.min(arr))

    def _aggregate_stats(self, slice_stats: List[Dict]) -> Dict:
        """Aggregate episode stats across slices."""
        return self.trainer._aggregate_agent_stats(slice_stats)
