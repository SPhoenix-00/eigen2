"""
Episode execution functions for Project Eigen 2.

Extracted from ERLTrainer — runs agents through trading environments
and collects transitions/stats.
"""

import numpy as np
from typing import Tuple, Dict, Optional

from utils.config import Config


def run_episode(
    agent,
    env,
    start_idx: int,
    end_idx: int,
    training: bool = True,
    replay_buffer=None,
    transition_collector: list = None,
) -> Tuple[float, Dict]:
    """
    Run one episode with an agent using a persistent environment.

    Args:
        agent: DDPGAgent to run
        env: Persistent TradingEnvironment to reuse
        start_idx: Starting day index (first day of trading period)
        end_idx: Ending day index (includes settlement period)
        training: Whether this is training (adds to replay buffer)
        replay_buffer: Optional replay buffer for direct writes (parallel mode)
        transition_collector: Optional list to collect transitions for batch writing (local mode)

    Returns:
        Tuple of (cumulative_reward, episode_info)
    """
    # Calculate trading end
    trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

    # Set environment training mode
    env.set_training_mode(training)

    # Reset persistent environment with new indices
    state, info = env.reset(
        start_idx=start_idx,
        end_idx=end_idx,
        trading_end_idx=trading_end_idx
    )
    cumulative_reward = 0.0
    steps = 0

    # Run episode
    while True:
        action = agent.select_action(state, add_noise=training)
        next_state, reward, terminated, truncated, info = env.step(action)

        # Store transition
        transition = {
            'state': state.astype(np.float32),
            'action': action.astype(np.float32),
            'reward': reward,
            'next_state': next_state.astype(np.float32),
            'done': float(terminated or truncated)
        }

        if transition_collector is not None:
            transition_collector.append(transition)
        elif replay_buffer is not None:
            replay_buffer.add(**transition)

        cumulative_reward += reward
        steps += 1
        state = next_state

        if terminated or truncated:
            break

    # Get episode summary
    episode_summary = env.get_episode_summary()
    episode_summary['steps'] = steps

    final_fitness = float(cumulative_reward)

    # Apply zero-trades penalty
    if episode_summary['num_trades'] == 0:
        final_fitness -= episode_summary['zero_trades_penalty']

    # Apply win rate bonus
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


def run_episode_batched(
    agent,
    env,
    start_idx: int,
    end_idx: int,
    training: bool = True,
    batch_size: int = 16,
    replay_buffer=None,
) -> Tuple[float, Dict]:
    """
    Run one episode with batched inference for faster evaluation.

    Args:
        agent: DDPGAgent to run
        env: Persistent TradingEnvironment to reuse
        start_idx: Starting day index
        end_idx: Ending day index
        training: Whether this is training
        batch_size: Number of states to batch together (default: 16)
        replay_buffer: Optional replay buffer for transition storage

    Returns:
        Tuple of (cumulative_reward, episode_info)
    """
    trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS

    env.set_training_mode(training)
    state, info = env.reset(
        start_idx=start_idx,
        end_idx=end_idx,
        trading_end_idx=trading_end_idx
    )
    cumulative_reward = 0.0
    steps = 0

    state_buffer = []
    replay_buffer_data = []
    terminated = False
    truncated = False

    while True:
        state_buffer.append(state.copy())

        if len(state_buffer) >= batch_size:
            states_array = np.stack(state_buffer)
            actions_batch = agent.select_actions_batch(states_array, add_noise=training)

            for i, action in enumerate(actions_batch):
                next_state, reward, terminated, truncated, info = env.step(action)

                if training:
                    replay_buffer_data.append({
                        'state': state_buffer[i].astype(np.float32),
                        'action': action.astype(np.float32),
                        'reward': reward,
                        'next_state': next_state.astype(np.float32),
                        'done': float(terminated or truncated)
                    })

                cumulative_reward += reward
                steps += 1
                state = next_state

                if terminated or truncated:
                    break

            state_buffer = []

            if terminated or truncated:
                break

    # Process remaining states
    if len(state_buffer) > 0:
        states_array = np.stack(state_buffer)
        actions_batch = agent.select_actions_batch(states_array, add_noise=training)

        for i, action in enumerate(actions_batch):
            next_state, reward, terminated, truncated, info = env.step(action)

            if training:
                replay_buffer_data.append({
                    'state': state_buffer[i].astype(np.float32),
                    'action': action.astype(np.float32),
                    'reward': reward,
                    'next_state': next_state.astype(np.float32),
                    'done': float(terminated or truncated)
                })

            cumulative_reward += reward
            steps += 1
            state = next_state

            if terminated or truncated:
                break

    # Add transitions to replay buffer
    if training and replay_buffer is not None:
        for transition in replay_buffer_data:
            replay_buffer.add(
                state=transition['state'],
                action=transition['action'],
                reward=transition['reward'],
                next_state=transition['next_state'],
                done=transition['done']
            )

    # Get episode summary
    episode_summary = env.get_episode_summary()
    episode_summary['steps'] = steps

    final_fitness = float(cumulative_reward)

    if episode_summary['num_trades'] == 0:
        final_fitness -= episode_summary['zero_trades_penalty']

    return final_fitness, episode_summary
