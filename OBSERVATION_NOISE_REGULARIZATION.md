# Observation Noise Regularization

## Overview

This document describes the observation noise regularization feature added to combat overfitting in the trading agent. This technique prevents the agent from memorizing exact market values and forces it to learn more robust, generalizable trading patterns.

## Problem Statement

Without regularization, RL agents can overfit to training data by learning overly precise rules like:
- "If RSI is exactly 0.453 and TRIX is exactly -0.12, buy"

These precise rules fail to generalize to unseen data where values are slightly different.

## Solution

Add small Gaussian noise to observations during training only. This forces the agent to learn "fuzzy" rules that are more robust:
- "If RSI is low and TRIX is negative, buy"

## Implementation

### 1. Configuration ([utils/config.py](utils/config.py:49))

Added a new configuration parameter:

```python
# Regularization
OBSERVATION_NOISE_STD = 0.01  # Gaussian noise added to observations during training to prevent overfitting
```

### 2. Environment Changes ([environment/trading_env.py](environment/trading_env.py))

#### a. Added `is_training` flag

The environment now accepts an `is_training` parameter in its constructor:

```python
def __init__(self, ..., is_training: bool = True):
    """
    Args:
        ...
        is_training: If True, applies observation noise for regularization
    """
    self.is_training = is_training
```

#### b. Added `set_training_mode()` method

Allows dynamic switching between training and evaluation modes:

```python
def set_training_mode(self, is_training: bool):
    """
    Set whether environment is in training mode.

    Args:
        is_training: If True, applies observation noise for regularization
    """
    self.is_training = is_training
```

#### c. Modified `_get_observation()` method

Applies Gaussian noise only during training:

```python
def _get_observation(self) -> np.ndarray:
    """
    Get current observation (normalized context window).

    Applies Gaussian noise during training as a regularization technique
    to prevent overfitting. This encourages the agent to learn robust,
    generalized patterns rather than memorizing exact values.
    """
    # ... normalize data ...

    # Add observation noise for regularization during training
    if self.is_training:
        noise = np.random.normal(0.0, Config.OBSERVATION_NOISE_STD, normalized.shape)
        normalized = normalized + noise

    return normalized.astype(np.float32)
```

### 3. Training Integration ([training/erl_trainer.py](training/erl_trainer.py:312-313))

The `run_episode()` method sets the environment's training mode based on whether it's a training or validation episode:

```python
def run_episode(self, agent, env, start_idx, end_idx, training: bool = True):
    # Set environment training mode (affects observation noise for regularization)
    env.set_training_mode(training)

    # ... rest of episode logic ...
```

This ensures:
- **Training episodes**: Observation noise is applied (helps prevent overfitting)
- **Validation episodes**: No noise applied (accurate evaluation on clean data)

### 4. Evaluation Scripts

Updated evaluation scripts to explicitly set `is_training=False`:

- [evaluate_best_agent.py](evaluate_best_agent.py:220): `is_training=False`
- [evaluate_champions.py](evaluate_champions.py:75): `is_training=False`

## Testing

A comprehensive test suite was created: [test_observation_noise.py](test_observation_noise.py)

The test verifies:
1. ✓ Training mode adds noise to observations (different each reset)
2. ✓ Evaluation mode does not add noise (identical each reset)
3. ✓ Dynamic mode switching works correctly

Run the test:
```bash
./venv/bin/python test_observation_noise.py
```

## Benefits

1. **Prevents Overfitting**: Agent can't memorize exact values
2. **Improves Generalization**: Forces learning of robust patterns
3. **Better Validation Performance**: Model should perform better on unseen data
4. **No Impact on Evaluation**: Validation and testing use clean observations

## Tuning the Noise Level

The noise level is controlled by `Config.OBSERVATION_NOISE_STD` (default: 0.01).

- **Too low** (< 0.005): May not provide enough regularization
- **Too high** (> 0.05): May make training unstable or too difficult
- **Recommended range**: 0.01 - 0.02

To adjust, modify the value in [utils/config.py](utils/config.py:49).

## Usage

### Creating Environments

```python
# Training environment (with noise)
env_train = TradingEnvironment(
    ...,
    is_training=True  # Default
)

# Evaluation environment (no noise)
env_eval = TradingEnvironment(
    ...,
    is_training=False
)
```

### Dynamic Mode Switching

```python
# Switch to evaluation mode
env.set_training_mode(False)

# Switch back to training mode
env.set_training_mode(True)
```

## Related Work

This technique is inspired by common regularization methods in deep learning:
- **Data augmentation** in computer vision (random crops, rotations)
- **Dropout** in neural networks
- **Input noise injection** in robust optimization

The key insight: making training slightly "harder" by adding noise forces the model to learn more general, robust features that transfer better to test data.
