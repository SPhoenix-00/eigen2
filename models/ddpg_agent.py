"""
DDPG Agent for Project Eigen 2
Deep Deterministic Policy Gradient agent with target networks
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Optional
import copy
from collections import deque
from torch.amp import autocast, GradScaler

from models.networks import Actor, Critic
from utils.config import Config


class DDPGAgent:
    """
    DDPG Agent with Actor-Critic networks and target networks.
    """
    
    def __init__(self, agent_id: int = 0):
        """
        Initialize DDPG agent.

        Args:
            agent_id: Unique identifier for this agent (used in population)
        """
        self.agent_id = agent_id
        self.device = Config.DEVICE
        self.is_elite = False  # Track if this agent is an elite (for replay buffer diversity)
        
        # Actor networks
        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        # Critic networks
        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=Config.ACTOR_LR,
            weight_decay=Config.WEIGHT_DECAY
        )
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=Config.CRITIC_LR,
            weight_decay=Config.WEIGHT_DECAY
        )

        self.actor_scaler = GradScaler('cuda')
        self.critic_scaler = GradScaler('cuda')
        
        # Exploration noise
        self.noise_scale = Config.NOISE_SCALE
        
        # Training statistics
        # CRITICAL FIX: Use deque with maxlen to prevent unbounded growth (~10-20MB per gen)
        # Only last 1000 values are used by get_stats() anyway
        self.actor_loss_history = deque(maxlen=1000)
        self.critic_loss_history = deque(maxlen=1000)
        self.update_count = 0
        
    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action using current policy.
        
        Args:
            state: State observation [context_days, num_columns, 9]
            add_noise: Whether to add exploration noise
            
        Returns:
            Action [108, 2]
        """
        self.actor.eval()
        
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            
            # Get action from actor
            action = self.actor(state_tensor)
            action = action.squeeze(0).cpu().numpy()
            
            # Add exploration noise if training
            if add_noise:
                noise = np.random.normal(0, self.noise_scale, action.shape)
                action = action + noise

                # Clip to valid ranges
                action[:, 0] = np.maximum(action[:, 0], 0)  # Coefficient >= 0
                action[:, 1] = np.clip(action[:, 1], Config.MIN_SALE_TARGET, Config.MAX_SALE_TARGET)

            # !!! CRITICAL FIX: DO NOT ROUND HERE !!!
            # This operation is non-differentiable and breaks the DDPG gradient.
            # It causes the agent to overfit to the training noise.
            # action[:, 0] = np.round(action[:, 0]) # <-- DELETED

            # Cap coefficients at 100 to prevent reward explosion
            # (reward = coefficient × gain_pct, so uncapped coefficients could destabilize training)
            action[:, 0] = np.clip(action[:, 0], 0, 100)

        self.actor.train()
        return action

    def select_actions_batch(self, states: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """
        Batched action selection for faster evaluation.

        Processes multiple states in a single forward pass through the actor network,
        providing significant speedup (10-15%) for GPU inference during evaluation.

        Args:
            states: Batch of state observations [batch_size, context_days, num_columns, features]
            add_noise: Whether to add exploration noise (typically False for evaluation)

        Returns:
            actions: Batch of actions [batch_size, num_stocks, 2]
        """
        self.actor.eval()

        with torch.no_grad():
            # Convert to tensor and move to GPU
            states_tensor = torch.FloatTensor(states).to(self.device)

            # Single forward pass for entire batch (FAST!)
            actions_tensor = self.actor(states_tensor)

            # Move back to CPU
            actions = actions_tensor.cpu().numpy()

            # Add noise if requested (training mode)
            if add_noise:
                noise = np.random.normal(0, self.noise_scale, actions.shape)
                actions = actions + noise

                # Clip to valid ranges
                actions[:, :, 0] = np.maximum(actions[:, :, 0], 0)  # Coefficient >= 0
                actions[:, :, 1] = np.clip(actions[:, :, 1], Config.MIN_SALE_TARGET, Config.MAX_SALE_TARGET)

            # Cap coefficients
            actions[:, :, 0] = np.clip(actions[:, :, 0], 0, 100)

        self.actor.train()
        return actions

    def update(self, batch: dict, accumulate: bool = False) -> Tuple[float, float]:
        """
        Update actor and critic networks using a batch of experiences.
        
        Args:
            batch: Dictionary with 'states', 'actions', 'rewards', 'next_states', 'dones'
            accumulate: If True, accumulate gradients without stepping optimizer
            
        Returns:
            Tuple of (critic_loss, actor_loss)
        """
        # Extract batch data
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        next_states = batch['next_states'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # ============ Update Critic ============
        with torch.no_grad():
            # Get next actions from target actor
            next_actions = self.actor_target(next_states)
            
            # Get target Q-values
            target_q = self.critic_target(next_states, next_actions)
            
            # Compute target: r + gamma * Q_target(s', a')
            target_q = rewards + (1 - dones) * Config.GAMMA * target_q
        
        with autocast(device_type='cuda'):
            current_q = self.critic(states, actions)
            critic_loss = nn.MSELoss()(current_q, target_q)
        
        # Scale loss for gradient accumulation
        if accumulate:
            critic_loss = critic_loss / Config.GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass
        self.critic_scaler.scale(critic_loss).backward()
        
        # Only step if not accumulating or if explicitly told
        if not accumulate:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_scaler.step(self.critic_optimizer)
            self.critic_scaler.update() # Update scaler
            self.critic_optimizer.zero_grad()
        
        # ============ Update Actor ============
        # Freeze critic to save computation
        for param in self.critic.parameters():
            param.requires_grad = False
        
        with autocast(device_type='cuda'):
            actor_actions = self.actor(states)
            actor_loss = -self.critic(states, actor_actions).mean()
        
        # Scale loss for gradient accumulation
        if accumulate:
            actor_loss = actor_loss / Config.GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass
        self.actor_scaler.scale(actor_loss).backward()
        
        # Only step if not accumulating
        if not accumulate:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_scaler.step(self.actor_optimizer)
            self.actor_scaler.update() # Update scaler
            self.actor_optimizer.zero_grad()
            
        # Unfreeze critic
        for param in self.critic.parameters():
            param.requires_grad = True
        
        # ============ Update Target Networks ============
        # Only update targets after actual optimizer step
        if not accumulate:
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            
            # Track statistics
            self.update_count += 1
            
            # Decay noise
            self.noise_scale = max(Config.MIN_NOISE, self.noise_scale * Config.NOISE_DECAY)
        
        # Store losses as Python floats before cleanup
        critic_loss_value = critic_loss.item() * (Config.GRADIENT_ACCUMULATION_STEPS if accumulate else 1)
        actor_loss_value = actor_loss.item() * (Config.GRADIENT_ACCUMULATION_STEPS if accumulate else 1)

        self.actor_loss_history.append(actor_loss_value)
        self.critic_loss_history.append(critic_loss_value)

        # Explicitly delete batch tensors to free GPU memory immediately
        del states, actions, rewards, next_states, dones
        del critic_loss, actor_loss

        return critic_loss_value, actor_loss_value
    
    def _soft_update(self, source: nn.Module, target: nn.Module):
        """
        Soft update target network: θ_target = τ * θ_source + (1 - τ) * θ_target
        
        Args:
            source: Source network
            target: Target network
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                Config.TAU * source_param.data + (1 - Config.TAU) * target_param.data
            )
    
    def save(self, path: str):
        """Save agent state."""
        torch.save({
            'agent_id': self.agent_id,
            'actor_state_dict': self.actor.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'noise_scale': self.noise_scale,
            'update_count': self.update_count,
        }, path)
    
    def load(self, path: str):
        """Load agent state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.agent_id = checkpoint['agent_id']
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.noise_scale = checkpoint['noise_scale']
        self.update_count = checkpoint['update_count']
    
    def clone(self) -> 'DDPGAgent':
        """Create a deep copy of this agent."""
        new_agent = DDPGAgent(agent_id=self.agent_id)

        # CRITICAL FIX: Use state_dict() directly without deepcopy
        # PyTorch's load_state_dict already creates new tensor copies
        # Using deepcopy creates temporary duplicates that linger in memory (~3-5GB per gen)
        new_agent.actor.load_state_dict(self.actor.state_dict())
        new_agent.actor_target.load_state_dict(self.actor_target.state_dict())
        new_agent.critic.load_state_dict(self.critic.state_dict())
        new_agent.critic_target.load_state_dict(self.critic_target.state_dict())
        new_agent.noise_scale = self.noise_scale
        new_agent.is_elite = self.is_elite  # Preserve elite status

        # Clear GPU cache after cloning
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return new_agent
    
    def get_stats(self) -> dict:
        """Get training statistics."""
        return {
            'agent_id': self.agent_id,
            'update_count': self.update_count,
            'noise_scale': self.noise_scale,
            'avg_actor_loss': np.mean(self.actor_loss_history[-100:]) if self.actor_loss_history else 0,
            'avg_critic_loss': np.mean(self.critic_loss_history[-100:]) if self.critic_loss_history else 0,
        }


# Test agent
if __name__ == "__main__":
    print("Testing DDPG Agent...\n")
    
    # Create agent
    agent = DDPGAgent(agent_id=0)
    print(f"Agent ID: {agent.agent_id}")
    print(f"Actor parameters: {sum(p.numel() for p in agent.actor.parameters()):,}")
    print(f"Critic parameters: {sum(p.numel() for p in agent.critic.parameters()):,}")
    print(f"Device: {agent.device}")
    
    # Test action selection
    print("\n--- Testing Action Selection ---")
    dummy_state = np.random.randn(Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL)
    
    # Without noise
    action_no_noise = agent.select_action(dummy_state, add_noise=False)
    print(f"Action shape: {action_no_noise.shape}")
    print(f"Coefficients range (no noise): [{action_no_noise[:, 0].min():.2f}, {action_no_noise[:, 0].max():.2f}]")
    print(f"Sale targets range (no noise): [{action_no_noise[:, 1].min():.2f}, {action_no_noise[:, 1].max():.2f}]")
    
    # With noise
    action_with_noise = agent.select_action(dummy_state, add_noise=True)
    print(f"Coefficients range (with noise): [{action_with_noise[:, 0].min():.2f}, {action_with_noise[:, 0].max():.2f}]")
    print(f"Sale targets range (with noise): [{action_with_noise[:, 1].min():.2f}, {action_with_noise[:, 1].max():.2f}]")
    
    # Test update with dummy batch
    print("\n--- Testing Network Update ---")
    batch_size = 4  # Reduced from 32 due to GPU memory constraints
    dummy_batch = {
        'states': torch.randn(batch_size, Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL),
        'actions': torch.randn(batch_size, Config.NUM_INVESTABLE_STOCKS, Config.ACTION_DIM),
        'rewards': torch.randn(batch_size, 1),
        'next_states': torch.randn(batch_size, Config.CONTEXT_WINDOW_DAYS, Config.TOTAL_COLUMNS, Config.FEATURES_PER_CELL),
        'dones': torch.zeros(batch_size, 1),
    }
    
    print(f"Batch size: {batch_size}")
    print("Performing update...")
    
    import time
    start = time.time()
    critic_loss, actor_loss = agent.update(dummy_batch)
    elapsed = time.time() - start
    
    print(f"Update time: {elapsed:.3f} seconds")
    print(f"Critic loss: {critic_loss:.4f}")
    print(f"Actor loss: {actor_loss:.4f}")
    print(f"Noise scale after update: {agent.noise_scale:.4f}")
    
    # Test cloning
    print("\n--- Testing Agent Cloning ---")
    cloned_agent = agent.clone()
    print(f"Cloned agent ID: {cloned_agent.agent_id}")
    
    # Verify parameters are the same
    actor_params_match = all(
        torch.allclose(p1, p2) 
        for p1, p2 in zip(agent.actor.parameters(), cloned_agent.actor.parameters())
    )
    print(f"Actor parameters match: {actor_params_match}")
    
    # Test save/load
    print("\n--- Testing Save/Load ---")
    save_path = "test_agent.pth"
    agent.save(save_path)
    print(f"Saved to {save_path}")
    
    new_agent = DDPGAgent(agent_id=1)
    new_agent.load(save_path)
    print(f"Loaded agent ID: {new_agent.agent_id}")
    print(f"Update count: {new_agent.update_count}")
    
    # Clean up
    import os
    os.remove(save_path)
    print("Cleaned up test file")
    
    print("\n✓ DDPG Agent test complete!")