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
import json
import os

from models.networks import Actor, Critic
from utils.config import Config

# #region debug log
DEBUG_LOG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.cursor', 'debug.log')
def _debug_log(location, message, data, hypothesis_id=None):
    try:
        log_entry = {
            "sessionId": "debug-session",
            "runId": "pre-fix",
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(__import__('time').time() * 1000)
        }
        if hypothesis_id:
            log_entry["hypothesisId"] = hypothesis_id
        with open(DEBUG_LOG_PATH, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except:
        pass
# #endregion


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
        
        # ROCm-specific: Detect backend for special handling
        from utils.device import get_gpu_backend
        gpu_backend = get_gpu_backend()
        self.use_rocm_mode = (gpu_backend == "ROCm")
        
        # CRITICAL FOR ROCm: Initialize networks directly on GPU and NEVER move them
        # Moving networks to/from GPU causes memory access faults on ROCm
        # So we create them on GPU and keep them there permanently
        self.actor = Actor().to(self.device)
        self.actor_target = Actor().to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic = Critic().to(self.device)
        self.critic_target = Critic().to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # For ROCm: Synchronize after network creation to ensure GPU operations complete
        if self.use_rocm_mode and torch.cuda.is_available():
            torch.cuda.synchronize()
        
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

        # GradScaler for mixed precision - only enabled on CUDA
        scaler_device = 'cuda' if self.device.type == 'cuda' else 'cpu'
        self.actor_scaler = GradScaler(scaler_device, enabled=(self.device.type == 'cuda'))
        self.critic_scaler = GradScaler(scaler_device, enabled=(self.device.type == 'cuda'))
        
        # Exploration noise
        self.noise_scale = Config.NOISE_SCALE
        
        # Training statistics
        # CRITICAL FIX: Use deque with maxlen to prevent unbounded growth (~10-20MB per gen)
        # Only last 1000 values are used by get_stats() anyway
        self.actor_loss_history = deque(maxlen=1000)
        self.critic_loss_history = deque(maxlen=1000)
        self.update_count = 0

    def move_to_device(self, device: torch.device, recreate_optimizers: bool = True):
        """
        Move agent to a new device, properly handling optimizer references.

        CRITICAL FOR ROCm: Networks are initialized on GPU and NEVER moved.
        Moving networks causes memory access faults on ROCm, so this is a no-op
        if the agent is already on the target device (which it always is for ROCm).

        Args:
            device: Target device (e.g., torch.device('cpu') or torch.device('cuda'))
            recreate_optimizers: If True, recreate optimizers to reference new params.
                                 Set False for inference-only (saves time but breaks training).
        """
        # CRITICAL FOR ROCm: Never move networks - they're already on GPU permanently
        # Moving networks to/from GPU causes memory access faults on ROCm
        if self.use_rocm_mode:
            # For ROCm, networks are always on GPU (self.device = Config.DEVICE = GPU)
            # If someone tries to move to GPU, it's already there - no-op
            # If someone tries to move to CPU, we refuse (would cause faults later)
            if device.type == 'cuda':
                return  # Already on GPU, no-op
            else:
                # Refuse to move ROCm agents to CPU - would break training
                # Networks must stay on GPU for ROCm
                return  # No-op: keep on GPU
        
        # Standard device movement for non-ROCm
        # Compare by type, not object equality
        # torch.device('cuda') != torch.device('cuda:0') but both are cuda device 0
        same_type = self.device.type == device.type
        if same_type and device.type == 'cpu':
            return  # Already on CPU
        if same_type and device.type == 'cuda':
            # Both are CUDA - check if same GPU index (None means default = 0)
            current_idx = self.device.index if self.device.index is not None else 0
            target_idx = device.index if device.index is not None else 0
            if current_idx == target_idx:
                return  # Already on same CUDA device

        # Move networks to new device
        self.actor = self.actor.to(device)
        self.actor_target = self.actor_target.to(device)
        self.critic = self.critic.to(device)
        self.critic_target = self.critic_target.to(device)

        # Update device tracking
        old_device = self.device
        self.device = device

        if recreate_optimizers:
            # Recreate optimizers to reference the new parameters
            # This drops references to old GPU tensors, allowing them to be freed
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

            # Recreate GradScalers for the new device
            scaler_device = 'cuda' if device.type == 'cuda' else 'cpu'
            self.actor_scaler = GradScaler(scaler_device, enabled=(device.type == 'cuda'))
            self.critic_scaler = GradScaler(scaler_device, enabled=(device.type == 'cuda'))

    def select_action(self, state: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Select action using current policy.

        Args:
            state: State observation [context_days, num_columns, num_features]
            add_noise: Whether to add exploration noise

        Returns:
            Action [108, 2]
        """
        self.actor.eval()
        
        with torch.no_grad():
            # Convert to tensor and add batch dimension
            # For ROCm, device is CPU; for others, use configured device
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

            # Safety clip for coefficients after noise addition
            # NOTE: The Actor network now clamps coefficients to [0, 100] internally via torch.clamp,
            # ensuring the Critic sees the same range during training as inference.
            # This numpy clip is a safety net for noise that might push values outside bounds.
            action[:, 0] = np.clip(action[:, 0], 0, 100)

        self.actor.train()
        return action

    def select_actions_batch(self, states: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """
        Batched action selection for faster evaluation.

        Uses GPU if available for fast batch inference (except ROCm which uses CPU).
        CPU→GPU transfer is amortized over the entire batch (125 items).

        Args:
            states: Batch of state observations [batch_size, context_days, num_columns, features]
            add_noise: Whether to add exploration noise (typically False for evaluation)

        Returns:
            actions: Batch of actions [batch_size, num_stocks, 2]
        """
        self.actor.eval()

        with torch.no_grad():
            # Transfer to device and run batch inference
            # For ROCm, device is CPU; for others, use configured device
            states_tensor = torch.FloatTensor(states).to(self.device)

            # Forward pass through actor network
            actions_tensor = self.actor(states_tensor)
            actions = actions_tensor.cpu().numpy()

            # Add noise if requested (vectorized)
            if add_noise:
                noise = np.random.normal(0, self.noise_scale, actions.shape)
                actions = actions + noise
                actions[:, :, 0] = np.maximum(actions[:, :, 0], 0)
                actions[:, :, 1] = np.clip(actions[:, :, 1], Config.MIN_SALE_TARGET, Config.MAX_SALE_TARGET)

            # Safety clip for coefficients
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
        # DEBUG: Track if this is the first update call
        is_first_update = (self.update_count == 0)
        
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): Agent {self.agent_id}, first update call")
            print(f"      Batch keys: {list(batch.keys())}")
            for k, v in batch.items():
                print(f"      {k}: shape={v.shape}, device={v.device}, contiguous={v.is_contiguous()}")
        
        # CRITICAL FOR ROCm: Networks are already on GPU, never move them
        # Just ensure batch tensors are on the same device (GPU)
        # Use non_blocking=False for ROCm to prevent memory access faults
        transfer_mode = not self.use_rocm_mode  # non_blocking only for non-ROCm
        
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): Transferring batch tensors to device {self.device}...")
        
        states = batch['states'].to(self.device, non_blocking=transfer_mode)
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): states transferred")
        actions = batch['actions'].to(self.device, non_blocking=transfer_mode)
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): actions transferred")
        rewards = batch['rewards'].to(self.device, non_blocking=transfer_mode)
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): rewards transferred")
        next_states = batch['next_states'].to(self.device, non_blocking=transfer_mode)
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): next_states transferred")
        dones = batch['dones'].to(self.device, non_blocking=transfer_mode)
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): dones transferred")
        
        # CRITICAL FOR ROCm: Check for NaN/Inf values and replace them (ROCm crashes on NaN)
        # This is a safety check in case NaN values slipped through from the DataLoader
        for tensor_name, tensor in [('states', states), ('actions', actions), ('rewards', rewards), 
                                    ('next_states', next_states), ('dones', dones)]:
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                nan_count = torch.isnan(tensor).sum().item()
                inf_count = torch.isinf(tensor).sum().item()
                print(f"    [WARNING] update(): {tensor_name} contains NaN/Inf! NaN: {nan_count}, Inf: {inf_count}")
                print(f"    [WARNING] update(): Replacing NaN/Inf with zeros to prevent ROCm crash...")
                # Replace NaN and Inf with zeros
                if tensor_name == 'next_states':
                    next_states = torch.where(torch.isnan(next_states) | torch.isinf(next_states), 
                                            torch.zeros_like(next_states), next_states)
                elif tensor_name == 'states':
                    states = torch.where(torch.isnan(states) | torch.isinf(states), 
                                        torch.zeros_like(states), states)
                elif tensor_name == 'actions':
                    actions = torch.where(torch.isnan(actions) | torch.isinf(actions), 
                                         torch.zeros_like(actions), actions)
                elif tensor_name == 'rewards':
                    rewards = torch.where(torch.isnan(rewards) | torch.isinf(rewards), 
                                         torch.zeros_like(rewards), rewards)
                elif tensor_name == 'dones':
                    dones = torch.where(torch.isnan(dones) | torch.isinf(dones), 
                                       torch.zeros_like(dones), dones)
        
        # For ROCm: Synchronize after tensor transfer to ensure data is ready
        if self.use_rocm_mode:
            if is_first_update:
                print(f"    [DEBUG] update(): Synchronizing GPU after tensor transfer...")
            torch.cuda.synchronize()
            if is_first_update:
                print(f"    [DEBUG] update(): GPU synchronized")
        
        # ============ Update Critic ============
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): Starting critic update...")
            print(f"    [DEBUG] update(): next_states properties:")
            print(f"      shape={next_states.shape}, device={next_states.device}, dtype={next_states.dtype}")
            print(f"      contiguous={next_states.is_contiguous()}, requires_grad={next_states.requires_grad}")
            print(f"      has_nan={torch.isnan(next_states).any().item()}, has_inf={torch.isinf(next_states).any().item()}")
            print(f"      min={next_states.min().item():.6f}, max={next_states.max().item():.6f}, mean={next_states.mean().item():.6f}")
            print(f"    [DEBUG] update(): actor_target properties:")
            print(f"      device={next(iter(self.actor_target.parameters())).device}")
            print(f"      training mode={self.actor_target.training}")
            # Ensure actor_target is in eval mode (should be, but verify)
            if self.actor_target.training:
                print(f"    [DEBUG] update(): WARNING: actor_target is in training mode, setting to eval...")
                self.actor_target.eval()
                torch.cuda.synchronize()
        
        with torch.no_grad():
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): Entering torch.no_grad() context...")
                print(f"    [DEBUG] update(): About to call actor_target(next_states)...")
                print(f"    [DEBUG] update(): Synchronizing GPU before forward pass...")
                torch.cuda.synchronize()
                print(f"    [DEBUG] update(): GPU synchronized, calling actor_target...")
                print(f"    [DEBUG] update(): next_states memory address: {next_states.data_ptr()}")
            
            # Get next actions from target actor (already on GPU)
            # CRITICAL: This is where the memory access fault occurs
            # ROCm workaround: Create a fresh contiguous copy to ensure proper memory layout
            if self.use_rocm_mode:
                # Ensure tensor is contiguous and create a fresh copy to avoid memory issues
                if not next_states.is_contiguous():
                    if is_first_update:
                        print(f"    [DEBUG] update(): Making next_states contiguous...")
                    next_states = next_states.contiguous()
                    torch.cuda.synchronize()
                
                # Create a fresh copy to break any potential memory aliasing issues
                if is_first_update:
                    print(f"    [DEBUG] update(): Creating fresh copy of next_states for ROCm...")
                next_states_copy = next_states.clone()
                torch.cuda.synchronize()
                
                if is_first_update:
                    print(f"    [DEBUG] update(): Fresh copy created, verifying...")
                    print(f"      Copy shape: {next_states_copy.shape}, device: {next_states_copy.device}")
                    print(f"      Copy contiguous: {next_states_copy.is_contiguous()}")
                    print(f"      Copy has_nan: {torch.isnan(next_states_copy).any().item()}")
                next_states = next_states_copy
            
            try:
                if self.use_rocm_mode and is_first_update:
                    print(f"    [DEBUG] update(): EXECUTING: next_actions = self.actor_target(next_states)")
                next_actions = self.actor_target(next_states)
                if self.use_rocm_mode and is_first_update:
                    print(f"    [DEBUG] update(): actor_target forward pass SUCCESS, shape={next_actions.shape}")
                    torch.cuda.synchronize()
                    print(f"    [DEBUG] update(): GPU synchronized after actor_target")
            except Exception as e:
                if self.use_rocm_mode and is_first_update:
                    print(f"    [ERROR] update(): actor_target forward pass FAILED: {e}")
                    print(f"    [ERROR] update(): Exception type: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()
                raise
            
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): Computing target_q from critic_target...")
            # Get target Q-values (already on GPU)
            target_q = self.critic_target(next_states, next_actions)
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): target_q computed, shape={target_q.shape}")
            
            # Compute target: r + gamma * Q_target(s', a')
            target_q = rewards + (1 - dones) * Config.GAMMA * target_q
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): target_q updated with rewards")
        
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): Computing current_q with autocast...")
        # Use autocast for GPU (works on both CUDA and ROCm)
        with autocast(device_type='cuda'):
            current_q = self.critic(states, actions)
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): current_q computed, shape={current_q.shape}")
            critic_loss = nn.MSELoss()(current_q, target_q)
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): critic_loss computed, value={critic_loss.item()}")
        
        # Scale loss for gradient accumulation
        if accumulate:
            critic_loss = critic_loss / Config.GRADIENT_ACCUMULATION_STEPS
        
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): Starting critic backward pass...")
        # Backward pass
        self.critic_scaler.scale(critic_loss).backward()
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): Critic backward pass completed")
        
        # Only step if not accumulating
        if not accumulate:
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): Clipping critic gradients...")
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): Stepping critic optimizer...")
            self.critic_scaler.step(self.critic_optimizer)
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): Updating critic scaler...")
            self.critic_scaler.update()
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): Zeroing critic gradients...")
            self.critic_optimizer.zero_grad()
            
            # For ROCm: Synchronize after optimizer step
            if self.use_rocm_mode:
                if is_first_update:
                    print(f"    [DEBUG] update(): Synchronizing GPU after critic optimizer step...")
                torch.cuda.synchronize()
                if is_first_update:
                    print(f"    [DEBUG] update(): GPU synchronized after critic step")
        
        # ============ Update Actor ============
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): Starting actor update...")
        
        # Freeze critic to save computation
        for param in self.critic.parameters():
            param.requires_grad = False
        
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): Computing actor_actions with autocast...")
        with autocast(device_type='cuda'):
            actor_actions = self.actor(states)
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): actor_actions computed, shape={actor_actions.shape}")
            actor_loss = -self.critic(states, actor_actions).mean()
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): actor_loss computed, value={actor_loss.item()}")
        
        # Scale loss for gradient accumulation
        if accumulate:
            actor_loss = actor_loss / Config.GRADIENT_ACCUMULATION_STEPS
        
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): Starting actor backward pass...")
        # Backward pass
        self.actor_scaler.scale(actor_loss).backward()
        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): Actor backward pass completed")
        
        # Only step if not accumulating
        if not accumulate:
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): Clipping actor gradients...")
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): Stepping actor optimizer...")
            self.actor_scaler.step(self.actor_optimizer)
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): Updating actor scaler...")
            self.actor_scaler.update()
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): Zeroing actor gradients...")
            self.actor_optimizer.zero_grad()
            
            # For ROCm: Synchronize after optimizer step
            if self.use_rocm_mode:
                if is_first_update:
                    print(f"    [DEBUG] update(): Synchronizing GPU after actor optimizer step...")
                torch.cuda.synchronize()
                if is_first_update:
                    print(f"    [DEBUG] update(): GPU synchronized after actor step")
        
        # Unfreeze critic
        for param in self.critic.parameters():
            param.requires_grad = True
        
        # ============ Update Target Networks ============
        # Only update targets after actual optimizer step
        if not accumulate:
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): Starting soft update of target networks...")
            self._soft_update(self.actor, self.actor_target)
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): actor_target soft updated")
            self._soft_update(self.critic, self.critic_target)
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): critic_target soft updated")
            
            # For ROCm: Synchronize after soft update
            if self.use_rocm_mode:
                if is_first_update:
                    print(f"    [DEBUG] update(): Synchronizing GPU after soft update...")
                torch.cuda.synchronize()
                if is_first_update:
                    print(f"    [DEBUG] update(): GPU synchronized after soft update")
            
            # Track statistics
            self.update_count += 1
            
            # Decay noise
            self.noise_scale = max(Config.MIN_NOISE, self.noise_scale * Config.NOISE_DECAY)
            
            if self.use_rocm_mode and is_first_update:
                print(f"    [DEBUG] update(): Update complete! update_count={self.update_count}")
        
        # Store losses as Python floats before cleanup
        critic_loss_value = critic_loss.item() * (Config.GRADIENT_ACCUMULATION_STEPS if accumulate else 1)
        actor_loss_value = actor_loss.item() * (Config.GRADIENT_ACCUMULATION_STEPS if accumulate else 1)

        self.actor_loss_history.append(actor_loss_value)
        self.critic_loss_history.append(critic_loss_value)

        if self.use_rocm_mode and is_first_update:
            print(f"    [DEBUG] update(): Cleaning up tensors...")
        
        # Explicitly delete batch tensors to free GPU memory immediately
        del states, actions, rewards, next_states, dones
        del critic_loss, actor_loss
        
        # For ROCm: Final synchronization and cache clear
        if self.use_rocm_mode:
            if is_first_update:
                print(f"    [DEBUG] update(): Final GPU synchronization and cache clear...")
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            if is_first_update:
                print(f"    [DEBUG] update(): Cleanup complete, returning losses")

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
        # #region agent log
        _debug_log("ddpg_agent.py:335", "load() called", {
            "path": path,
            "torch_version": torch.__version__,
            "has_weights_only_param": hasattr(torch.load, '__code__') and 'weights_only' in str(torch.load.__code__.co_varnames)
        }, "A")
        # #endregion
        checkpoint = torch.load(path, map_location=self.device)
        # #region agent log
        _debug_log("ddpg_agent.py:336", "load() torch.load succeeded", {
            "checkpoint_keys": list(checkpoint.keys()),
            "has_numpy_objects": any(isinstance(v, np.ndarray) or isinstance(v, np.generic) for v in checkpoint.values() if isinstance(v, (dict, list, tuple)))
        }, "A")
        # #endregion
        self.agent_id = checkpoint['agent_id']
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        self.noise_scale = checkpoint['noise_scale']
        self.update_count = checkpoint['update_count']

        # Safety: Ensure networks are on correct device after loading
        self.actor = self.actor.to(self.device)
        self.actor_target = self.actor_target.to(self.device)
        self.critic = self.critic.to(self.device)
        self.critic_target = self.critic_target.to(self.device)

    def load_weights_only(self, path: str):
        """
        Load only network weights without optimizer states.

        This is used for Hall of Fame injection where we want the champion's
        learned policy but with fresh optimizers that can adapt quickly to
        the current training regime. Loading stale optimizer states causes
        60% slowdown due to mismatched momentum and memory layout issues.

        Args:
            path: Path to checkpoint file
        """
        # #region agent log
        _debug_log("ddpg_agent.py:365", "load_weights_only() called", {
            "path": path,
            "torch_version": torch.__version__,
            "torch_version_major": int(torch.__version__.split('.')[0]) if '.' in torch.__version__ else None,
            "torch_version_minor": int(torch.__version__.split('.')[1]) if '.' in torch.__version__ and len(torch.__version__.split('.')) > 1 else None,
            "file_exists": os.path.exists(path) if path else False
        }, "B")
        # #endregion
        try:
            checkpoint = torch.load(path, map_location=self.device)
            # #region agent log
            _debug_log("ddpg_agent.py:365", "load_weights_only() torch.load succeeded", {
                "checkpoint_keys": list(checkpoint.keys()) if isinstance(checkpoint, dict) else "not_dict"
            }, "B")
            # #endregion
        except Exception as e:
            # #region agent log
            _debug_log("ddpg_agent.py:365", "load_weights_only() torch.load failed", {
                "error_type": type(e).__name__,
                "error_message": str(e),
                "error_has_weights_only": "weights_only" in str(e).lower()
            }, "B")
            # #endregion
            raise
        self.agent_id = checkpoint['agent_id']
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.noise_scale = checkpoint.get('noise_scale', Config.NOISE_SCALE)
        # Note: optimizers remain fresh (initialized in __init__)
        # Note: update_count remains 0 (this agent starts fresh in the population)

        # Safety: Ensure networks are on correct device after loading
        self.actor = self.actor.to(self.device)
        self.actor_target = self.actor_target.to(self.device)
        self.critic = self.critic.to(self.device)
        self.critic_target = self.critic_target.to(self.device)
    
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

        # CRITICAL: Ensure networks are on the correct device after loading state_dict
        # load_state_dict doesn't move tensors - it keeps them on the source device
        # If source agent was on CPU, we need to move to target device (GPU if available)
        target_device = new_agent.device
        new_agent.actor = new_agent.actor.to(target_device)
        new_agent.actor_target = new_agent.actor_target.to(target_device)
        new_agent.critic = new_agent.critic.to(target_device)
        new_agent.critic_target = new_agent.critic_target.to(target_device)

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