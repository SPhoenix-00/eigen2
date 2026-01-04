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
        
        # Enable debug output for ROCm on first training update only
        # (disabled during eval/validation to reduce noise)
        if self.use_rocm_mode:
            self.actor_target._debug_rocm = True
            self.actor._debug_rocm = True
            self._debug_enabled = True
            # Disable after first update to reduce noise
            self._debug_enabled = True
        
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

    def _forward_chunked(self, network, states, chunk_size=32):
        """
        Splits a large batch into smaller chunks to prevent ROCm kernel crashes.
        Mathematically identical to running the full batch at once.
        
        Args:
            network: The network to run forward pass on (Actor or Critic)
            states: Input states tensor [batch, ...]
            chunk_size: Size of each chunk (default 32 for ROCm stability)
            
        Returns:
            Output tensor with same batch dimension as input
        """
        # If batch is small enough, just run it normally
        if states.shape[0] <= chunk_size:
            return network(states)
        
        outputs = []
        # Iterate through the batch in chunks
        for i in range(0, states.shape[0], chunk_size):
            # Slice the batch
            batch_chunk = states[i:i + chunk_size]
            
            # Run forward pass on the chunk
            # Note: This keeps the graph connected for Main Actor,
            # and works fine for Target Actor (no_grad) too.
            chunk_output = network(batch_chunk)
            outputs.append(chunk_output)
            
        # Stitch the results back together
        return torch.cat(outputs, dim=0)

    def update(self, batch: dict, accumulate: bool = False) -> Tuple[float, float]:
        """
        Update actor and critic networks using a batch of experiences.
        
        Args:
            batch: Dictionary with 'states', 'actions', 'rewards', 'next_states', 'dones'
            accumulate: If True, accumulate gradients without stepping optimizer
            
        Returns:
            Tuple of (critic_loss, actor_loss)
        """
        # Track if this is the first update call (for minimal logging only)
        is_first_update = (self.update_count == 0)
        
        # Disable SymLog debug output after first update to reduce noise
        if self.update_count == 1 and hasattr(self, '_debug_enabled') and self._debug_enabled:
            if hasattr(self.actor_target, '_debug_rocm'):
                self.actor_target._debug_rocm = False
            if hasattr(self.actor, '_debug_rocm'):
                self.actor._debug_rocm = False
        
        # CRITICAL FOR ROCm: Networks are already on GPU, never move them
        # Just ensure batch tensors are on the same device (GPU)
        # Use non_blocking=False for ROCm to prevent memory access faults
        transfer_mode = not self.use_rocm_mode  # non_blocking only for non-ROCm
        
        states = batch['states'].to(self.device, non_blocking=transfer_mode)
        actions = batch['actions'].to(self.device, non_blocking=transfer_mode)
        rewards = batch['rewards'].to(self.device, non_blocking=transfer_mode)
        next_states = batch['next_states'].to(self.device, non_blocking=transfer_mode)
        dones = batch['dones'].to(self.device, non_blocking=transfer_mode)
        
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
        
        # Ensure actor_target is in eval mode (should be, but verify)
        if self.actor_target.training:
            self.actor_target.eval()
        
        with torch.no_grad():
            # Get next actions from target actor (already on GPU)
            # CRITICAL ROCm FIX: Use micro-batching to prevent memory access faults
            # Process large batches in chunks to avoid ROCm attention kernel crashes
            # Mathematically identical to full batch, but physically processes in smaller chunks
            if self.use_rocm_mode:
                next_actions = self._forward_chunked(self.actor_target, next_states, chunk_size=32)
            else:
                next_actions = self.actor_target(next_states)
            
            # Get target Q-values (already on GPU)
            # CRITICAL ROCm FIX: Also chunk the critic_target forward pass
            if self.use_rocm_mode:
                # Chunking Critic Target explicitly (takes 2 args: states and actions)
                next_q_values_list = []
                chunk_size = 32
                for i in range(0, next_states.shape[0], chunk_size):
                    s_chunk = next_states[i:i+chunk_size]
                    a_chunk = next_actions[i:i+chunk_size]
                    next_q_values_list.append(self.critic_target(s_chunk, a_chunk))
                target_q = torch.cat(next_q_values_list, dim=0)
            else:
                target_q = self.critic_target(next_states, next_actions)
            
            # Compute target: r + gamma * Q_target(s', a')
            target_q = rewards + (1 - dones) * Config.GAMMA * target_q
        
        # Use autocast for GPU (works on both CUDA and ROCm)
        with autocast(device_type='cuda'):
            # CRITICAL ROCm FIX: Chunk the main critic forward pass to prevent memory access faults
            # This preserves gradients - Autograd engine stitches the graph through torch.cat
            if self.use_rocm_mode:
                # Chunked execution for main critic (preserves gradients)
                current_q_list = []
                chunk_size = 32
                
                for i in range(0, states.shape[0], chunk_size):
                    # Slice the batch
                    s_chunk = states[i:i + chunk_size]
                    a_chunk = actions[i:i + chunk_size]
                    
                    # Run forward pass (Gradients are automatically tracked!)
                    q_chunk = self.critic(s_chunk, a_chunk)
                    current_q_list.append(q_chunk)
                
                # Stitch results together
                # Autograd will backpropagate through this 'cat' operation correctly
                current_q = torch.cat(current_q_list, dim=0)
            else:
                current_q = self.critic(states, actions)
            
            critic_loss = nn.MSELoss()(current_q, target_q)
        
        # Scale loss for gradient accumulation
        if accumulate:
            critic_loss = critic_loss / Config.GRADIENT_ACCUMULATION_STEPS
        
        # Backward pass
        self.critic_scaler.scale(critic_loss).backward()
        
        # Only step if not accumulating
        if not accumulate:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
            self.critic_scaler.step(self.critic_optimizer)
            self.critic_scaler.update()
            self.critic_optimizer.zero_grad()
            
            # For ROCm: Synchronize after optimizer step (critical for stability)
            if self.use_rocm_mode:
                torch.cuda.synchronize()
        
        # ============ Update Actor ============
        # Freeze critic to save computation
        for param in self.critic.parameters():
            param.requires_grad = False
        
        with autocast(device_type='cuda'):
            # CRITICAL ROCm FIX: Chunk the main actor forward pass to prevent memory access faults
            # This preserves gradients - Autograd engine stitches the graph through torch.cat
            if self.use_rocm_mode:
                # Chunked execution for main actor (preserves gradients)
                actor_actions = self._forward_chunked(self.actor, states, chunk_size=32)
            else:
                actor_actions = self.actor(states)
            
            # CRITICAL ROCm FIX: Also chunk the critic call that uses actor_actions
            # We need to pass both states and new actor_actions to the critic in chunks
            if self.use_rocm_mode:
                # Chunked execution for critic with actor_actions (preserves gradients)
                actor_loss_list = []
                chunk_size = 32
                
                for i in range(0, states.shape[0], chunk_size):
                    s_chunk = states[i:i + chunk_size]
                    a_chunk = actor_actions[i:i + chunk_size]
                    
                    # Get Q-value for this chunk
                    q_chunk = self.critic(s_chunk, a_chunk)
                    actor_loss_list.append(q_chunk)
                
                # Combine and calculate mean loss
                # Autograd will backpropagate through this 'cat' operation correctly
                full_q_values = torch.cat(actor_loss_list, dim=0)
                actor_loss = -full_q_values.mean()
            else:
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
            self.actor_scaler.update()
            self.actor_optimizer.zero_grad()
            
            # For ROCm: Synchronize after optimizer step (critical for stability)
            if self.use_rocm_mode:
                torch.cuda.synchronize()
        
        # Unfreeze critic
        for param in self.critic.parameters():
            param.requires_grad = True
        
        # ============ Update Target Networks ============
        # Only update targets after actual optimizer step
        if not accumulate:
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic, self.critic_target)
            
            # For ROCm: Synchronize after soft update (critical for stability)
            if self.use_rocm_mode:
                torch.cuda.synchronize()
            
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
        
        # For ROCm: Final synchronization and cache clear (critical for stability)
        if self.use_rocm_mode:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

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