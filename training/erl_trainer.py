"""
ERL Trainer for Project Eigen 2
Evolutionary Reinforcement Learning training loop
"""

import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import time
from torch.utils.tensorboard import SummaryWriter
import os
import json

from data.loader import StockDataLoader
from environment.trading_env import TradingEnvironment
from models.ddpg_agent import DDPGAgent
from models.replay_buffer import ReplayBuffer
from erl.genetic_ops import create_next_generation
from utils.config import Config
from utils.display import print_generation_summary, print_final_summary, plot_fitness_progress


class ERLTrainer:
    """
    Evolutionary Reinforcement Learning Trainer.
    Manages population, training, and evolution.
    """
    
    def __init__(self, data_loader: StockDataLoader):
        """
        Initialize ERL trainer.
        
        Args:
            data_loader: Loaded data with train/val splits
        """
        self.data_loader = data_loader
        
        # Compute normalization stats ONCE and cache them
        print("Computing and caching normalization statistics...")
        self.normalization_stats = data_loader.compute_normalization_stats()
        
        # Initialize population
        print(f"Initializing population of {Config.POPULATION_SIZE} agents...")
        self.population = [DDPGAgent(agent_id=i) for i in range(Config.POPULATION_SIZE)]
        
        # Shared replay buffer
        self.replay_buffer = ReplayBuffer()
        
        # Training environments (one per agent for parallel rollouts)
        self.train_start_idx = Config.CONTEXT_WINDOW_DAYS
        self.train_end_idx = len(data_loader.train_indices)
        
        # Validation environment
        self.val_start_idx = len(data_loader.train_indices)
        self.val_end_idx = len(data_loader.data_array)
        
        # Logging
        self.writer = SummaryWriter(log_dir=str(Config.LOG_DIR))
        self.start_generation = 0
        self.generation = 0
        self.best_fitness = float('-inf')
        self.best_agent = None
        
        # Statistics
        self.fitness_history = []
        self.generation_times = []
        
        print(f"Training range: days {self.train_start_idx} to {self.train_end_idx}")
        print(f"Validation range: days {self.val_start_idx} to {self.val_end_idx}")
    
    def run_episode(self, agent: DDPGAgent, start_idx: int, end_idx: int, 
                   training: bool = True) -> Tuple[float, Dict]:
        """
        Run one episode with an agent.
        
        Args:
            agent: Agent to run
            start_idx: Starting day index (first day of trading period)
            end_idx: Ending day index (includes settlement period)
            training: Whether this is training (adds to replay buffer)
            
        Returns:
            Tuple of (cumulative_reward, episode_info)
        """
        # Calculate trading end (when model stops opening new positions)
        trading_end_idx = start_idx + Config.TRADING_PERIOD_DAYS
        
        # Create environment for this episode
        env = TradingEnvironment(
            data_array=self.data_loader.data_array,
            dates=self.data_loader.dates,
            normalization_stats=self.normalization_stats,  # Use cached stats
            start_idx=start_idx,
            end_idx=end_idx,
            trading_end_idx=trading_end_idx
        )
        
        # Reset environment
        state, info = env.reset()
        cumulative_reward = 0.0
        steps = 0
        
        # Run episode
        while True:
            # Select action
            action = agent.select_action(state, add_noise=training)
            
            # Take step
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store transition in replay buffer (if training)
            if training:
                self.replay_buffer.add(
                    state=state,
                    action=action,
                    reward=reward,
                    next_state=next_state,
                    done=float(terminated or truncated)
                )
            
            cumulative_reward += reward
            steps += 1
            
            # Move to next state
            state = next_state
            
            if terminated or truncated:
                break
        
        # Get episode summary
        episode_summary = env.get_episode_summary()
        episode_summary['steps'] = steps
        
        # CRITICAL: Use cumulative_reward from environment
        # This includes ALL penalties (inaction, losses, etc.)
        final_fitness = float(cumulative_reward)
        
        # Apply zero-trades penalty if no trades were made
        if episode_summary['num_trades'] == 0:
            final_fitness -= Config.ZERO_TRADES_PENALTY
            print(f"WARNING: Agent made 0 trades. Applying penalty of -{Config.ZERO_TRADES_PENALTY}. Final fitness: {final_fitness}")
        
        # Validation: inactive agents should have catastrophic negative fitness
        if episode_summary['num_trades'] == 0:
            expected_catastrophic = -10000
            if final_fitness > expected_catastrophic:
                print(f"WARNING: Inactive agent has fitness {final_fitness}, should be < {expected_catastrophic}")
        
        return final_fitness, episode_summary
    
    def evaluate_population(self) -> Tuple[List[float], Dict]:
        """
        Evaluate all agents in population (fitness scores).
        
        Returns:
            Tuple of (fitness_scores, aggregate_stats)
        """
        fitness_scores = []
        all_episode_stats = []
        
        print(f"\n--- Generation {self.generation}: Evaluating Population ---")
        
        for agent in tqdm(self.population, desc="Evaluating agents"):
            # Calculate episode indices
            # Need: context (504) + trading (125) + settlement (20) = 649 days total
            total_days_needed = Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
            
            max_start = self.train_end_idx - total_days_needed
            if max_start <= self.train_start_idx:
                raise ValueError(f"Not enough training data: need {total_days_needed} days")
            
            start_idx = np.random.randint(self.train_start_idx, max_start)
            end_idx = start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
            
            # Run episode
            fitness, episode_info = self.run_episode(
                agent=agent,
                start_idx=start_idx,
                end_idx=end_idx,
                training=True
            )
            
            fitness_scores.append(fitness)
            all_episode_stats.append(episode_info)
        
        # Ensure fitness_scores are all plain floats
        fitness_scores = [float(f) for f in fitness_scores]
        
        # Aggregate statistics
        aggregate_stats = {
            'total_trades': int(sum(s['num_trades'] for s in all_episode_stats)),
            'avg_trades_per_agent': float(sum(s['num_trades'] for s in all_episode_stats) / len(all_episode_stats)),
            'total_wins': int(sum(s['num_wins'] for s in all_episode_stats)),
            'total_losses': int(sum(s['num_losses'] for s in all_episode_stats)),
            'avg_win_rate': float(sum(s['win_rate'] for s in all_episode_stats if s['num_trades'] > 0) / len([s for s in all_episode_stats if s['num_trades'] > 0])) if any(s['num_trades'] > 0 for s in all_episode_stats) else 0.0,
            'agents_with_positive_fitness': int(sum(1 for f in fitness_scores if f > 0)),
        }
        
        return (fitness_scores, aggregate_stats)
    
    def train_population(self):
        """Train all agents using shared replay buffer with gradient accumulation."""
        if not self.replay_buffer.is_ready():
            print(f"Buffer not ready: {len(self.replay_buffer)} / {Config.MIN_BUFFER_SIZE}")
            return
        
        print(f"\n--- Training Population (Buffer: {len(self.replay_buffer)}) ---")
        
        # Train each agent
        for agent in tqdm(self.population, desc="Training agents"):
            actor_losses = []
            critic_losses = []
            
            # Multiple gradient steps per agent
            for step in range(Config.GRADIENT_STEPS_PER_GENERATION):
                # Gradient accumulation loop
                for accum_step in range(Config.GRADIENT_ACCUMULATION_STEPS):
                    # Sample batch
                    batch = self.replay_buffer.sample(Config.BATCH_SIZE)
                    
                    # Update with gradient accumulation
                    is_last_accum = (accum_step == Config.GRADIENT_ACCUMULATION_STEPS - 1)
                    critic_loss, actor_loss = agent.update(batch, accumulate=not is_last_accum)
                    
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
            
            # Log agent stats
            if agent.agent_id == 0:  # Log first agent as representative
                self.writer.add_scalar('Train/Actor_Loss', np.mean(actor_losses), self.generation)
                self.writer.add_scalar('Train/Critic_Loss', np.mean(critic_losses), self.generation)
    
    def evolve_population(self, fitness_scores: List[float]):
        """
        Evolve population using genetic algorithm.
        
        Args:
            fitness_scores: Fitness for each agent
        """
        print(f"\n--- Evolving Population ---")
        
        # Create next generation
        self.population = create_next_generation(self.population, fitness_scores)
        
        print(f"Generation {self.generation} -> {self.generation + 1}")
    
    def validate_best_agent(self) -> Dict:
        """
        Validate best agent on validation set.
        
        Returns:
            Validation results
        """
        if self.best_agent is None:
            return {}
        
        print(f"\n--- Validating Best Agent ---")
        
        # Validation uses the validation period
        # end_idx includes settlement period
        total_days_needed = Config.CONTEXT_WINDOW_DAYS + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
        
        if self.val_end_idx - self.val_start_idx < Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS:
            print(f"Warning: Validation set too small, using partial period")
            end_idx = self.val_end_idx
        else:
            end_idx = self.val_start_idx + Config.TRADING_PERIOD_DAYS + Config.SETTLEMENT_PERIOD_DAYS
        
        # Run on validation data
        fitness, episode_info = self.run_episode(
            agent=self.best_agent,
            start_idx=self.val_start_idx,
            end_idx=end_idx,
            training=False
        )
        
        print(f"Validation fitness: {fitness:.2f}")
        print(f"Validation win rate: {episode_info['win_rate']:.1%}")
        print(f"Validation trades: {episode_info['num_trades']}")
        
        return {
            'fitness': fitness,
            'win_rate': episode_info['win_rate'],
            'num_trades': episode_info['num_trades'],
            'avg_reward_per_trade': episode_info['avg_reward_per_trade']
        }
    
    def save_checkpoint(self):
        """Saves the entire training state to a checkpoint directory."""
        checkpoint_dir = Config.CHECKPOINT_DIR
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n--- Saving checkpoint for Generation {self.generation} ---")

        # 1. Save best agent
        if self.best_agent is not None:
            best_path = checkpoint_dir / "best_agent.pth"
            self.best_agent.save(str(best_path))
        
        # 2. Save population
        pop_dir = checkpoint_dir / "population"
        pop_dir.mkdir(exist_ok=True)
        for agent in self.population:
            agent_path = pop_dir / f"agent_{agent.agent_id}.pth"
            agent.save(str(agent_path))
        
        # 3. Save the replay buffer
        #buffer_path = checkpoint_dir / "replay_buffer.pkl"
        #self.replay_buffer.save(str(buffer_path))
        
        # 4. Save the trainer state
        trainer_state = {
            'generation': self.generation,
            'best_fitness': self.best_fitness
        }
        state_path = checkpoint_dir / "trainer_state.json"
        with open(state_path, 'w') as f:
            json.dump(trainer_state, f, indent=4)

        print(f"✓ Checkpoint saved to {checkpoint_dir}")

    def load_checkpoint(self):
        """Loads the entire training state from the checkpoint directory."""
        checkpoint_dir = Config.CHECKPOINT_DIR
        print(f"\n--- Loading checkpoint from {checkpoint_dir} ---")

        # 1. Check if directory exists
        if not checkpoint_dir.exists():
            print("! No checkpoint directory found. Starting new training.")
            return

        # 2. Load Population
        pop_dir = checkpoint_dir / "population"
        if pop_dir.exists():
            try:
                for i, agent in enumerate(self.population):
                    agent_path = pop_dir / f"agent_{agent.agent_id}.pth"
                    if agent_path.exists():
                        agent.load(str(agent_path))
                    else:
                        print(f"! Warning: Missing agent file: {agent_path}")
                print(f"✓ Loaded {len(self.population)} agents from population.")
            except Exception as e:
                print(f"❌ Error loading agent population: {e}. Starting with new agents.")
        else:
            print("! Population directory not found. Starting with new agents.")

        # 3. Load Best Agent
        best_agent_path = checkpoint_dir / "best_agent.pth"
        if best_agent_path.exists():
            try:
                # Re-initialize best_agent before loading
                self.best_agent = DDPGAgent(agent_id='best') 
                self.best_agent.load(str(best_agent_path))
                print("✓ Loaded best agent.")
            except Exception as e:
                print(f"❌ Error loading best agent: {e}.")
        
        # 4. Load Replay Buffer
        """ buffer_path = checkpoint_dir / "replay_buffer.pkl"
        if buffer_path.exists():
            try:
                self.replay_buffer = ReplayBuffer.load(str(buffer_path))
                print(f"✓ Loaded replay buffer (Size: {len(self.replay_buffer)})")
            except Exception as e:
                print(f"❌ Error loading replay buffer: {e}. Starting with empty buffer.")
        else:
            print("! Replay buffer not found. Starting with empty buffer.") """
        
        # 5. Load Trainer State
        state_path = checkpoint_dir / "trainer_state.json"
        if state_path.exists():
            try:
                with open(state_path, 'r') as f:
                    trainer_state = json.load(f)
                
                # This is the key part:
                self.start_generation = trainer_state.get('generation', 0) + 1
                self.best_fitness = trainer_state.get('best_fitness', float('-inf'))
                print(f"✓ Resuming from Generation {self.start_generation}")
                print(f"✓ Loaded best fitness: {self.best_fitness:.2f}")
            except Exception as e:
                print(f"❌ Error loading trainer state: {e}. Starting from scratch.")
        else:
            print("! Trainer state file not found. Starting from generation 0.")
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print("Starting ERL Training")
        print("="*60)
        
        # Use start_generation for the loop
        for gen in range(self.start_generation, Config.NUM_GENERATIONS):
            self.generation = gen  # Keep this to track the *current* gen
            gen_start_time = time.time()
            
            print(f"\n{'='*60}")
            print(f"Generation {gen + 1} / {Config.NUM_GENERATIONS}")
            print(f"{'='*60}")
            
            # 1. Evaluate population (collect experiences)
            fitness_scores, pop_stats = self.evaluate_population()
            
            # Track statistics
            self.fitness_history.append(fitness_scores)
            mean_fitness = np.mean(fitness_scores)
            max_fitness = np.max(fitness_scores)
            min_fitness = np.min(fitness_scores)
            
            print(f"\nFitness statistics:")
            print(f"  Mean: {mean_fitness:.2f}")
            print(f"  Max: {max_fitness:.2f}")
            print(f"  Min: {min_fitness:.2f}")
            print(f"  Std: {np.std(fitness_scores):.2f}")
            
            # Log to tensorboard
            self.writer.add_scalar('Fitness/Mean', mean_fitness, gen)
            self.writer.add_scalar('Fitness/Max', max_fitness, gen)
            self.writer.add_scalar('Fitness/Min', min_fitness, gen)
            self.writer.add_scalar('Fitness/Std', np.std(fitness_scores), gen)
            
            # Update best agent
            best_idx = None
            for i, f in enumerate(fitness_scores):
                if f == max_fitness:
                    best_idx = i
                    break
            
            if best_idx is not None and fitness_scores[best_idx] > self.best_fitness:
                self.best_fitness = fitness_scores[best_idx]
                self.best_agent = self.population[best_idx].clone()
            
            # Print comprehensive summary
            print_generation_summary(
                gen=gen,
                total_gens=Config.NUM_GENERATIONS,
                fitness_scores=fitness_scores,
                pop_stats=pop_stats,
                buffer_size=len(self.replay_buffer),
                best_fitness=self.best_fitness,
                gen_time=0,  # Will update below
                avg_gen_time=np.mean(self.generation_times) if self.generation_times else 0
            )
            
            # 2. Train agents using replay buffer
            self.train_population()
            
            # 3. Evolve population
            self.evolve_population(fitness_scores)
            
            # 4. Validate best agent periodically
            if (gen + 1) % Config.LOG_FREQUENCY == 0:
                val_results = self.validate_best_agent()
                if val_results:
                    self.writer.add_scalar('Validation/Fitness', val_results['fitness'], gen)
                    self.writer.add_scalar('Validation/WinRate', val_results['win_rate'], gen)
            
            # 5. Save checkpoint periodically
            if (gen + 1) % Config.SAVE_FREQUENCY == 0:
                self.save_checkpoint()
            
            # Generation time
            gen_time = time.time() - gen_start_time
            self.generation_times.append(gen_time)
            
            # Show progress plot every 5 generations
            if (gen + 1) % 5 == 0:
                plot_fitness_progress(self.fitness_history)
            
            # Buffer stats
            buffer_stats = self.replay_buffer.get_stats()
            self.writer.add_scalar('Buffer/Size', buffer_stats['size'], gen)
            self.writer.add_scalar('Buffer/Utilization', buffer_stats['utilization'], gen)
        
        # Final save
        self.save_checkpoint()
        
        # Final summary
        print_final_summary(self)
        
        self.writer.close()


# Main execution
if __name__ == "__main__":
    print("Initializing Project Eigen 2 Training...\n")
    
    # Load data
    print("Loading data...")
    loader = StockDataLoader()
    data_array, stats = loader.load_and_prepare()
    
    # Create trainer
    trainer = ERLTrainer(loader)
    
    # Start training
    trainer.train()
    
    print("\n✓ Training complete!")