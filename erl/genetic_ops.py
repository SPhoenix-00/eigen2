"""
Genetic Algorithm Operations for Project Eigen 2
Selection, crossover, and mutation for ERL population
"""

import torch
import numpy as np
from typing import List, Tuple
import copy

from models.ddpg_agent import DDPGAgent
from utils.config import Config


def tournament_selection(population: List[DDPGAgent], 
                        fitness_scores: List[float],
                        num_parents: int,
                        tournament_size: int = 3) -> List[DDPGAgent]:
    """
    Select parents using tournament selection.
    
    Args:
        population: List of agents
        fitness_scores: Fitness score for each agent
        num_parents: Number of parents to select
        tournament_size: Size of each tournament
        
    Returns:
        List of selected parent agents
    """
    parents = []
    
    for _ in range(num_parents):
        # Random tournament
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        
        # Winner is agent with highest fitness
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        parents.append(population[winner_idx])
    
    return parents


def elitism_selection(population: List[DDPGAgent],
                     fitness_scores: List[float],
                     num_elite: int) -> List[DDPGAgent]:
    """
    Select top performers (elites).
    
    Args:
        population: List of agents
        fitness_scores: Fitness score for each agent
        num_elite: Number of elites to select
        
    Returns:
        List of elite agents
    """
    # Sort by fitness (descending)
    sorted_indices = np.argsort(fitness_scores)[::-1]
    elite_indices = sorted_indices[:num_elite]
    
    elites = [population[i] for i in elite_indices]
    return elites


def crossover(parent1: DDPGAgent, parent2: DDPGAgent, 
             alpha: float = None) -> DDPGAgent:
    """
    Create offspring by blending two parent networks.
    
    Args:
        parent1: First parent agent
        parent2: Second parent agent
        alpha: Blend coefficient. If None, randomly sampled from [CROSSOVER_ALPHA_MIN, CROSSOVER_ALPHA_MAX]
        
    Returns:
        Offspring agent
    """
    if alpha is None:
        alpha = np.random.uniform(Config.CROSSOVER_ALPHA_MIN, Config.CROSSOVER_ALPHA_MAX)
    
    # Create new agent
    offspring = DDPGAgent(agent_id=-1)  # Will be assigned proper ID later
    
    # Blend actor networks: child = alpha * parent1 + (1-alpha) * parent2
    for child_param, p1_param, p2_param in zip(
        offspring.actor.parameters(),
        parent1.actor.parameters(),
        parent2.actor.parameters()
    ):
        child_param.data.copy_(alpha * p1_param.data + (1 - alpha) * p2_param.data)
    
    # Blend critic networks
    for child_param, p1_param, p2_param in zip(
        offspring.critic.parameters(),
        parent1.critic.parameters(),
        parent2.critic.parameters()
    ):
        child_param.data.copy_(alpha * p1_param.data + (1 - alpha) * p2_param.data)
    
    # Copy target networks from blended networks
    offspring.actor_target.load_state_dict(offspring.actor.state_dict())
    offspring.critic_target.load_state_dict(offspring.critic.state_dict())
    
    return offspring


def mutate(agent: DDPGAgent, mutation_rate: float = None, 
          mutation_std: float = None) -> DDPGAgent:
    """
    Mutate an agent's network weights.
    
    Args:
        agent: Agent to mutate
        mutation_rate: Fraction of weights to mutate. If None, uses Config.MUTATION_RATE
        mutation_std: Standard deviation of mutation noise. If None, uses Config.MUTATION_STD
        
    Returns:
        Mutated agent (new instance)
    """
    if mutation_rate is None:
        mutation_rate = Config.MUTATION_RATE
    if mutation_std is None:
        mutation_std = Config.MUTATION_STD
    
    # Clone agent
    mutated = agent.clone()
    
    # Mutate actor
    for param in mutated.actor.parameters():
        if len(param.shape) == 0:  # Skip scalar parameters
            continue
        
        # Create mask for which weights to mutate
        mask = torch.rand_like(param) < mutation_rate
        
        # Add Gaussian noise to selected weights
        noise = torch.randn_like(param) * mutation_std
        param.data += mask.float() * noise
    
    # Mutate critic
    for param in mutated.critic.parameters():
        if len(param.shape) == 0:
            continue
        
        mask = torch.rand_like(param) < mutation_rate
        noise = torch.randn_like(param) * mutation_std
        param.data += mask.float() * noise
    
    # Update target networks
    mutated.actor_target.load_state_dict(mutated.actor.state_dict())
    mutated.critic_target.load_state_dict(mutated.critic.state_dict())
    
    return mutated

def create_next_generation(population: List[DDPGAgent],
                          fitness_scores: List[float]) -> List[DDPGAgent]:
    """
    Create next generation using selection, crossover, and mutation.
    Calculates population segments dynamically using Config.POPULATION_SIZE.
    """
    pop_size = Config.POPULATION_SIZE
    
    # -----------------------------------------------------------------
    # NEW: Calculate segment sizes dynamically based on POPULATION_SIZE
    # -----------------------------------------------------------------
    num_elites = int(pop_size * Config.ELITE_FRAC)
    num_offspring = int(pop_size * Config.OFFSPRING_FRAC)
    
    # Mutants fill the remaining space to guarantee the pop_size is matched
    num_mutants = pop_size - num_elites - num_offspring
    
    if num_elites == 0 and len(population) > 0:
        # Ensure at least one elite if possible, to allow mutation
        num_elites = 1
        # Adjust mutants to compensate
        if num_mutants > 0:
            num_mutants -= 1
        elif num_offspring > 0: # Take from offspring if no mutants
            num_offspring -= 1
        
    # -----------------------------------------------------------------
    
    next_gen = []
    
    # 1. Elitism: Keep top performers
    # Use the new dynamic 'num_elites'
    elites = elitism_selection(population, fitness_scores, num_elites)
    next_gen.extend(elites)
    
    print(f"  Elites: {len(elites)} agents (fitness: {[fitness_scores[population.index(e)] for e in elites[:3]]}...)")
    
    # 2. Crossover: Generate offspring
    # Use the new dynamic 'num_offspring'
    offspring = []
    if num_offspring > 0:
        for _ in range(num_offspring):
            # Select two different parents via tournament
            parent1 = tournament_selection(population, fitness_scores, num_parents=1, tournament_size=3)[0]
            
            # Ensure parent2 is different from parent1
            attempts = 0
            while attempts < 10:
                parent2 = tournament_selection(population, fitness_scores, num_parents=1, tournament_size=3)[0]
                if parent2.agent_id != parent1.agent_id:
                    break
                attempts += 1
            
            if parent2.agent_id == parent1.agent_id and len(population) > 1:
                other_agents = [a for a in population if a.agent_id != parent1.agent_id]
                if other_agents: # Check if other_agents is not empty
                    parent2 = np.random.choice(other_agents)
                else: # Fallback if only one unique agent
                    parent2 = parent1 
            
            child = crossover(parent1, parent2)
            offspring.append(child)
        
    next_gen.extend(offspring)
    print(f"  Offspring: {len(offspring)} agents via crossover")
    
    # 3. Mutation: Create mutants from elites
    # Use the new dynamic 'num_mutants'
    mutants = []
    if num_mutants > 0 and len(elites) > 0: # Must have mutants to create and elites to mutate from
        for _ in range(num_mutants):
            # Select elite to mutate
            elite = elites[np.random.randint(len(elites))]
            mutant = mutate(elite)
            mutants.append(mutant)
    
    next_gen.extend(mutants)
    print(f"  Mutants: {len(mutants)} agents via mutation")
    
    # --- Fill remaining slots if any due to rounding ---
    # This is a robust way to ensure size match
    fill_count = pop_size - len(next_gen)
    if fill_count > 0 and len(elites) > 0:
        print(f"  Filling {fill_count} remaining slots with mutants.")
        for _ in range(fill_count):
            elite = elites[np.random.randint(len(elites))]
            mutant = mutate(elite)
            next_gen.append(mutant)

    # Assign new agent IDs
    for i, agent in enumerate(next_gen):
        agent.agent_id = i
    
    # This assertion will now pass for ANY population size
    assert len(next_gen) == Config.POPULATION_SIZE, \
        f"Population size mismatch: {len(next_gen)} != {Config.POPULATION_SIZE}"
    
    return next_gen

# Test genetic operations
if __name__ == "__main__":
    print("Testing Genetic Algorithm Operations...\n")
    
    # Create a small population
    print("--- Creating Initial Population ---")
    pop_size = 6
    population = [DDPGAgent(agent_id=i) for i in range(pop_size)]
    print(f"Population size: {len(population)}")
    
    # Assign random fitness scores
    fitness_scores = np.random.randn(pop_size) * 100
    fitness_scores = sorted(fitness_scores, reverse=True)  # Sort for clearer demo
    print(f"Fitness scores: {[f'{f:.2f}' for f in fitness_scores]}")
    
    # Test elitism selection
    print("\n--- Testing Elitism Selection ---")
    num_elite = 3
    elites = elitism_selection(population, fitness_scores, num_elite)
    print(f"Selected {len(elites)} elites")
    print(f"Elite IDs: {[e.agent_id for e in elites]}")
    print(f"Elite fitness: {[fitness_scores[population.index(e)] for e in elites]}")
    
    # Test tournament selection
    print("\n--- Testing Tournament Selection ---")
    num_parents = 2
    parents = tournament_selection(population, fitness_scores, num_parents, tournament_size=3)
    print(f"Selected {len(parents)} parents via tournament")
    print(f"Parent IDs: {[p.agent_id for p in parents]}")
    print(f"Parent fitness: {[fitness_scores[population.index(p)] for p in parents]}")
    
    # Test crossover
    print("\n--- Testing Crossover ---")
    parent1, parent2 = parents[0], parents[1]
    offspring = crossover(parent1, parent2, alpha=0.5)
    print(f"Created offspring from parents {parent1.agent_id} and {parent2.agent_id}")
    
    # Verify offspring is different from both parents
    p1_actor_param = list(parent1.actor.parameters())[0]
    p2_actor_param = list(parent2.actor.parameters())[0]
    offspring_actor_param = list(offspring.actor.parameters())[0]
    
    print(f"Parent1 sample weight: {p1_actor_param.flatten()[0].item():.6f}")
    print(f"Parent2 sample weight: {p2_actor_param.flatten()[0].item():.6f}")
    print(f"Offspring sample weight: {offspring_actor_param.flatten()[0].item():.6f}")
    print(f"Offspring is blend: {not torch.allclose(offspring_actor_param, p1_actor_param) and not torch.allclose(offspring_actor_param, p2_actor_param)}")
    
    # Test mutation
    print("\n--- Testing Mutation ---")
    original = population[0]
    mutated = mutate(original, mutation_rate=0.1, mutation_std=0.01)
    
    orig_param = list(original.actor.parameters())[0]
    mut_param = list(mutated.actor.parameters())[0]
    
    diff = (orig_param - mut_param).abs()
    num_changed = (diff > 1e-6).sum().item()
    total_params = orig_param.numel()
    
    print(f"Original agent ID: {original.agent_id}")
    print(f"Mutated agent ID: {mutated.agent_id}")
    print(f"Parameters changed: {num_changed} / {total_params} ({num_changed/total_params*100:.1f}%)")
    print(f"Sample original weight: {orig_param.flatten()[0].item():.6f}")
    print(f"Sample mutated weight: {mut_param.flatten()[0].item():.6f}")
    
    # Test full generation creation
    print("\n--- Testing Full Generation Creation ---")
    # Temporarily override config for testing
    original_config = (Config.NUM_PARENTS, Config.NUM_OFFSPRING, Config.NUM_MUTANTS)
    Config.NUM_PARENTS = 3
    Config.NUM_OFFSPRING = 2
    Config.NUM_MUTANTS = 1
    Config.POPULATION_SIZE = 6
    
    next_gen = create_next_generation(population, fitness_scores)
    
    print(f"\nNext generation size: {len(next_gen)}")
    print(f"Agent IDs: {[a.agent_id for a in next_gen]}")
    
    # Restore config
    Config.NUM_PARENTS, Config.NUM_OFFSPRING, Config.NUM_MUTANTS = original_config
    
    print("\n✓ Genetic algorithm operations test complete!")