"""
Quick test of training loop with minimal configuration
"""

from data.loader import StockDataLoader
from training.erl_trainer import ERLTrainer
from utils.config import Config

# Temporarily override config for quick test
original_config = {
    'NUM_GENERATIONS': Config.NUM_GENERATIONS,
    'POPULATION_SIZE': Config.POPULATION_SIZE,
    'EPISODE_LENGTH': Config.EPISODE_LENGTH,
    'GRADIENT_STEPS_PER_GENERATION': Config.GRADIENT_STEPS_PER_GENERATION,
}

# Test configuration
Config.NUM_GENERATIONS = 2
Config.POPULATION_SIZE = 4
Config.NUM_PARENTS = 2
Config.NUM_OFFSPRING = 1
Config.NUM_MUTANTS = 1
Config.TRADING_PERIOD_DAYS = 50  # Short for testing (instead of 125)
Config.SETTLEMENT_PERIOD_DAYS = 30  # Updated to match new MAX_HOLDING_PERIOD of 30
Config.EPISODE_LENGTH = 50  # For backward compatibility
Config.GRADIENT_STEPS_PER_GENERATION = 5  # Few gradient steps

print("="*60)
print("Quick Training Test (2 generations, 4 agents)")
print("="*60)

# Load data
print("\nLoading data...")
loader = StockDataLoader()
data_array, stats = loader.load_and_prepare()

# Create trainer
print("\nCreating trainer...")
trainer = ERLTrainer(loader)

# Run training
print("\nStarting training...")
trainer.train()

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
print(f"Best fitness: {trainer.best_fitness:.2f}")
print(f"Buffer size: {len(trainer.replay_buffer)}")

# Restore config
for key, value in original_config.items():
    setattr(Config, key, value)

print("\n✓ Quick test passed! Ready for full training.")