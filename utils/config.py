"""
Configuration file for Project Eigen 2
All hyperparameters and settings in one place
"""

import torch
from pathlib import Path

class Config:
    # ============ Data Parameters ============
    DATA_PATH = Path(__file__).parent.parent / "Eigen2_Master_PY_OUTPUT.pkl"
    DATE_COLUMN = 0  # Column A (0-indexed)
    INVESTABLE_START_COL = 9  # Column J (0-indexed) - First investable stock: DFAC
    INVESTABLE_END_COL = 116  # Column DO (0-indexed, inclusive) - Last investable stock: VXX
    NUM_INVESTABLE_STOCKS = 108
    TOTAL_COLUMNS = 117  # Skinny dataset: only loading first 117 columns (columns 0-116) from the pkl
    FEATURES_PER_CELL = 5  # [close, RSI, MACD_signal, TRIX, diff20DMA] - selected from original 9
    
    CONTEXT_WINDOW_DAYS = 504  # 2 years of trading days
    TRAIN_TEST_SPLIT = 0.95  # 95% train, 5% validation (6 months) - DEPRECATED: now using HOLDOUT_DAYS
    INTERIM_VALIDATION_DAYS = 252  # Interim validation set for walk-forward validation during training
    HOLDOUT_DAYS = 251  # Last 251 rows reserved for final "highlander round" (completely secret)
    
    # ============ Action Space Parameters ============
    NUM_ACTIONS = NUM_INVESTABLE_STOCKS  # 108 stocks
    ACTION_DIM = 2  # [coefficient, sale_target]
    MIN_COEFFICIENT = 1.0  # Minimum position size (or 0 for no action)
    MIN_SALE_TARGET = 10.0  # Minimum 10% gain target
    MAX_SALE_TARGET = 50.0  # Maximum 50% gain target
    COEFFICIENT_THRESHOLD = 0.5  # Threshold for deciding action vs no-action
    
    # ============ Environment Parameters ============
    # Holding period structure: agent must hold for MIN_HOLDING_PERIOD,
    # then has LIQUIDATION_WINDOW days to exit, forced liquidation at MAX_HOLDING_PERIOD
    MIN_HOLDING_PERIOD = 20  # Minimum holding period (cannot sell before this)
    LIQUIDATION_WINDOW = 10  # Days available to liquidate after min hold (days 21-30)
    MAX_HOLDING_PERIOD = MIN_HOLDING_PERIOD + LIQUIDATION_WINDOW  # 30 total days

    LOSS_PENALTY_MULTIPLIER = 1.0  # Losses treated equally to gains (was 3.0, removed penalty to fix zombie agents)
    INACTION_PENALTY = 20.0  # Penalty per day without an open position (quadrupled from original 5.0)
    FORCED_EXIT_PENALTY = 10.0  # Penalty for forced exits due to max_holding_period (equal to original inaction penalty)
    ZERO_TRADES_PENALTY = 10000.0  # Heavy penalty for making NO trades at all

    TRADING_PERIOD_DAYS = 125  # 6 months - period where model can open new positions
    SETTLEMENT_PERIOD_DAYS = 30  # Additional days to close remaining positions (must be >= MAX_HOLDING_PERIOD)
    EPISODE_LENGTH = TRADING_PERIOD_DAYS  # For backward compatibility
    
    # ============ Model Architecture ============
    # Feature extraction
    CNN_FILTERS = 32
    CNN_KERNEL_SIZE = 3
    
    # Temporal processing
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    LSTM_BIDIRECTIONAL = True
    
    # Attention
    USE_ATTENTION = True
    ATTENTION_HEADS = 8
    
    # Actor network
    ACTOR_HIDDEN_DIMS = [256, 128, 64]
    
    # Critic network
    CRITIC_HIDDEN_DIMS = [256, 128]
    
    # ============ DDPG Parameters ============
    GAMMA = 0.99  # Discount factor
    TAU = 0.005  # Soft update parameter
    ACTOR_LR = 1e-4
    CRITIC_LR = 3e-4
    WEIGHT_DECAY = 5e-4
    
    # Replay buffer
    BUFFER_SIZE = 135000  # Maximum buffer size
    BATCH_SIZE = 16
    MIN_BUFFER_SIZE = 9000  # Start training after this many transitions
    MIN_BUFFER_SIZE_SWEEP = 5000  # Lower threshold for sweeps (10 gens, faster DDPG)
    
    # Exploration noise
    NOISE_SCALE = 0.125  # Increased from 0.1 to boost exploration (positive correlation with fitness)
    NOISE_DECAY = 0.99995  # Slowed decay to maintain exploration longer (was 0.9999)
    MIN_NOISE = 0.01
    
    # ============ ERL Parameters ============
    POPULATION_SIZE = 31
    NUM_GENERATIONS = 100
    EPISODE_LENGTH = 125  # 6 months trading period (kept for compatibility, use TRADING_PERIOD_DAYS)
    
    # Selection
    """     NUM_PARENTS = 8  # Top performers to keep
    NUM_OFFSPRING = 6  # Generated via crossover
    NUM_MUTANTS = 2  # Random mutations """
    ELITE_FRAC = 0.125      # 12.5% of population (reduced from 0.375 for higher diversity)
    OFFSPRING_FRAC = 0.375   # 37.5% of population
    # MUTANT_FRAC will be the remainder (50%, massively increased for exploration)
    
    # Genetic operators
    CROSSOVER_ALPHA_MIN = 0.2  # Widened range for more diverse offspring (was 0.3)
    CROSSOVER_ALPHA_MAX = 0.8  # Widened range for more diverse offspring (was 0.7)
    MUTATION_RATE = 0.40  # Base mutation rate (doubled from 0.20 for aggressive exploration)
    MUTATION_STD = 0.05  # Base mutation magnitude (doubled from 0.025 for aggressive exploration)
    # NOTE: Adaptive mutation automatically boosts these values by 1.5x when validation fitness
    # plateaus for 3 consecutive generations (< 2% improvement), helping escape local optima
    # Max caps are set in ERLTrainer (0.8 for rate, 0.1 for std) to allow further increases
    
    # Training
    GRADIENT_STEPS_PER_GENERATION = 32
    GRADIENT_ACCUMULATION_STEPS = 4
    
    # ============ Training Parameters ============
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 4  # For data loading
    # NOTE: Random seed is now set dynamically per wandb run in ERLTrainer
    # This ensures parallel runs have unique, independent behavior
    
    # Checkpointing
    CHECKPOINT_DIR = Path("checkpoints")
    SAVE_FREQUENCY = 1  # Save every N generations (IMPORTANT: Set to 1 for safety!)
    # Replay buffer is saved every 5 generations and on first fill (if no buffer exists on cloud)
    
    # Logging
    LOG_DIR = Path("logs")
    LOG_FREQUENCY = 1  # Log every N generations
    
    # ============ Validation Parameters ============
    EVAL_EPISODES = 5  # Number of episodes for evaluation
    
    @classmethod
    def display(cls):
        """Print all configuration parameters"""
        print("=" * 60)
        print("Project Eigen 2 Configuration")
        print("=" * 60)
        for key, value in cls.__dict__.items():
            if not key.startswith('_') and not callable(value) and key != 'display':
                print(f"{key:.<40} {value}")
        print("=" * 60)
    
    @classmethod
    def validate(cls):
        """Validate configuration settings"""
        errors = []
        
        # Check data path exists
        if not Path(cls.DATA_PATH).exists():
            errors.append(f"Data file not found: {cls.DATA_PATH}")
        
        # Check directories exist
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check GPU availability
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. Training will be slow on CPU.")
        
        if errors:
            print("\n".join(errors))
            return False
        return True


if __name__ == "__main__":
    Config.display()