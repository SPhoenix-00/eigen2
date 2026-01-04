"""
Configuration file for Project Eigen 2
All hyperparameters and settings in one place
"""

import torch
from pathlib import Path

# Import the output filename from process_eigen_data.py
from process_eigen_data import OUTPUT_FILE_PKL

class Config:
    # ============ Data Parameters ============
    DATA_PATH = Path(__file__).parent.parent / OUTPUT_FILE_PKL
    DATE_COLUMN = 0  # Column A (0-indexed)
    INVESTABLE_START_COL = 9  # Column J (0-indexed) - First investable stock: DFAC
    INVESTABLE_END_COL = 116  # Column DO (0-indexed, inclusive) - Last investable stock: VXX
    NUM_INVESTABLE_STOCKS = 108
    TOTAL_COLUMNS = 117  # Skinny dataset: only loading first 117 columns (columns 0-116) from the pkl
    FEATURES_PER_CELL = 5  # [close, RSI, MACD_signal, TRIX, diff20DMA] - selected from original 9
    
    CONTEXT_WINDOW_DAYS = 151  # ~6 months of trading days
    TRAIN_TEST_SPLIT = 0.95  # DEPRECATED - Use VALIDATION_DAYS and COMMITTEE_HOLDOUT_DAYS instead

    # ============ Data Split Configuration ============
    # STRICT SEPARATION: Training → Validation → Holdout (no overlap)
    #
    # Example for 5000-day dataset:
    #   Training:   Days 0-4244    (4245 days) ← Used for training episodes
    #   Validation: Days 4245-4747 (503 days)  ← Used for walk-forward validation during training
    #   Holdout:    Days 4748-4999 (252 days)  ← ONLY for committee testing (NEVER seen by agents)
    #
    VALIDATION_DAYS = 503  # Days reserved for walk-forward validation during training
    COMMITTEE_HOLDOUT_DAYS = 252  # Days reserved EXCLUSIVELY for committee validation

    # Total reserved days: VALIDATION_DAYS + COMMITTEE_HOLDOUT_DAYS = 755 days
    # Training will use: (total_days - 755 - MIN_HOLDING_PERIOD) days
    
    # ============ Action Space Parameters ============
    NUM_ACTIONS = NUM_INVESTABLE_STOCKS  # 108 stocks
    ACTION_DIM = 2  # [coefficient, sale_target]
    MIN_SALE_TARGET = 10.0  # Minimum 10% gain target
    MAX_SALE_TARGET = 50.0  # Maximum 50% gain target
    COEFFICIENT_THRESHOLD = 1.0  # Threshold for opening position (stock must score >= this)
    
    # ============ Environment Parameters ============
    # Holding period structure: agent must hold for MIN_HOLDING_PERIOD,
    # then has LIQUIDATION_WINDOW days to exit, forced liquidation at MAX_HOLDING_PERIOD
    MIN_HOLDING_PERIOD = 20  # Minimum holding period (cannot sell before this)
    LIQUIDATION_WINDOW = 10  # Days available to liquidate after min hold (days 21-30)
    MAX_HOLDING_PERIOD = MIN_HOLDING_PERIOD + LIQUIDATION_WINDOW  # 30 total days

    LOSS_PENALTY_MULTIPLIER = 1.0  # DEPRECATED: Use CONSISTENCY_LOSS_MULTIPLIER instead. Kept for backward compatibility with sweep configs.
    INACTION_PENALTY = 0.0  # Penalty per day without an open position (reduced from 20.0 to smooth landscape)
    FORCED_EXIT_PENALTY_PCT = 0.01  # 3% penalty on position size (entry_price * coefficient)
    ZERO_TRADES_PENALTY_NORMAL = 2000.0  # Heavy penalty for making NO trades in normal mode
    ZERO_TRADES_PENALTY_CONSISTENCY = 500.0  # Penalty for making NO trades in consistency mode
    ZERO_TRADES_PENALTY_GAUNTLET = 10.0  # Soft penalty during stabilization/gauntlet (tactical no-trade is acceptable)
    HURDLE_RATE = 0.006  # 0.6% transaction cost per trade (mimics real trading costs, disincentivizes high-volume strategies)
    CONVICTION_SCALING_POWER = 1.25  # Power law exponent for conviction scaling (convex reward surface encourages high-confidence bets)

    # Win rate bonus - rewards consistent winning
    WIN_RATE_BONUS_THRESHOLD = 75.0  # Win rate % above which bonus kicks in
    WIN_RATE_BONUS_MIN_TRADES = 15   # Minimum trades required for bonus to apply
    # Bonus formula: (win_rate% - threshold)^2, e.g., 76% = 1pt, 100% = 625pt

    # ROI-based scoring adjustment
    ROI_ADJUSTMENT_MULTIPLIER = 50.0  # Multiplier for ROI adjustment in fitness scoring
    # Formula: Score = Fitness + (|Fitness| × multiplier × (AgentROI − MedianROI) / 100)
    ROI_CONFIDENCE_MIN_TRADES = 30  # Normal mode: minimum quality trades for full ROI bonus credit
    ROI_CONFIDENCE_MIN_TRADES_CONSISTENCY = 50  # Consistency mode: higher bar for stricter requirements
    # Confidence factor = min(1.0, quality_count / threshold)
    # This prevents "lucky snipers" who make few high-ROI trades from getting inflated fitness
    ROI_QUALITY_THRESHOLD = 10.0  # Default minimum gain_pct for a trade to count as "quality"
    ROI_USE_HOF_MEDIAN_AS_THRESHOLD = False  # If True, use HoF median ROI as threshold (supersedes default)

    # Hall of Fame erosion mechanism
    HOF_EROSION_ALPHA = 0.33  # EMA smoothing factor for gradual score erosion (0-1)
    # Higher α = faster erosion. Default 0.33 means scores converge to new median over ~3 generations
    # This prevents early agents from having unfair ROI advantages as the median rises

    TRADING_PERIOD_DAYS = 125  # 6 months - period where model can open new positions
    SETTLEMENT_PERIOD_DAYS = 30  # Additional days to close remaining positions (must be >= MAX_HOLDING_PERIOD)
    EPISODE_LENGTH = TRADING_PERIOD_DAYS  # For backward compatibility

    # Observation noise for regularization (prevents overfitting during training)
    OBSERVATION_NOISE_STD = 0.01  # Standard deviation of Gaussian noise added to observations during training
    
    # ============ Model Architecture ============
    # Feature extraction
    CNN_FILTERS = 32
    CNN_KERNEL_SIZE = 3
    
    # Temporal processing
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    LSTM_BIDIRECTIONAL = True
    
    # Attention
    # CRITICAL ROCm FIX: Disable attention for ROCm due to memory access faults
    # ROCm's scaled_dot_product_attention has bugs that cause crashes
    # This is automatically set based on GPU backend detection
    USE_ATTENTION = True
    ATTENTION_HEADS = 8
    
    @classmethod
    def _disable_attention_for_rocm(cls):
        """Disable attention mechanism for ROCm backend to avoid memory access faults."""
        try:
            from utils.device import get_gpu_backend
            gpu_backend = get_gpu_backend()
            if gpu_backend == "ROCm":
                cls.USE_ATTENTION = False
                print(f"[Config] Attention disabled for ROCm backend (memory access fault workaround)")
        except Exception:
            pass  # If detection fails, keep default (attention enabled)
    
    @classmethod
    def initialize(cls):
        """Initialize configuration, including ROCm-specific settings."""
        cls._disable_attention_for_rocm()
    
    # Actor network
    ACTOR_HIDDEN_DIMS = [256, 128, 64]
    
    # Critic network
    CRITIC_HIDDEN_DIMS = [256, 128]

    # Dropout rates for regularization
    DROPOUT_RATE = 0.2  # Standard dropout for dense layers
    DROPOUT_RATE_HEADS = 0.2  # Dropout for action heads
    DROPOUT_2D_RATE = 0.1  # Dropout2d for convolutional layers

    # ============ DDPG Parameters ============
    GAMMA = 0.99  # Discount factor
    TAU = 0.005  # Soft update parameter
    ACTOR_LR = 1e-4
    CRITIC_LR = 3e-4
    WEIGHT_DECAY = 5e-4
    
    # Replay buffer
    BUFFER_SIZE = 1500000  # Maximum buffer size
    BATCH_SIZE = 160
    LOCAL_BATCH_SIZE = 64  # Batch size for local mode training
    LOCAL_GRADIENT_ACCUMULATION_STEPS = 1  # No accumulation in local mode - each step = 1 disk read (vs 16x with accumulation)
    LOCAL_NUM_DATALOADER_WORKERS = 0  # Run in main process - Windows/WSL multiprocessing overhead is massive for 64-item batches
    LOCAL_POPULATION_SIZE = 32  # Smaller population for local mode (vs 96 for distributed)
    LOCAL_GRADIENT_STEPS_PER_GENERATION = 8  # Fewer gradient steps for local mode (vs 32 for distributed)
    LOCAL_TRAINING_AGENT_BATCH_SIZE = 8  # Train 8 agents at a time on GPU (4 batches of 8 = 32 agents)
    MIN_BUFFER_SIZE = 23200  # Start training after this many transitions
    MIN_BUFFER_SIZE_SWEEP = 5000  # Lower threshold for sweeps (10 gens, faster DDPG)
    
    # Exploration noise
    NOISE_SCALE = 0.125  # Increased from 0.1 to boost exploration (positive correlation with fitness)
    NOISE_DECAY = 0.99995  # Slowed decay to maintain exploration longer (was 0.9999)
    MIN_NOISE = 0.01
    
    # ============ ERL Parameters ============
    POPULATION_SIZE = 96
    NUM_GENERATIONS = 100
    EPISODE_LENGTH = 125  # 6 months trading period (kept for compatibility, use TRADING_PERIOD_DAYS)
    
    # Selection (fraction-based)
    ELITE_FRAC = 0.4       # 40% of population
    OFFSPRING_FRAC = 0.4   # 40% of population
    # MUTANT_FRAC is the remainder (20%)

    # Heroes mode - when loading pre-trained agents from Hall of Fame
    # Uses higher elite fraction since these agents are already well-trained
    HEROES_ELITE_FRAC = 0.50    # 50% elites (16 agents) - preserve more proven performers
    HEROES_OFFSPRING_FRAC = 0.375  # 37.5% offspring (12 agents) - blend elite genetics
    # HEROES_MUTANT_FRAC = 0.125 (remainder: 4 agents) - minimal random exploration

    # Single-agent mode - focused refinement of one Global50 agent
    # Population is initialized from clones of the single agent with varying mutation levels
    SINGLE_CLONE_FRAC = 0.50            # 50% pure clones (no mutation)
    SINGLE_NORMAL_MUTATION_FRAC = 0.25  # 25% with normal mutation rate
    SINGLE_PLATEAU_MUTATION_FRAC = 0.25 # 25% with plateau (1.5x) mutation rate
    # Note: These must sum to 1.0

    SINGLE_STABILIZATION_GENERATIONS = 5  # Generations before breakthrough detection starts
    SINGLE_TARGET_BREAKTHROUGHS = 4       # Training ends after 4 confirmed breakthroughs

    # Multi-agent mode - sequential refinement of all 9 committee members
    # Trains one committee member at a time using standard POPULATION_SIZE
    # Rotates to next member after each breakthrough (5% improvement)
    MULTI_BREAKTHROUGH_THRESHOLD = 0.05  # 5% improvement over member's baseline score
    MULTI_TARGET_TURNOVERS = 3           # End after 3 complete turnovers (all 9 members get 3 breakthroughs each)

    # Maverick mode - aggressive training mode with FOMO/ROI-First reward functions
    # Used to produce aggressive signal generators that break committee inaction
    MAVERICK_MODE = False  # Set via --maverick flag
    MAVERICK_CAP = 5       # Maximum Mavericks allowed in Global 50 (The "Highlander" Rule)
    MAVERICK_TARGET_RANK = 20  # Training stops when Maverick reaches this rank or higher
    MAVERICK_MARKET_BENCHMARK_COL = 44  # Column for market benchmark (S&P 500 proxy) for FOMO calculation

    # Consistency mode - loss magnification for training on consistency
    # Applied ONLY when --consistency flag is used. Normal mode uses 1.0 (no magnification)
    CONSISTENCY_LOSS_MULTIPLIER = 1.5  # Magnify losses by 1.5x to focus training on reducing drawdowns

    # Genetic operators
    CROSSOVER_ALPHA_MIN = 0.2
    CROSSOVER_ALPHA_MAX = 0.8
    MUTATION_RATE = 0.20  # Base mutation rate for normal mode
    MUTATION_RATE_CONSISTENCY = 0.15  # Mutation rate for consistency mode (lower to preserve stable traits)
    MUTATION_STD = 0.025  # Base mutation magnitude
    # NOTE: Adaptive mutation automatically boosts these values by 1.5x when validation fitness
    # plateaus for 3 consecutive generations (< 2% improvement), helping escape local optima
    # Max caps are set in ERLTrainer (0.8 for rate, 0.1 for std) to allow further increases
    
    # Training
    GRADIENT_STEPS_PER_GENERATION = 32
    GRADIENT_STEPS_PER_GENERATION_STABILIZATION = 10  # Reduced steps during stabilization phase
    GRADIENT_ACCUMULATION_STEPS = 1
    
    # ============ Training Parameters ============
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    NUM_WORKERS = 6  # For data loading (deprecated, kept for compatibility)
    NUM_DATALOADER_WORKERS = 6  # Number of background workers for async batch loading
    # With 4 workers, batches are prepared in parallel while GPU trains
    # Higher = more CPU usage but better GPU utilization
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
    EVAL_NUM_WORKERS = 48  # Maximum parallel workers for evaluation (actual: min(cpu_count-1, this value))
    SKIP_REEVALUATION_ON_RESUME = True  # Skip re-evaluating agents on resume (only needed if reward function changed)

    # ============ Gauntlet Mode Parameters ============
    # Shift from "Run for N Generations" to "Achieve N Confirmed Breakthroughs"
    # This addresses the "Ghost Score" problem where lucky validation spikes
    # lock in unrealistic baselines that stall training progress
    #
    # FIRST BREAKTHROUGH: Automatically accepted (any gauntlet score establishes baseline)
    # SUBSEQUENT BREAKTHROUGHS: Must exceed confirmed baseline by threshold percentage

    GAUNTLET_MODE_ENABLED = True  # Enable Gauntlet Mode breakthrough validation

    # Efficiency Gating - prevents "volume swindling" where agents inflate scores via capital usage
    # Gauntlet scores are adjusted by ROI efficiency: Score_Final = f(raw_score, ROI / baseline)
    # - Positive scores scaled by (ROI / baseline): 10% ROI = neutral, 20% = 2x boost, 5% = 0.5x deflation
    # - Negative scores with positive ROI: rescued by dividing (ROI is king)
    # - Negative scores with negative ROI: amplified by multiplying
    EFFICIENCY_BASELINE_ROI = 8.0  # Baseline ROI % for neutral efficiency adjustment

    # Breakthrough detection
    BREAKTHROUGH_WARMUP_GENERATIONS = 3  # No breakthrough detection until Generation > this value (let population churn)
    BREAKTHROUGH_THRESHOLD_NORMAL = 0.10  # 10% improvement over baseline in normal mode
    BREAKTHROUGH_THRESHOLD_CONSISTENCY = 0.05  # 5% improvement in consistency mode (stricter)

    # Breakthrough quorum - number of agents that must breach threshold simultaneously
    BREAKTHROUGH_QUORUM_NORMAL = 1  # Require 1 agent to breach in normal mode
    BREAKTHROUGH_QUORUM_CONSISTENCY = 1  # Only test highest agent per generation in consistency mode

    # Asset Selection - Two-phase filtering to prevent wasting GPU time on weak candidates
    MAX_CANDIDATES_FOR_STRESS_TEST = 5  # Phase 1: Truncate queue to top K candidates after deduplication
    STRESS_TEST_ENABLED = True  # Phase 2: Run stress test using pessimistic fitness before stabilization

    # Stabilization phase - lock training on candidate for N generations
    STABILIZATION_GENERATIONS = 3  # Allow networks to converge on new behavior
    STABILIZATION_GHOST_DETECTION = False  # If True, abort stabilization if fitness drops below baseline (disabled to give candidates a fair chance)

    # Gauntlet validation - rigorous stress test with many diverse slices
    GAUNTLET_NUM_SLICES = 20  # Number of validation slices for Gauntlet (vs 7 for normal validation)

    # ============ Committee Selection Parameters ============
    # Committee drafts top agents from Global50 to form an ensemble
    COMMITTEE_SIZE = 9  # Target committee size
    COMMITTEE_TOP_K_INITIAL = 20  # Start optimization with top K agents by gauntlet score
    COMMITTEE_EARLY_STOP = 5  # Stop expanding pool after N consecutive non-improvements
    COMMITTEE_CORRELATION_EXPONENT = 2  # Exponent for correlation penalty: (1 - avg_corr^exp)
    COMMITTEE_VALIDATION_SLICES = 5  # Number of holdout slices for validation
    COMMITTEE_QUORUM = 3  # Number of committee members required to agree for a trade
    COMMITTEE_VETO_COUNT = 5  # Number of silent members needed to veto a trade
    COMMITTEE_VETO_THRESHOLD = 0.75  # Coefficient threshold below which a member is considered "silent" for veto purposes

    # Breakthrough goals - stopping condition based on confirmed breakthroughs
    TARGET_BREAKTHROUGHS_NORMAL = 4  # Number of confirmed breakthroughs in normal mode
    TARGET_BREAKTHROUGHS_CONSISTENCY = 8  # Deprecated - not used in consistency mode (uses turnovers instead)

    # Hall of Fame turnover goals (consistency mode only)
    TARGET_HOF_TURNOVERS = 3  # Number of complete HoF turnovers required (minimum 2)
    # Unified turnover logic: All 10 HoF agents must have ROI >= previous median
    # Initial median is 0 (first breakthrough establishes baseline from gauntlet score)
    # Each turnover raises the bar: median₀=0 → median₁ → median₂ → median₃...
    # This creates a ratcheting quality mechanism where HoF progressively improves

    # Fallback to generation limit if breakthroughs not achieved
    MAX_GENERATIONS_GAUNTLET = 100  # Maximum generations before stopping regardless of breakthroughs
    
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
            expected_filename = cls.DATA_PATH.name
            errors.append(f"Data file not found: {cls.DATA_PATH}")
            errors.append(f"  Expected filename: {expected_filename}")
            errors.append(f"  Please download the training data file to: {cls.DATA_PATH.parent}")
            errors.append(f"  Or update OUTPUT_FILE_PKL in process_eigen_data.py if using a different file")
        
        # Check cloud credentials if cloud provider is set
        import os
        cloud_provider = os.environ.get("CLOUD_PROVIDER", "").lower()
        if cloud_provider == "gcs":
            creds_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or os.environ.get("GCS_CREDENTIALS")
            if not creds_path:
                errors.append("CLOUD_PROVIDER=gcs is set but GOOGLE_APPLICATION_CREDENTIALS is not set")
            elif not Path(creds_path).exists():
                errors.append(f"GCS credentials file not found: {creds_path}")
            bucket_name = os.environ.get("CLOUD_BUCKET")
            if not bucket_name:
                errors.append("CLOUD_PROVIDER=gcs is set but CLOUD_BUCKET is not set")
        
        # Check directories exist
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Check GPU availability
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available. Training will be slow on CPU.")
        
        if errors:
            print("\n" + "="*60)
            print("CONFIGURATION VALIDATION ERRORS:")
            print("="*60)
            for error in errors:
                print(f"  ❌ {error}")
            print("="*60 + "\n")
            return False
        return True


# Initialize ROCm-specific settings when module is imported
# This must happen before networks are created
Config._disable_attention_for_rocm()

if __name__ == "__main__":
    Config.display()