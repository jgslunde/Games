"""
Training script for Brandubh (7x7 Tafl) using AlphaZero.

This script configures and runs the training pipeline specifically for Brandubh.
For other board sizes or game variants, create a similar training script.
"""

import argparse
import multiprocessing as mp

# Import the generic training module
from train import TrainingConfig, train

# Import Brandubh-specific classes
from brandubh import Brandubh
from network import BrandubhNet, MoveEncoder
from agent import BrandubhAgent


# Custom type for temperature_threshold argument
def temperature_threshold_type(value):
    """Parse temperature threshold as either int or 'king' string."""
    if value.lower() == "king":
        return "king"
    try:
        return int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"temperature-threshold must be an integer or 'king', got: {value}"
        )


# =============================================================================
# DEFAULT COMMAND-LINE ARGUMENTS
# Modify these values to change defaults without using command-line arguments
# =============================================================================

DEFAULT_ITERATIONS = 1000
DEFAULT_GAMES = 512
DEFAULT_SIMS_ATTACKER_SELFPLAY = 300
DEFAULT_SIMS_DEFENDER_SELFPLAY = 300
DEFAULT_SIMS_ATTACKER_EVAL = 300
DEFAULT_SIMS_DEFENDER_EVAL = 300
DEFAULT_BATCH_SIZE = 256
DEFAULT_LEARNING_RATE = 1e-3*(DEFAULT_BATCH_SIZE/256)
DEFAULT_EPOCHS = 10
DEFAULT_BATCHES_PER_EPOCH = 100
DEFAULT_EVAL_VS_RANDOM = 64
DEFAULT_NUM_WORKERS = mp.cpu_count()  # Use all available CPU cores
DEFAULT_DEVICE = None  # None = auto-detect (cuda if available, else cpu)
DEFAULT_RESUME = None  # Path to checkpoint file, or None to start fresh

# Temperature parameters
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TEMPERATURE_THRESHOLD = "king"

# Network architecture
DEFAULT_RES_BLOCKS = 4
DEFAULT_CHANNELS = 64

# Replay buffer
DEFAULT_REPLAY_BUFFER_SIZE = 10_000_000
DEFAULT_MIN_BUFFER_SIZE = 10*DEFAULT_BATCH_SIZE
DEFAULT_USE_DATA_AUGMENTATION = True  # Enable symmetry-based data augmentation

# Learning rate decay and regularization
DEFAULT_LR_DECAY = 0.99
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_VALUE_LOSS_WEIGHT = 20.0

# Dynamic loss boosting
DEFAULT_USE_DYNAMIC_BOOSTING = True
DEFAULT_DYNAMIC_BOOST_ALPHA = 0.1
DEFAULT_DYNAMIC_BOOST_MIN = 0.2
DEFAULT_DYNAMIC_BOOST_MAX = 5.0
DEFAULT_ATTACKER_WIN_LOSS_BOOST = 1.0  # Static boost (only used if dynamic disabled)

DEFAULT_DRAW_PENALTY_ATTACKER = +0.5  # Draw counts as attacker win, but discouraged.
DEFAULT_DRAW_PENALTY_DEFENDER = -0.9  # Draw = loss for defender, but slightly encouraged.

# MCTS exploration
DEFAULT_C_PUCT = 1.4

# Game rules
DEFAULT_KING_CAPTURE_PIECES = 2  # 2, 3, or 4 pieces needed to capture king
DEFAULT_KING_CAN_CAPTURE = True  # Whether king participates in captures
DEFAULT_THRONE_IS_HOSTILE = False  # Whether throne acts as hostile square
DEFAULT_THRONE_ENABLED = True  # Whether throne exists and blocks movement

# Evaluation
DEFAULT_EVAL_GAMES = 128
DEFAULT_EVAL_WIN_RATE = 0.52
DEFAULT_EVAL_FREQUENCY = 4
DEFAULT_EVAL_VS_RANDOM_FREQUENCY = 2

# Checkpointing
DEFAULT_SAVE_FREQUENCY = 1
DEFAULT_CHECKPOINT_DIR = "checkpoints"


# =============================================================================
# COMMAND-LINE ARGUMENT PARSER
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Brandubh AlphaZero")
    
    # Core training parameters
    parser.add_argument("--iterations", type=int, default=DEFAULT_ITERATIONS,
                       help=f"Number of training iterations (default: {DEFAULT_ITERATIONS})")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES,
                       help=f"Self-play games per iteration (default: {DEFAULT_GAMES})")
    parser.add_argument("--sims-attacker-selfplay", type=int, default=DEFAULT_SIMS_ATTACKER_SELFPLAY,
                       help=f"MCTS simulations for attacker in self-play (default: {DEFAULT_SIMS_ATTACKER_SELFPLAY})")
    parser.add_argument("--sims-defender-selfplay", type=int, default=DEFAULT_SIMS_DEFENDER_SELFPLAY,
                       help=f"MCTS simulations for defender in self-play (default: {DEFAULT_SIMS_DEFENDER_SELFPLAY})")
    parser.add_argument("--sims-attacker-eval", type=int, default=DEFAULT_SIMS_ATTACKER_EVAL,
                       help=f"MCTS simulations for attacker in evaluation (default: {DEFAULT_SIMS_ATTACKER_EVAL})")
    parser.add_argument("--sims-defender-eval", type=int, default=DEFAULT_SIMS_DEFENDER_EVAL,
                       help=f"MCTS simulations for defender in evaluation (default: {DEFAULT_SIMS_DEFENDER_EVAL})")
    
    # Neural network training
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                       help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--lr", type=float, default=DEFAULT_LEARNING_RATE,
                       help=f"Learning rate (default: {DEFAULT_LEARNING_RATE})")
    parser.add_argument("--lr-decay", type=float, default=DEFAULT_LR_DECAY,
                       help=f"Learning rate decay per iteration (default: {DEFAULT_LR_DECAY})")
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY,
                       help=f"L2 regularization weight decay (default: {DEFAULT_WEIGHT_DECAY})")
    parser.add_argument("--value-loss-weight", type=float, default=DEFAULT_VALUE_LOSS_WEIGHT,
                       help=f"Weight for value loss relative to policy loss (default: {DEFAULT_VALUE_LOSS_WEIGHT})")
    
    # Dynamic boosting arguments
    parser.add_argument("--use-dynamic-boosting", action="store_true", default=DEFAULT_USE_DYNAMIC_BOOSTING,
                       help=f"Use dynamic loss boosting based on win rates (default: {DEFAULT_USE_DYNAMIC_BOOSTING})")
    parser.add_argument("--no-dynamic-boosting", action="store_false", dest="use_dynamic_boosting",
                       help="Disable dynamic boosting (use static boost)")
    parser.add_argument("--dynamic-boost-alpha", type=float, default=DEFAULT_DYNAMIC_BOOST_ALPHA,
                       help=f"Smoothing factor for win rate tracking (default: {DEFAULT_DYNAMIC_BOOST_ALPHA})")
    parser.add_argument("--dynamic-boost-min", type=float, default=DEFAULT_DYNAMIC_BOOST_MIN,
                       help=f"Minimum boost factor (default: {DEFAULT_DYNAMIC_BOOST_MIN})")
    parser.add_argument("--dynamic-boost-max", type=float, default=DEFAULT_DYNAMIC_BOOST_MAX,
                       help=f"Maximum boost factor (default: {DEFAULT_DYNAMIC_BOOST_MAX})")
    parser.add_argument("--attacker-win-loss-boost", type=float, default=DEFAULT_ATTACKER_WIN_LOSS_BOOST,
                       help=f"Static boost for attacker wins (only if dynamic disabled, default: {DEFAULT_ATTACKER_WIN_LOSS_BOOST})")
    
    parser.add_argument("--draw-penalty-attacker", type=float, default=DEFAULT_DRAW_PENALTY_ATTACKER,
                       help=f"Value penalty for attacker draws (default: {DEFAULT_DRAW_PENALTY_ATTACKER})")
    parser.add_argument("--draw-penalty-defender", type=float, default=DEFAULT_DRAW_PENALTY_DEFENDER,
                       help=f"Value penalty for defender draws (default: {DEFAULT_DRAW_PENALTY_DEFENDER})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                       help=f"Training epochs per iteration (default: {DEFAULT_EPOCHS})")
    parser.add_argument("--batches-per-epoch", type=int, default=DEFAULT_BATCHES_PER_EPOCH,
                       help=f"Number of batches sampled per epoch (default: {DEFAULT_BATCHES_PER_EPOCH})")
    
    # Network architecture
    parser.add_argument("--res-blocks", type=int, default=DEFAULT_RES_BLOCKS,
                       help=f"Number of residual blocks (default: {DEFAULT_RES_BLOCKS})")
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS,
                       help=f"Number of channels in conv layers (default: {DEFAULT_CHANNELS})")
    
    # MCTS parameters
    parser.add_argument("--c-puct", type=float, default=DEFAULT_C_PUCT,
                       help=f"MCTS exploration constant (default: {DEFAULT_C_PUCT})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                       help=f"Sampling temperature for move selection (default: {DEFAULT_TEMPERATURE})")
    parser.add_argument("--temperature-threshold", type=temperature_threshold_type, default=DEFAULT_TEMPERATURE_THRESHOLD,
                       help=f"Move number after which temperature=0, or 'king' to drop when king leaves throne (default: {DEFAULT_TEMPERATURE_THRESHOLD})")
    
    # Game rules
    parser.add_argument("--king-capture-pieces", type=int, default=DEFAULT_KING_CAPTURE_PIECES, choices=[2, 3, 4],
                       help=f"Number of pieces needed to capture king: 2 (standard), 3 (3/4 sides), 4 (all sides) (default: {DEFAULT_KING_CAPTURE_PIECES})")
    parser.add_argument("--king-can-capture", action="store_true", default=DEFAULT_KING_CAN_CAPTURE,
                       help=f"King can participate in captures (default: {DEFAULT_KING_CAN_CAPTURE})")
    parser.add_argument("--king-cannot-capture", action="store_false", dest="king_can_capture",
                       help="King cannot participate in captures")
    parser.add_argument("--throne-is-hostile", action="store_true", default=DEFAULT_THRONE_IS_HOSTILE,
                       help=f"Throne acts as hostile square for captures (default: {DEFAULT_THRONE_IS_HOSTILE})")
    parser.add_argument("--throne-not-hostile", action="store_false", dest="throne_is_hostile",
                       help="Throne does not act as hostile square")
    parser.add_argument("--throne-enabled", action="store_true", default=DEFAULT_THRONE_ENABLED,
                       help=f"Throne exists and blocks non-king movement (default: {DEFAULT_THRONE_ENABLED})")
    parser.add_argument("--throne-disabled", action="store_false", dest="throne_enabled",
                       help="Throne disabled - center square acts as normal square")
    
    # Replay buffer
    parser.add_argument("--replay-buffer-size", type=int, default=DEFAULT_REPLAY_BUFFER_SIZE,
                       help=f"Maximum replay buffer size (default: {DEFAULT_REPLAY_BUFFER_SIZE})")
    parser.add_argument("--min-buffer-size", type=int, default=DEFAULT_MIN_BUFFER_SIZE,
                       help=f"Minimum buffer size before training (default: {DEFAULT_MIN_BUFFER_SIZE})")
    parser.add_argument("--use-data-augmentation", action="store_true", default=DEFAULT_USE_DATA_AUGMENTATION,
                       help=f"Enable symmetry-based data augmentation (default: {DEFAULT_USE_DATA_AUGMENTATION})")
    parser.add_argument("--no-data-augmentation", action="store_false", dest="use_data_augmentation",
                       help="Disable data augmentation")
    
    # Evaluation
    parser.add_argument("--eval-games", type=int, default=DEFAULT_EVAL_GAMES,
                       help=f"Games for network evaluation (default: {DEFAULT_EVAL_GAMES})")
    parser.add_argument("--eval-win-rate", type=float, default=DEFAULT_EVAL_WIN_RATE,
                       help=f"Win rate threshold to replace best model (default: {DEFAULT_EVAL_WIN_RATE})")
    parser.add_argument("--eval-frequency", type=int, default=DEFAULT_EVAL_FREQUENCY,
                       help=f"Evaluate every N iterations (default: {DEFAULT_EVAL_FREQUENCY})")
    parser.add_argument("--eval-vs-random", type=int, default=DEFAULT_EVAL_VS_RANDOM,
                       help=f"Games vs random per color (default: {DEFAULT_EVAL_VS_RANDOM})")
    parser.add_argument("--eval-vs-random-frequency", type=int, default=DEFAULT_EVAL_VS_RANDOM_FREQUENCY,
                       help=f"Evaluate vs random every N iterations (default: {DEFAULT_EVAL_VS_RANDOM_FREQUENCY})")
    
    # System
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
                       help=f"Number of parallel workers (default: {DEFAULT_NUM_WORKERS} CPUs)")
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE,
                       help="Device (cuda/cpu, default: auto-detect)")
    
    # Checkpointing
    parser.add_argument("--save-frequency", type=int, default=DEFAULT_SAVE_FREQUENCY,
                       help=f"Save checkpoint every N iterations (default: {DEFAULT_SAVE_FREQUENCY})")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_CHECKPOINT_DIR,
                       help=f"Directory for saving checkpoints (default: {DEFAULT_CHECKPOINT_DIR})")
    parser.add_argument("--resume", type=str, default=DEFAULT_RESUME,
                       help="Resume from checkpoint file")
    
    args = parser.parse_args()
    
    # Create config
    config = TrainingConfig()
    
    # Set game-specific configuration
    config.game_class = Brandubh
    config.network_class = BrandubhNet
    config.agent_class = BrandubhAgent
    config.move_encoder_class = MoveEncoder
    config.board_size = 7  # Brandubh is 7x7
    
    # Core training parameters
    config.num_iterations = args.iterations
    config.num_games_per_iteration = args.games
    config.num_mcts_sims_attacker = args.sims_attacker_selfplay
    config.num_mcts_sims_defender = args.sims_defender_selfplay
    config.eval_mcts_sims_attacker = args.sims_attacker_eval
    config.eval_mcts_sims_defender = args.sims_defender_eval
    
    # Neural network training
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.lr_decay = args.lr_decay
    config.weight_decay = args.weight_decay
    config.value_loss_weight = args.value_loss_weight
    
    # Dynamic boosting
    config.use_dynamic_boosting = args.use_dynamic_boosting
    config.dynamic_boost_alpha = args.dynamic_boost_alpha
    config.dynamic_boost_min = args.dynamic_boost_min
    config.dynamic_boost_max = args.dynamic_boost_max
    config.attacker_win_loss_boost = args.attacker_win_loss_boost
    
    config.draw_penalty_attacker = args.draw_penalty_attacker
    config.draw_penalty_defender = args.draw_penalty_defender
    config.num_epochs = args.epochs
    config.batches_per_epoch = args.batches_per_epoch
    
    # Network architecture
    config.num_res_blocks = args.res_blocks
    config.num_channels = args.channels
    
    # MCTS parameters
    config.c_puct = args.c_puct
    config.temperature = args.temperature
    config.temperature_threshold = args.temperature_threshold
    
    # Game rules
    config.king_capture_pieces = args.king_capture_pieces
    config.king_can_capture = args.king_can_capture
    config.throne_is_hostile = args.throne_is_hostile
    config.throne_enabled = args.throne_enabled
    
    # Replay buffer
    config.replay_buffer_size = args.replay_buffer_size
    config.min_buffer_size = args.min_buffer_size
    config.use_data_augmentation = args.use_data_augmentation
    
    # Evaluation
    config.eval_games = args.eval_games
    config.eval_win_rate = args.eval_win_rate
    config.eval_frequency = args.eval_frequency
    config.eval_vs_random_games = args.eval_vs_random
    config.eval_vs_random_frequency = args.eval_vs_random_frequency
    
    # System
    config.num_workers = args.num_workers
    if args.device is not None:
        config.device = args.device
    
    # Checkpointing
    config.save_frequency = args.save_frequency
    config.checkpoint_dir = args.checkpoint_dir
    
    # Run training
    train(config, resume_from=args.resume)
