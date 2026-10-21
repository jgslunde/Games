"""
Training configuration: the TrainingConfig object used by train.py, and its JSON form.

A run is described by one JSON file with these sections (every key is optional; missing keys take
the TrainingConfig defaults below, and the resolved config is written to the checkpoint directory
as run_config.json when training starts, so a run's settings can always be read back):

    {
      "name": "...",                       free text, shown in the log
      "game": {"ruleset": "copenhagen", "max_game_length": 300},
                                           a named rule set of tafl.RULESETS (it fixes the board,
                                           the starting position and every rule)
      "network": {...}                     a NetworkConfig (network_config.py), or the path of a
                                           JSON file holding one, relative to this file
      "self_play": {...}, "training": {...}, "replay_buffer": {...},
      "evaluation": {...}, "system": {...}
    }

Values can be overridden on the command line with "section.key=value", e.g.
"system.num_workers=16" or "network.features.history_length=4" (see apply_overrides).
"""

import copy
import json
import multiprocessing as mp
import os
from typing import Dict, List, Optional

from network_config import NetworkConfig
from tafl import RULESETS, Rules, policy_size


class TrainingConfig:
    """Configuration for training (the defaults are generic; the JSON configs set everything)."""

    name = ""                         # Free-text label of the run

    # Game
    ruleset = "brandubh"              # A rule set name of tafl.RULESETS
    max_game_length = 500             # Maximum plies before declaring a draw

    # Self-play
    num_iterations = 100              # Number of training iterations
    num_games_per_iteration = 100     # Self-play games per iteration
    num_mcts_sims_attacker = 100      # MCTS simulations for attacker in self-play
    num_mcts_sims_defender = 100      # MCTS simulations for defender in self-play
    c_puct = 1.4                      # MCTS exploration constant for self-play
    fpu_reduction = -0.5              # First Play Urgency: reduction relative to parent Q-value
    search_discount = 1.0             # Search depth discount per move during MCTS backup
    temperature = 1.0                 # Sampling temperature for moves
    temperature_mode = "king"         # "fixed": drop at threshold, "king": drop when king leaves, "decay": linear
    temperature_threshold = 15        # Move number after which temperature = 0 (for "fixed" mode)
    temperature_decay_moves = 30      # Number of moves for linear decay (for "decay" mode)
    add_dirichlet_noise = True        # Add Dirichlet noise to root for exploration
    dirichlet_alpha = 0.3             # Concentration parameter (lower = more concentrated)
    dirichlet_epsilon = 0.25          # Weight of noise (0.25 = 25% noise, 75% network priors)
    use_random_opening_moves = False  # In 50% of games, play 1-4 random moves (2-8 ply) before MCTS
    use_dynamic_sim_balancing = False # Move simulations to the side that scores too little
    dynamic_sim_adjustment_rate = 0.1 # Fraction of the distance to the band corrected per update
    dynamic_sim_min_win_rate = 0.2    # Keep each side's score >= this (band [x, 1-x]); 0.5 = push towards 50%
    reward_discount = 0.98            # Value-target discount per ply (1.0 = no discount)
    track_unique_states = False       # Report unique board states seen in self-play (adds overhead)

    # Network (a NetworkConfig; a fresh copy per instance, see __init__)
    network = NetworkConfig()

    # Training
    batch_size = 32                   # Training batch size
    num_epochs = 10                   # Epochs per iteration
    batches_per_epoch = 100           # Number of batches to sample per epoch
    learning_rate = 0.001             # Initial learning rate
    lr_decay = 0.95                   # Learning rate decay per iteration
    lr_floor = 1e-6                   # Minimum learning rate
    use_stepping_lr = False           # If True, use the stepping schedule instead of exponential decay
    lr_step_iterations = []           # Iterations at which to step LR (e.g., [50, 100, 150])
    lr_step_values = []               # LR values for each step (len(lr_step_iterations) + 1 of them)
    weight_decay = 1e-4               # L2 regularization
    value_loss_weight = 10.0          # Weight for value loss (policy loss weight is always 1.0)
    ending_loss_weight = 0.5          # Weight for the loss of the ending output (how the game ends)
    moves_left_loss_weight = 1.0      # Weight for the loss of the moves-left output
    use_dynamic_boosting = True       # Scale losses by the smoothed win rates of the two sides
    dynamic_boost_alpha = 0.1         # Smoothing factor for win rate tracking
    dynamic_boost_min = 0.5           # Minimum boost factor
    dynamic_boost_max = 3.0           # Maximum boost factor
    attacker_win_loss_boost = 1.0     # Static boost (only used if use_dynamic_boosting=False)
    draw_penalty_attacker = 0.0       # Draw value for attackers
    draw_penalty_defender = 0.0       # Draw value for defenders

    # Replay buffer
    replay_buffer_size = 50000        # Maximum samples (positions x symmetries) in the buffer
    min_buffer_size = 1000            # Minimum samples before training
    use_data_augmentation = True      # Train on random board symmetries of every position

    # Evaluation
    eval_games = 20                   # Games for the gating match (new vs best network)
    eval_win_rate = 0.55              # Score threshold to replace the best network
    eval_frequency = 5                # Evaluate every N iterations
    eval_mcts_sims_attacker = 100     # MCTS simulations for attacker in evaluation
    eval_mcts_sims_defender = 100     # MCTS simulations for defender in evaluation
    eval_c_puct = 1.4                 # MCTS exploration constant for evaluation
    eval_fpu_reduction = -0.5         # First Play Urgency for evaluation
    eval_temperature = 0.0            # Temperature for evaluation games (0 = deterministic)
    eval_temperature_mode = "fixed"   # "fixed", "king", or "decay"
    eval_temperature_threshold = 0    # Move threshold for eval temperature (for "fixed" mode)
    eval_temperature_decay_moves = 0  # Decay moves for eval (for "decay" mode)
    eval_vs_random_games = 10         # Games vs random (per colour)
    eval_vs_random_frequency = 1      # Evaluate vs random every N iterations

    # System
    num_workers = None                # Parallel self-play workers (None = all CPUs)
    device = "cpu"                    # Device of the training process
    temp_dir = None                   # Directory for worker files (e.g. "/dev/shm"); None = system default
    checkpoint_dir = "checkpoints"    # Directory for checkpoints, history and the resolved config
    save_frequency = 5                # Save a checkpoint every N iterations
    log_frequency = 1                 # Log every N iterations
    verbose = True                    # Print detailed logs

    # Kept for the Agent created in the training loop (self-play uses the per-side counts)
    num_mcts_simulations = 100

    def __init__(self):
        self.network = NetworkConfig()
        if self.num_workers is None:
            self.num_workers = mp.cpu_count()

    # --- derived from the rule set ----------------------------------------------------------
    @property
    def rules(self) -> Rules:
        return RULESETS[self.ruleset]

    @property
    def board_size(self) -> int:
        return self.rules.board_size

    @property
    def num_squares(self) -> int:
        return self.board_size * self.board_size

    @property
    def policy_size(self) -> int:
        """num_squares * 4 directions * (board_size - 1) distances"""
        return policy_size(self.board_size)

    # --- JSON form ---------------------------------------------------------------------------
    def to_dict(self) -> dict:
        return to_dict(self)

    def to_flat_dict(self) -> dict:
        return to_flat_dict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "TrainingConfig":
        return from_dict(data)


# The JSON sections and the TrainingConfig attributes in each
SECTIONS: Dict[str, List[str]] = {
    "game": ["ruleset", "max_game_length"],
    "self_play": ["num_iterations", "num_games_per_iteration", "num_mcts_sims_attacker",
                  "num_mcts_sims_defender", "c_puct", "fpu_reduction", "search_discount", "temperature",
                  "temperature_mode", "temperature_threshold", "temperature_decay_moves",
                  "add_dirichlet_noise", "dirichlet_alpha", "dirichlet_epsilon", "use_random_opening_moves",
                  "use_dynamic_sim_balancing", "dynamic_sim_adjustment_rate", "dynamic_sim_min_win_rate",
                  "reward_discount", "track_unique_states"],
    "training": ["batch_size", "num_epochs", "batches_per_epoch", "learning_rate", "lr_decay", "lr_floor",
                 "use_stepping_lr", "lr_step_iterations", "lr_step_values", "weight_decay",
                 "value_loss_weight", "ending_loss_weight", "moves_left_loss_weight",
                 "use_dynamic_boosting", "dynamic_boost_alpha", "dynamic_boost_min",
                 "dynamic_boost_max", "attacker_win_loss_boost", "draw_penalty_attacker",
                 "draw_penalty_defender"],
    "replay_buffer": ["replay_buffer_size", "min_buffer_size", "use_data_augmentation"],
    "evaluation": ["eval_games", "eval_win_rate", "eval_frequency", "eval_mcts_sims_attacker",
                   "eval_mcts_sims_defender", "eval_c_puct", "eval_fpu_reduction", "eval_temperature",
                   "eval_temperature_mode", "eval_temperature_threshold", "eval_temperature_decay_moves",
                   "eval_vs_random_games", "eval_vs_random_frequency"],
    "system": ["num_workers", "device", "temp_dir", "checkpoint_dir", "save_frequency", "log_frequency",
               "verbose"],
}
FIELD_SECTION = {field: section for section, fields in SECTIONS.items() for field in fields}


def to_dict(config: TrainingConfig) -> dict:
    """The sectioned JSON form of a config (network nested, all values JSON-serialisable)."""
    data = {"name": config.name}
    for section, fields in SECTIONS.items():
        data[section] = {field: copy.deepcopy(getattr(config, field)) for field in fields}
        if section == "game":
            data["network"] = config.network.to_dict()
    return data


def to_flat_dict(config: TrainingConfig) -> dict:
    """
    Every setting as one flat dict (as stored in checkpoints): the sectioned fields, the board size,
    the complete rules under 'rules' (so a checkpoint keeps its rules even if a named rule set is
    changed later) and the network config under 'network'.
    """
    flat = {"name": config.name, "board_size": config.board_size, "rules": config.rules.to_dict()}
    for fields in SECTIONS.values():
        for field in fields:
            flat[field] = copy.deepcopy(getattr(config, field))
    flat["network"] = config.network.to_dict()
    return flat


def from_dict(data: dict, base: Optional[TrainingConfig] = None) -> TrainingConfig:
    """
    A TrainingConfig from the sectioned JSON form. Keys left out keep the values of `base` (or the
    class defaults); unknown sections or keys are an error, so a typo cannot pass unnoticed.
    """
    config = copy.deepcopy(base) if base is not None else TrainingConfig()
    unknown = set(data) - set(SECTIONS) - {"name", "network"}
    if unknown:
        raise ValueError(f"Unknown config sections {sorted(unknown)}; valid: {sorted(SECTIONS) + ['name', 'network']}")
    config.name = data.get("name", config.name)
    for section, fields in SECTIONS.items():
        values = data.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Section '{section}' must be an object")
        bad = set(values) - set(fields)
        if bad:
            raise ValueError(f"Unknown keys {sorted(bad)} in section '{section}'; valid: {fields}")
        for field, value in values.items():
            setattr(config, field, value)
    if "network" in data:
        network = data["network"]
        config.network = network if isinstance(network, NetworkConfig) else NetworkConfig.from_dict(network)
    if config.num_workers is None:
        config.num_workers = mp.cpu_count()
    if config.ruleset not in RULESETS:
        raise ValueError(f"Unknown rule set '{config.ruleset}'; choose one of {sorted(RULESETS)}")
    return config


def load_config(path: str, base: Optional[TrainingConfig] = None) -> TrainingConfig:
    """Read a run config file; a "network" entry that is a string names a network JSON file."""
    with open(path) as f:
        data = json.load(f)
    if isinstance(data.get("network"), str):
        network_path = os.path.join(os.path.dirname(os.path.abspath(path)), data["network"])
        with open(network_path) as f:
            data["network"] = json.load(f)
    return from_dict(data, base)


def save_config(config: TrainingConfig, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
        f.write("\n")


def parse_override(text: str):
    """'section.key=value' -> (['section', 'key'], value), the value parsed as JSON if it is JSON."""
    if "=" not in text:
        raise ValueError(f"Override '{text}' must look like section.key=value")
    path, raw = text.split("=", 1)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return path.split("."), value


def apply_overrides(config: TrainingConfig, overrides: List[str]) -> TrainingConfig:
    """
    Apply command-line overrides: 'self_play.num_iterations=5', 'network.channels=48',
    'network.features.history_length=4', 'network.blocks=["line","line"]'. A bare key without a
    section is looked up in the sections.
    """
    data = config.to_dict()
    for text in overrides:
        keys, value = parse_override(text)
        if len(keys) == 1:
            if keys[0] == "name":
                keys = ["name"]
            elif keys[0] in FIELD_SECTION:
                keys = [FIELD_SECTION[keys[0]], keys[0]]
            else:
                raise ValueError(f"Unknown setting '{keys[0]}'")
        target = data
        for key in keys[:-1]:
            if key not in target or not isinstance(target[key], dict):
                raise ValueError(f"Unknown setting path '{'.'.join(keys)}'")
            target = target[key]
        if keys[-1] not in target and keys[0] != "name":
            raise ValueError(f"Unknown setting path '{'.'.join(keys)}'")
        target[keys[-1]] = value
    return from_dict(data)


def describe(config: TrainingConfig) -> str:
    """The config as indented JSON text (for the log)."""
    return json.dumps(config.to_dict(), indent=2)
