#!/usr/bin/env python3
"""
Plot the training progress of a run from its training_history.json or its output.txt.

Both files hold the same information. train.py writes training_history.json after every iteration
and prints the output that run_training.py --log saves as output.txt. The JSON file is the safer
source: a resumed run continues it from the resumed checkpoint, while output.txt also keeps the
output of iterations that were run again (here the last output of each iteration is used).
output.txt is read into the layout of training_history.json (described in train.py), so that all
plots are made from one layout.

Plots, saved next to the (first) input file:
  plot_elo.png                       ELO vs iteration 0, and the ELO difference of each evaluation
  plot_win_rates.png                 Win and draw rates in self-play and in the new-vs-best evaluation
  plot_losses.png                    Policy, value and L2 losses per iteration
  plot_gradients.png                 Gradient norms and average game length per iteration
  plot_buffer_time.png               Replay buffer size, iteration time and learning rate
  plot_epoch_losses.png              Policy, value and L2 losses per epoch
  plot_epoch_gradients.png           Gradient norms per epoch
  plot_diversity.png                 Unique board states, game length and draw rate
  plot_endings.png                   Share of games ending in each way (king escaped, ...), in
                                     self-play and in the new-vs-best evaluation
  plot_ending_moves_left_losses.png  Losses of the ending and moves-left outputs per epoch

Several files of the same kind, of runs that continue each other (a new run started with
--load-weights from the previous one), are joined; the cumulative ELO of each run then continues
from the end of the previous run.

Usage: python plot.py <training_history.json | output.txt> [more files of continuing runs ...]

Example: python plot.py checkpoints/brandubh/baseline/training_history.json
Example: python plot.py checkpoints/brandubh/baseline/output.txt
"""

import json
import os
import re
import sys
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


# Colors of the game endings: blues are defender wins, reds are attacker wins. Repetition is an
# attacker win in most rule sets and a draw in some, so it gets its own neutral color.
ENDING_COLORS = {
    'king_escaped': '#1f4e8c', 'exit_fort': '#4a86c8', 'attackers_blocked': '#9cc3ea',
    'king_captured': '#a32020', 'encircled': '#d9534f', 'defenders_blocked': '#f2a39a',
    'repetition': '#7a6a9a', 'move_limit': '#b5b5b5',
}

# The columns of the per-epoch table in output.txt, by number of columns (newer and older format)
EPOCH_COLUMNS = {
    14: ['policy_loss', 'value_loss', 'ending_loss', 'moves_left_loss', 'l2_loss', 'total_loss',
         'policy_grad', 'value_grad', 'val_policy_loss', 'val_value_loss', 'val_ending_loss',
         'val_moves_left_loss', 'val_l2_loss', 'val_total_loss'],
    10: ['policy_loss', 'value_loss', 'l2_loss', 'total_loss', 'policy_grad', 'value_grad',
         'val_policy_loss', 'val_value_loss', 'val_l2_loss', 'val_total_loss'],
}

# Lines of output.txt with one value for the history: (pattern, history key, conversion). Scores
# are printed in percent and stored as fractions.
VALUE_LINES = [
    (r'Generated (\d+) training positions', 'positions', int),
    (r'Average game length: ([\d.]+) moves', 'avg_game_length', float),
    (r'Replay buffer size: (\d+)', 'buffer_size', int),
    (r'Learning rate: ([\d.e-]+)', 'learning_rate', float),
    (r'New network (?:results: \d+W-\d+L-\d+D \(|win rate: )([\d.]+)%', 'win_rates', lambda text: float(text) / 100),
    (r'ELO (?:gain|difference): ([+-]?[\d.]+)', 'elo_gain', float),
    (r'Cumulative ELO \(vs iteration 0\): ([+-]?[\d.]+)', 'cumulative_elo', float),
    (r'Total: (?:\d+/\d+ wins|\d+W-\d+L-\d+D) \(([\d.]+)%', 'vs_random_win_rate', lambda text: float(text) / 100),
    (r'ELO vs Random: ([+-]?\d+)', 'vs_random_elo', float),
    (r'^\s+Total:\s+([\d.]+)s\s*$', 'total_time', float),  # of the timing summary
]


def parse_output_file(filepath: str) -> dict:
    """Read the printed output of train.py into the layout of training_history.json, with the
    values the plots use. Older output formats are read as far as they go. When a run was
    restarted, the last output of each iteration is used."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    by_iteration = {}  # iteration number -> {history key: value} from its output
    values = None      # the dict of the iteration being read
    section = None     # '[1/4]' self-play, '[2/4]' training, '[3/4]' vs random, '[4/4]' new vs best
    fixed_simulations = None  # [attacker, defender] simulations of a run without balancing
    for i, line in enumerate(lines):
        match = re.match(r'Iteration (\d+)/\d+', line)
        if match:
            # An iteration run again starts over
            values = by_iteration[int(match.group(1))] = {'simulations': fixed_simulations}
            continue
        # Printed once at the start of a run (and of each restart)
        match = re.search(r'Using fixed simulation counts \(attacker=(\d+), defender=(\d+)\)', line)
        if match:
            fixed_simulations = [int(match.group(1)), int(match.group(2))]
        if re.search(r'Using dynamic simulation balancing', line):
            fixed_simulations = None
        if re.match(r'\[\d/4\]', line):
            section = line[:5]
            continue
        if values is None:
            continue
        # The two evaluations print the same lines; their results go to different lists
        results_key = 'vs_random_results' if section == '[3/4]' else 'eval_results'

        for pattern, key, convert in VALUE_LINES:
            match = re.search(pattern, line)
            if match:
                values[key] = convert(match.group(1))

        # MCTS simulations per move of each side in this iteration's self-play (with balancing)
        match = re.search(r'Using dynamic simulation counts: Attacker (\d+), Defender (\d+)', line)
        if match:
            values['simulations'] = [int(match.group(1)), int(match.group(2))]

        # Self-play results (game counts)
        match = re.match(r'Results: (\d+) attacker wins.*?, (\d+) defender wins.*?, (\d+) draws', line)
        if match:
            values['attacker_wins'], values['defender_wins'], values['draws'] = map(int, match.groups())

        # Board state diversity of self-play, e.g. "Unique states (skip 6 ply): 8809 / 8812 (99.97%)"
        match = re.search(r'Unique states \(skip (initial|6 ply)\): (\d+) / (\d+)', line)
        if match:
            name = 'skip_initial' if match.group(1) == 'initial' else 'skip_6ply'
            values.setdefault('unique_states', {}).update(
                {f'unique_{name}': int(match.group(2)), f'total_{name}': int(match.group(3))})

        # Wins and draws of the evaluated network per side, e.g. "As Attacker: 42/64 wins, 0 draws"
        # (older outputs have no draws)
        match = re.search(r'As (Attacker|Defender): (\d+)/(\d+) wins(?:, (\d+) draws)?', line)
        if match:
            side = match.group(1).lower() + 's'
            values.setdefault(results_key, {}).setdefault(side, {}).update(
                games=int(match.group(3)), wins=int(match.group(2)),
                draws=None if match.group(4) is None else int(match.group(4)))

        # Table of how the games ended. The line after "How the games ended:" names the columns
        # (self-play games; or the evaluated network as attackers, as defenders), then each line is
        # an ending with "count ( share%)" per column, e.g. "  encircled    18 ( 28.1%)    2 (  3.1%)"
        if 'How the games ended:' in line:
            columns = []
            for row in lines[i + 2:]:
                row_match = re.match(r'\s+([a-z_]+)((?:\s+\d+ \(\s*[\d.]+%\))+)\s*$', row)
                if not row_match:
                    break
                counts = [int(n) for n in re.findall(r'(\d+) \(', row_match.group(2))]
                if not columns:
                    columns = [{} for _ in counts]
                for column, count in zip(columns, counts):
                    column[row_match.group(1)] = count
            if len(columns) == 1:
                values['endings'] = columns[0]
            else:
                for side, column in zip(('attackers', 'defenders'), columns):
                    values.setdefault(results_key, {}).setdefault(side, {})['endings'] = column

        # Losses and gradient norms averaged over the iteration, e.g. "  Training   - Policy: 4.67,
        # Value: 0.07, Ending: 0.17, Moves left: 0.11, L2: 132.2, Total: 4.97", and the older
        # "Final losses - Policy: 4.67, Value: 0.07"
        match = re.match(r'\s*(Training|Validation|Gradients|Final losses)\s+-\s+(.*)', line)
        if match:
            prefix, suffix = {'Training': ('', '_loss'), 'Final losses': ('', '_loss'),
                              'Validation': ('val_', '_loss'), 'Gradients': ('', '_grad')}[match.group(1)]
            for name, value in re.findall(r'([A-Za-z][A-Za-z0-9 ]*): ([\d.]+|nan)', match.group(2)):
                values[prefix + name.lower().replace(' ', '_') + suffix] = float(value)

        # Per-epoch table rows, "  epoch | training losses | gradients | validation losses", and
        # the oldest format "Epoch 1/10: policy=2.3456, value=0.5678, ..."
        match = re.match(r'\s+\d+ \|(.*)$', line)
        numbers = re.findall(r'nan|[\d.]+', match.group(1)) if match else []
        match = re.search(r'Epoch \d+/\d+: policy=([\d.]+), value=([\d.]+)', line)
        if match:
            numbers = list(match.groups())
        columns = {**EPOCH_COLUMNS, 2: ['policy_loss', 'value_loss']}.get(len(numbers), [])
        for key, number in zip(columns, numbers):
            values.setdefault('epochs', {}).setdefault(key, []).append(float(number))

    # Collect the iterations into the lists of training_history.json
    iteration_keys = EPOCH_COLUMNS[14] + ['epochs', 'learning_rate', 'buffer_size']
    history = {key: [] for key in ['iterations'] + iteration_keys
               + ['eval_iterations', 'win_rates', 'elo_gain', 'cumulative_elo', 'eval_results',
                  'vs_random_iterations', 'vs_random_win_rate', 'vs_random_elo', 'vs_random_results']}
    history['selfplay_results'] = {key: [] for key in ['attacker_wins', 'defender_wins', 'draws',
                                                       'avg_game_length', 'endings', 'positions',
                                                       'simulations', 'unique_states']}
    history['timing'] = {'total_time': []}
    for iteration in sorted(by_iteration):
        values = by_iteration[iteration]
        if 'attacker_wins' not in values:  # no self-play results (yet)
            continue
        history['iterations'].append(iteration)
        for key in iteration_keys:
            history[key].append(values.get(key))
        for key in history['selfplay_results']:
            history['selfplay_results'][key].append(values.get(key))
        history['timing']['total_time'].append(values.get('total_time'))
        if 'cumulative_elo' in values:
            history['eval_iterations'].append(iteration)
            for key in ['win_rates', 'elo_gain', 'cumulative_elo', 'eval_results']:
                history[key].append(values.get(key))
        if 'vs_random_elo' in values:
            history['vs_random_iterations'].append(iteration)
            for key in ['vs_random_win_rate', 'vs_random_elo', 'vs_random_results']:
                history[key].append(values.get(key))
    return history


def side_results(results: list, side: str, key: str) -> np.ndarray:
    """One value per evaluation from evaluation_summary results (see train.py): `key` ('games',
    'wins' or 'draws') of the evaluated network as `side`; NaN where it was not recorded."""
    return np.array([(result or {}).get(side, {}).get(key) for result in results], dtype=float)


def plot_elo(history: dict, output_dir: str):
    """ELO of the best network vs iteration 0, and the ELO difference of each evaluation."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Cumulative ELO vs First Version', fontsize=12, color='tab:blue')
    ax1.plot(history['eval_iterations'], history['cumulative_elo'], 's-', color='tab:blue',
             label='Cumulative ELO', linewidth=2, markersize=5)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=0)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Individual ELO Difference', fontsize=12, color='tab:orange')
    ax2.plot(history['eval_iterations'], np.array(history['elo_gain'], dtype=float), 'o-',
             color='tab:orange', label='ELO Difference', linewidth=2, markersize=4, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.axhline(y=0, color='tab:orange', linestyle='--', linewidth=1, alpha=0.5)

    ax1.set_title('ELO Ratings During Training', fontsize=14, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_elo.png'), dpi=150)
    plt.close()


def plot_win_rates(history: dict, output_dir: str):
    """Win and draw rates of the self-play games, and the results of the new network in the
    new-vs-best evaluations."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Self-play: shares of all games of the iteration
    ax = axes[0]
    selfplay = history['selfplay_results']
    counts = {name: np.array(selfplay[key], dtype=float)
              for name, key in [('Attacker', 'attacker_wins'), ('Defender', 'defender_wins'), ('Draw', 'draws')]}
    total = sum(counts.values())
    for (name, count), marker in zip(counts.items(), ['o-', 's-', '^-']):
        ax.plot(history['iterations'], 100 * count / total, marker, label=name, linewidth=2, markersize=4)
    # The attackers' share of the MCTS simulations per move of both sides: the win rates depend on
    # how the search effort is divided (balanced so that the attacker score stays in a band)
    simulations = np.array([counts if counts else [np.nan, np.nan] for counts in selfplay['simulations']],
                           dtype=float)
    ax.plot(history['iterations'], 100 * simulations[:, 0] / simulations.sum(axis=1), '--',
            color='tab:blue', label='Attacker share of simulations', linewidth=1.5)
    ax.axhline(y=50, color='gray', linestyle=':', linewidth=1)
    ax.set_title('Self-Play Win Rates and Simulation Balance', fontsize=13, fontweight='bold')

    # Evaluation: the new network's wins as each side, its draws, and its score (draws count half)
    ax = axes[1]
    results = history['eval_results']
    games = {side: side_results(results, side, 'games') for side in ('attackers', 'defenders')}
    draws = side_results(results, 'attackers', 'draws') + side_results(results, 'defenders', 'draws')
    for label, rate, marker in [
            ('As Attacker (wins)', side_results(results, 'attackers', 'wins') / games['attackers'], 'o-'),
            ('As Defender (wins)', side_results(results, 'defenders', 'wins') / games['defenders'], 's-'),
            ('Draws', draws / (games['attackers'] + games['defenders']), '^-'),
            ('Score', np.array(history['win_rates'], dtype=float), 'D-')]:
        ax.plot(history['eval_iterations'], 100 * rate, marker, label=label, linewidth=2, markersize=4)
    ax.set_title('Evaluation: New vs Best Network', fontsize=13, fontweight='bold')

    for ax in axes:
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_ylabel('Share of Games (%)', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_win_rates.png'), dpi=150)
    plt.close()


def plot_policy_value_l2(x: np.ndarray, losses: dict, xlabel: str, title: str, path: str, per_epoch: bool):
    """Policy, value and L2 losses (training solid, validation dashed) on three y-axes.

    Args:
        x: Iterations, or epoch numbers
        losses: '<name>_loss' and 'val_<name>_loss' for name policy, value, l2 (missing keys are
            not plotted)
        per_epoch: Lines without markers (there are many epochs)
    """
    fig, ax1 = plt.subplots(figsize=(14 if per_epoch else 12, 6))
    axes = [ax1, ax1.twinx(), ax1.twinx()]
    axes[2].spines['right'].set_position(('outward', 60))

    for ax, name, label, marker, color, val_color in [
            (axes[0], 'policy', 'Policy', 'o', 'tab:blue', 'lightblue'),
            (axes[1], 'value', 'Value', 's', 'tab:red', 'lightcoral'),
            (axes[2], 'l2', 'L2', '^', 'tab:green', 'lightgreen')]:
        style = {'linewidth': 1.5} if per_epoch else {'marker': marker, 'linewidth': 2, 'markersize': 4}
        if f'{name}_loss' in losses:
            ax.plot(x, np.array(losses[f'{name}_loss'], dtype=float), '-', color=color,
                    label=f'{label} Loss (Train)', **style)
        if f'val_{name}_loss' in losses:
            ax.plot(x, np.array(losses[f'val_{name}_loss'], dtype=float), '--', color=val_color,
                    label=f'{label} Loss (Val)', **style)
        ax.set_ylabel(f'{label} Loss', fontsize=12, color=color)
        ax.tick_params(axis='y', labelcolor=color)

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title, fontsize=14, fontweight='bold')
    handles = [handle for ax in axes for handle in ax.get_legend_handles_labels()[0]]
    ax1.legend(handles=handles, loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_gradients(history: dict, output_dir: str):
    """Gradient norms of the policy and value heads, and the average game length, per iteration."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    iterations = history['iterations']
    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.set_ylabel('Gradient Norm', fontsize=12)
    ax1.plot(iterations, np.array(history['policy_grad'], dtype=float), 'o-', color='tab:blue',
             label='Policy Gradient Norm', linewidth=2, markersize=4)
    ax1.plot(iterations, np.array(history['value_grad'], dtype=float), 's-', color='tab:red',
             label='Value Gradient Norm', linewidth=2, markersize=4)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Avg Game Length (moves)', fontsize=12, color='tab:green')
    ax2.plot(iterations, np.array(history['selfplay_results']['avg_game_length'], dtype=float), '^-',
             color='tab:green', label='Avg Game Length', linewidth=2, markersize=4, alpha=0.7)
    ax2.tick_params(axis='y', labelcolor='tab:green')

    ax1.set_title('Gradient Norms and Game Length (Per Iteration)', fontsize=14, fontweight='bold')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_gradients.png'), dpi=150)
    plt.close()


def plot_buffer_and_time(history: dict, output_dir: str):
    """Replay buffer size, iteration time and learning rate, on three y-axes."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    axes = [ax1, ax1.twinx(), ax1.twinx()]
    axes[2].spines['right'].set_position(('outward', 60))

    for ax, values, label, marker, color in [
            (axes[0], history['buffer_size'], 'Buffer Size', 'o-', 'tab:green'),
            (axes[1], history['timing']['total_time'], 'Total Time (s)', 's-', 'tab:orange'),
            (axes[2], history['learning_rate'], 'Learning Rate', '^-', 'tab:blue')]:
        ax.plot(history['iterations'], np.array(values, dtype=float), marker, color=color,
                label=label, linewidth=2, markersize=4)
        ax.set_ylabel(label, fontsize=12, color=color)
        ax.tick_params(axis='y', labelcolor=color)
        ax.set_ylim(bottom=0)
    ax1.ticklabel_format(style='plain', axis='y')  # buffer size without an exponent

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Buffer Size, Iteration Time, and Learning Rate', fontsize=14, fontweight='bold')
    handles = [handle for ax in axes for handle in ax.get_legend_handles_labels()[0]]
    ax1.legend(handles=handles, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_buffer_time.png'), dpi=150)
    plt.close()


def plot_epoch_gradients(epochs: dict, output_dir: str):
    """Gradient norms of the policy and value heads per epoch, over all iterations."""
    fig, ax = plt.subplots(figsize=(12, 6))

    epoch_numbers = np.arange(len(epochs['policy_loss']))
    ax.plot(epoch_numbers, epochs['policy_grad'], '-', color='tab:blue', label='Policy Gradient Norm',
            linewidth=1.5, alpha=0.8)
    ax.plot(epoch_numbers, epochs['value_grad'], '-', color='tab:red', label='Value Gradient Norm',
            linewidth=1.5, alpha=0.8)

    ax.set_xlabel('Training Epoch (Cumulative)', fontsize=12)
    ax.set_ylabel('Gradient Norm', fontsize=12)
    ax.set_title('Gradient Norms (Per Epoch - All Iterations)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_epoch_gradients.png'), dpi=150)
    plt.close()


def plot_diversity(history: dict, output_dir: str):
    """Share of distinct board states in self-play, average game length and draw rate."""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    axes = [ax1, ax1.twinx(), ax1.twinx()]
    axes[2].spines['right'].set_position(('outward', 60))

    iterations = history['iterations']
    selfplay = history['selfplay_results']

    # Distinct states as a share of all states, without the first 1 or 6 plies of each game
    for name, label, marker, color in [('skip_initial', 'Unique States (skip initial)', 'o-', 'tab:blue'),
                                       ('skip_6ply', 'Unique States (skip 6 ply)', 's-', 'tab:cyan')]:
        share = [100 * states[f'unique_{name}'] / states[f'total_{name}']
                 if states and states.get(f'total_{name}') else np.nan for states in selfplay['unique_states']]
        ax1.plot(iterations, share, marker, color=color, label=label, linewidth=2, markersize=4)
    ax1.set_ylabel('Unique States Fraction (%)', fontsize=12, color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim([0, 105])

    axes[1].plot(iterations, np.array(selfplay['avg_game_length'], dtype=float), '^-', color='tab:green',
                 label='Avg Game Length', linewidth=2, markersize=4, alpha=0.7)
    axes[1].set_ylabel('Avg Game Length (moves)', fontsize=12, color='tab:green')
    axes[1].tick_params(axis='y', labelcolor='tab:green')

    games = (np.array(selfplay['attacker_wins'], dtype=float) + np.array(selfplay['defender_wins'], dtype=float)
             + np.array(selfplay['draws'], dtype=float))
    axes[2].plot(iterations, 100 * np.array(selfplay['draws'], dtype=float) / games, 'D-', color='tab:orange',
                 label='Draw Rate', linewidth=2, markersize=4, alpha=0.7)
    axes[2].set_ylabel('Draw Rate (%)', fontsize=12, color='tab:orange')
    axes[2].tick_params(axis='y', labelcolor='tab:orange')
    axes[2].set_ylim([0, 105])

    ax1.set_xlabel('Iteration', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title('Game Diversity Metrics (Per Iteration)', fontsize=14, fontweight='bold')
    handles = [handle for ax in axes for handle in ax.get_legend_handles_labels()[0]]
    ax1.legend(handles=handles, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_diversity.png'), dpi=150)
    plt.close()


def plot_endings(history: dict, output_dir: str):
    """Share of games that ended in each way, stacked, for self-play and for the new-vs-best
    evaluation (the games of the new network as attackers and as defenders added together)."""
    # (iteration, {ending: number of games}) for each iteration with a table
    selfplay = [(iteration, endings) for iteration, endings
                in zip(history['iterations'], history['selfplay_results']['endings']) if endings]
    evaluation = [(iteration, Counter(result['attackers']['endings']) + Counter(result['defenders']['endings']))
                  for iteration, result in zip(history['eval_iterations'], history['eval_results'])
                  if result and 'endings' in result.get('attackers', {}) and 'endings' in result.get('defenders', {})]
    # All endings that occur, in the order of the tables
    names = list(dict.fromkeys(name for _, endings in selfplay + evaluation for name in endings))
    colors = [ENDING_COLORS.get(name, 'black') for name in names]

    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    for ax, title, tables in [(axes[0], 'Self-Play Games', selfplay),
                              (axes[1], 'Evaluation Games (New vs Best Network)', evaluation)]:
        if tables:
            # Rows are iterations, columns are endings; an ending missing from a table counts as 0
            counts = np.array([[endings.get(name, 0) for name in names] for _, endings in tables], dtype=float)
            percents = 100 * counts / counts.sum(axis=1, keepdims=True)
            ax.stackplot([iteration for iteration, _ in tables], percents.T, labels=names, colors=colors,
                         edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Iteration', fontsize=12)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_ylim([0, 100])
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel('Share of Games (%)', fontsize=12)

    # One legend for both panels, listed top to bottom in the same order as the stacked areas
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    fig.legend(handles[::-1], names[::-1], loc='center right', fontsize=10,
               title='Blue: defenders win\nRed: attackers win', title_fontsize=9)

    plt.tight_layout(rect=[0, 0, 0.86, 1])
    plt.savefig(os.path.join(output_dir, 'plot_endings.png'), dpi=150)
    plt.close()


def plot_ending_moves_left_losses(epochs: dict, output_dir: str):
    """Per-epoch losses of the ending output and the moves-left output, training and validation."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    epoch_numbers = np.arange(len(epochs['policy_loss']))
    for ax, title, color, name in [(axes[0], 'Ending Loss', 'tab:purple', 'ending_loss'),
                                   (axes[1], 'Moves-Left Loss', 'tab:brown', 'moves_left_loss')]:
        ax.plot(epoch_numbers, epochs[name], '-', color=color, label='Train', linewidth=1.5, alpha=0.8)
        ax.plot(epoch_numbers, epochs['val_' + name], '--', color=color, label='Val', linewidth=1.5, alpha=0.5)
        ax.set_xlabel('Training Epoch (Cumulative)', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title} (Per Epoch - All Iterations)', fontsize=13, fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'plot_ending_moves_left_losses.png'), dpi=150)
    plt.close()


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot.py <training_history.json | output.txt> [more files of continuing runs ...]")
        sys.exit(1)
    filepaths = sys.argv[1:]

    histories = []
    for filepath in filepaths:
        print(f"Reading {filepath}")
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                histories.append(json.load(f))
            if 'eval_iterations' not in histories[-1]:
                sys.exit(f"{filepath} was written by an older train.py and does not say at which "
                         "iterations the evaluations ran. Plot the run's output.txt instead, or "
                         "resume the run once (train.py then fills this in).")
        else:
            histories.append(parse_output_file(filepath))

    # Join runs that continue each other. A later run counts its ELO from its own start.
    history = histories[0]
    for later in histories[1:]:
        offset = history['cumulative_elo'][-1] if history['cumulative_elo'] else 0.0
        later['cumulative_elo'] = [elo + offset for elo in later['cumulative_elo']]
        for key, values in history.items():
            if key in ('selfplay_results', 'timing'):
                for subkey in values:
                    values[subkey] += later[key][subkey]
            elif isinstance(values, list):
                values += later[key]

    # The per-epoch values of all iterations joined into one array per key (NaN where an older
    # output format lacks a key)
    tables = [table for table in history['epochs'] if table]
    epochs = {key: np.concatenate([np.array(table.get(key, [np.nan] * len(table['policy_loss'])), dtype=float)
                                   for table in tables])
              for key in dict.fromkeys(key for table in tables for key in table)}

    output_dir = os.path.dirname(filepaths[0]) or '.'
    print(f"\nCreating plots...")
    print(f"  - {len(history['iterations'])} iterations, {len(history['eval_iterations'])} evaluations, "
          f"{len(history['vs_random_iterations'])} evaluations vs random")
    print(f"  - {len(tables)} iterations with per-epoch losses "
          f"({len(epochs.get('policy_loss', []))} epochs)")

    if history['eval_iterations']:
        plot_elo(history, output_dir)
        print("  ✓ Saved plot_elo.png")
    plot_win_rates(history, output_dir)
    print("  ✓ Saved plot_win_rates.png")
    plot_policy_value_l2(history['iterations'], history, 'Iteration', 'Training Losses (Per Iteration)',
                         os.path.join(output_dir, 'plot_losses.png'), per_epoch=False)
    print("  ✓ Saved plot_losses.png")
    plot_gradients(history, output_dir)
    print("  ✓ Saved plot_gradients.png")
    plot_buffer_and_time(history, output_dir)
    print("  ✓ Saved plot_buffer_time.png")
    plot_diversity(history, output_dir)
    print("  ✓ Saved plot_diversity.png")
    if epochs:
        plot_policy_value_l2(np.arange(len(epochs['policy_loss'])), epochs, 'Training Epoch (Cumulative)',
                             'Training Losses (Per Epoch - All Iterations)',
                             os.path.join(output_dir, 'plot_epoch_losses.png'), per_epoch=True)
        print("  ✓ Saved plot_epoch_losses.png")
    if 'policy_grad' in epochs:
        plot_epoch_gradients(epochs, output_dir)
        print("  ✓ Saved plot_epoch_gradients.png")
    if np.any(np.isfinite(epochs.get('ending_loss', []))):
        plot_ending_moves_left_losses(epochs, output_dir)
        print("  ✓ Saved plot_ending_moves_left_losses.png")
    if any(history['selfplay_results']['endings']) or any(history['eval_results']):
        plot_endings(history, output_dir)
        print("  ✓ Saved plot_endings.png")

    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
