"""
Tests for the training-pipeline fixes:
1. Symmetry augmentation: every augmented policy target matches its augmented board.
2. StateHistory.from_game rebuilds the same history planes as tracking them move by move.
3. MCTS._evaluate (softmax over legal moves) equals the masked softmax over all moves.
4. play_game plays under the given rule set (here: defenders first).
5. FPU: an unvisited child is valued at the parent's value, seen from the player to move.
6. History planes: the network input during search has the same layout as the training states.
7. One MCTS simulation gives a finite policy (the network priors).
8. select_move and select_move_with_stats use the temperature value, and a time limit works.
9. play_game counts a win on the last allowed ply as a win.
10. An unarmed king neither captures nor helps to capture.
11. Random rollouts return the result for the player to move at the start of the rollout.
12. load_state_dict_compatible keeps the player plane when the history length differs.
13. SimulationBalancer: min_win_rate 0.5 reproduces the earlier rule; a band [m, 1-m] only pushes
    outside it, drifts back to an equal split inside it, and counts draws as half a win.
14. The replay buffer's symmetry tables equal augment_sample; the ring buffer keeps the newest data.
15. train_network: clear error for a too small buffer; small buffers, empty validation sets and
    draws work.
16. Temperature 0.001 gives finite probabilities, and a search puts the network into eval mode.
    (Also: self-play gives each position the game's ending and its plies left.)
17. 9x9 and 11x11 networks with history planes search their games.
18. The cached legal moves equal freshly generated ones.
19. Checkpoints are written atomically, hold all settings and return their training history.
20. The GUI takes the rule set, search settings and move limit from a checkpoint.

Run with: python test_training_fixes.py (exit code 1 if a test fails)
"""

import sys
from dataclasses import replace

import numpy as np
import torch

import agent as agent_module
from agent import Agent, RandomAgent, play_game
from features import FeatureSpec, StateHistory
from mcts import MCTS, MCTSNode, RandomRolloutMCTS
from network_config import NetworkConfig
from tafl import ATTACKER, DEFENDER, EMPTY, ENDINGS, KING, RULESETS, Tafl, move_index, policy_size
from tafl_network import TaflNet, move_index_tables
from train import (augment_sample, symmetry_permutations, ReplayBuffer, SimulationBalancer,
                   TrainingConfig, generate_self_play_data, train_network, save_checkpoint,
                   load_checkpoint, LOSS_KEYS)

# Brandubh without the throne and with a king captured by two attackers everywhere: the positions
# of the tests below were written for these rules
SIMPLE_7X7 = replace(RULESETS["brandubh"], name="simple_7x7", throne=False, king_capture="weak",
                     encirclement=False)


def small_network(board_size: int, history_length: int = 1) -> TaflNet:
    """A tiny untrained network (one res block of 8 channels)."""
    return TaflNet(NetworkConfig(channels=8, blocks=["res"],
                                 features=FeatureSpec(history_length=history_length)), board_size)


def random_position(rules, rng, max_plies=14):
    """Play a random number of random moves and return a non-finished game."""
    while True:
        game = Tafl(rules)
        for _ in range(rng.integers(1, max_plies)):
            moves = game.get_legal_moves()
            game.make_move(moves[rng.integers(len(moves))])
            if game.game_over:
                break
        if not game.game_over:
            return game


def set_position(game, pieces: dict, player: int):
    """Put `pieces` ({(row, col): piece}) on an empty board with `player` to move; returns the game."""
    game.board[:] = EMPTY
    for (r, c), piece in pieces.items():
        game.board[r, c] = piece
    game.current_player = player
    game.position_history = [game._position_key()]
    return game


def test_augmentation_consistent():
    """Each of the 8 symmetries must put policy mass only on moves that are legal-looking on its board."""
    rng = np.random.default_rng(0)
    for rules in (RULESETS["brandubh"], RULESETS["sami_tablut"]):
        n = rules.board_size
        to_square, _ = move_index_tables(n)
        for _ in range(20):
            game = random_position(rules, rng)
            state, player, legal = game.get_state(), game.current_player, game.get_legal_moves()
            policy = np.zeros(policy_size(n), dtype=np.float32)
            for move in legal:
                policy[move_index(move, n)] = 1.0 / len(legal)
            augmented = augment_sample(state, policy, 0.5, board_size=n)
            assert len(augmented) == 8
            for aug_state, aug_policy, aug_value in augmented:
                assert aug_value == 0.5 and np.isclose(aug_policy.sum(), 1.0)
                for idx in np.nonzero(aug_policy)[0]:
                    (fr, fc), (tr, tc) = divmod(idx // (4 * (n - 1)), n), divmod(to_square[idx], n)
                    # Plane 0 = attackers; planes 1 + 2 = defenders and king
                    own_piece = aug_state[0, fr, fc] if player == 0 else aug_state[1, fr, fc] + aug_state[2, fr, fc]
                    assert own_piece > 0, f"{rules.name}: move {idx} does not start on own piece"
                    assert aug_state[0:3, tr, tc].sum() == 0, f"{rules.name}: move {idx} ends on a piece"
    print("  augmentation: all 8 symmetries consistent on 7x7 and 9x9")
    return True


def test_from_game_matches_tracked_history():
    """from_game must give the same planes as pushing piece planes after every move (as in self-play)."""
    rng = np.random.default_rng(1)
    for history_length in (1, 4):
        game = Tafl("brandubh")
        tracked = StateHistory(history_length=history_length, board_size=7)
        tracked.push(game.get_piece_planes())
        for _ in range(30):
            rebuilt = StateHistory.from_game(game, history_length)
            expected = tracked.get_state_with_history(game.current_player)
            assert np.array_equal(rebuilt.get_state_with_history(game.current_player), expected)
            moves = game.get_legal_moves()
            game.make_move(moves[rng.integers(len(moves))])
            if game.game_over:
                break
            tracked.push(game.get_piece_planes())
    print("  from_game: rebuilt history equals tracked history")
    return True


def test_evaluate_matches_masked_softmax():
    """Softmax over legal moves must equal the masked softmax over all moves, restricted to legal moves."""
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    network = small_network(7).eval()
    mcts = MCTS(network, num_simulations=10)
    for _ in range(10):
        game = random_position("brandubh", rng)
        legal = game.get_legal_moves()
        probs, value = mcts._evaluate(game, legal)
        with torch.inference_mode():
            logits = network(torch.from_numpy(game.get_state()).unsqueeze(0))[0]
        logits = logits.numpy()[0]
        mask = np.zeros(policy_size(7), dtype=np.float32)
        mask[[move_index(m, 7) for m in legal]] = 1.0
        masked = MCTS._softmax(logits * mask + (1 - mask) * (-1e8))[[move_index(m, 7) for m in legal]]
        assert np.allclose(probs, masked, atol=1e-6) and np.isclose(probs.sum(), 1.0)
    print("  _evaluate: legal-move softmax equals masked softmax")
    return True


class Scripted:
    """An agent that plays given moves and records the rule set and player of every call."""

    def __init__(self, moves=None):
        self.moves, self.calls = list(moves or []), []

    def select_move(self, game, temperature=0.0):
        self.calls.append((game.rules.name, game.current_player))
        return self.moves.pop(0) if self.moves else game.get_legal_moves()[0]


def test_play_game_uses_rules():
    """play_game must play under the given rule set: in Sami Tablut with defenders first, the
    defenders' agent moves first."""
    attackers, defenders = Scripted(), Scripted()
    play_game(attackers, defenders, "sami_tablut", display=False, max_moves=3)
    assert defenders.calls[0] == ("sami_tablut", 1) and len(defenders.calls) == 2
    print("  play_game: game played under the given rule set")
    return True


def test_fpu_uses_parent_value():
    """An unvisited child must be valued at parent value + fpu_reduction (the sign used to be flipped)."""
    game = Tafl("brandubh")
    for parent_q, expect_unvisited in [(+0.8, True), (-0.8, False)]:
        root = MCTSNode(game.clone())
        root.expand(np.ones(len(game.get_legal_moves())))
        visited_move = next(iter(root.children))
        # The visited child is worth 0.5 less than the parent (seen from the player to move) when the
        # parent is winning, and 0.5 more when it is losing
        child_q = parent_q - 0.5 if parent_q > 0 else parent_q + 0.5
        child = root.children[visited_move]
        child.visit_count, child.total_value, child.mean_value = 1, -child_q, -child_q
        root.visit_count, root.total_value, root.mean_value = 2, 2 * parent_q, parent_q
        move, _ = root.select_child(c_puct=0.01, fpu_reduction=0.0)
        assert (move != visited_move) == expect_unvisited, f"parent Q {parent_q}: wrong child chosen"
    print("  FPU: unvisited children use the parent's value")
    return True


def test_search_input_matches_training_state():
    """During search the network must see [current, previous, ...], like the stored training states."""

    class RecordingNet(torch.nn.Module):
        def __init__(self, net):
            super().__init__()
            self.net, self.inputs = net, []

        def forward(self, x):
            self.inputs.append(x[0].numpy().copy())
            return self.net(x)

    torch.manual_seed(0)
    rng = np.random.default_rng(3)
    history_length = 3
    game = Tafl("brandubh")
    history = StateHistory(history_length, 7)
    history.push(game.get_piece_planes())
    for _ in range(3):  # track the history as self-play does
        game.make_move(game.get_legal_moves()[rng.integers(5)])
        history.push(game.get_piece_planes())
    train_state = history.get_state_with_history(game.current_player)

    for state_history in (history, None):  # self-play path, and the from_game path (Agent, play.py)
        net = RecordingNet(small_network(7, history_length).eval())
        MCTS(net, num_simulations=4, history_length=history_length).search(game, state_history=state_history)
        assert np.array_equal(net.inputs[0], train_state), "root input differs from the training state"
        # First child: its own planes, then the root's planes, then the one before
        assert np.array_equal(net.inputs[1][3:9], train_state[0:6]), "child input has the wrong history"
        assert not np.array_equal(net.inputs[1][0:3], net.inputs[1][3:6]), "child position appears twice"
    print("  history planes: search input matches the training layout")
    return True


def test_single_simulation_policy_is_finite():
    """With one simulation no child is visited; the visit distribution must fall back to the priors."""
    mcts = MCTS(small_network(7).eval(), num_simulations=1)
    probs = np.array(list(mcts.search(Tafl("brandubh")).values()))
    assert np.all(np.isfinite(probs)) and np.isclose(probs.sum(), 1.0)
    print("  one simulation: finite policy")
    return True


def test_temperature_and_time_limit():
    """A low temperature must almost always pick the most visited move (it used to act like 1.0)."""
    game = Tafl("brandubh")
    a, b = game.get_legal_moves()[:2]
    agent = Agent(small_network(7).eval(), num_simulations=2)
    agent.mcts.search = lambda g, state_history=None, time_limit=None: {a: 0.9, b: 0.1}
    np.random.seed(0)
    assert sum(agent.mcts.select_move(game, temperature=0.1) == b for _ in range(2000)) < 20
    assert sum(agent.select_move_with_stats(game, temperature=0.1)[0] == b for _ in range(2000)) < 20

    # Time-limited search (used by play.py --time-per-move). One second, so that even a fully loaded
    # node runs more than num_simulations simulations
    agent = Agent(small_network(7).eval(), num_simulations=2)
    move, _, _ = agent.select_move_with_stats(game, time_limit=1.0)
    assert move in game.get_legal_moves() and agent.mcts.root.visit_count > 2
    print("  temperature: low temperature picks the most visited move; time limit works")
    return True


def test_play_game_counts_last_ply_win():
    """A king capture on the last allowed ply must count as an attacker win, not as a draw."""
    start = {(3, 3): KING, (3, 1): ATTACKER, (1, 4): ATTACKER, (5, 0): ATTACKER, (6, 5): DEFENDER}
    # Ply 3 (the first ply beyond max_moves=2) sandwiches the king between (3,2) and (3,4)
    attackers, defenders = Scripted([(3, 1, 3, 2), (1, 4, 3, 4)]), Scripted([(6, 5, 6, 4)])
    original_tafl = agent_module.Tafl
    agent_module.Tafl = lambda rules: set_position(original_tafl(rules), start, player=0)
    try:
        game = play_game(attackers, defenders, SIMPLE_7X7, display=False, max_moves=2)
    finally:
        agent_module.Tafl = original_tafl
    assert game.winner == 0 and game.end_reason == "king_captured", \
        f"got {game.winner}, {game.end_reason}"
    print("  play_game: a win on the last allowed ply counts")
    return True


def test_unarmed_king_does_not_capture():
    """An unarmed king must not capture, also not with a corner or as the second piece."""
    unarmed = replace(SIMPLE_7X7, king_armed=False)
    # King moves next to its own defender, which stands next to a corner
    game = set_position(Tafl(unarmed), {(0, 1): DEFENDER, (2, 2): KING, (5, 5): ATTACKER, (6, 4): ATTACKER}, 1)
    game.make_move((2, 2, 0, 2))
    assert game.board[0, 1] == DEFENDER, "king captured its own defender"
    # King moves next to an attacker that stands next to a corner
    game = set_position(Tafl(unarmed), {(0, 1): ATTACKER, (2, 2): KING, (5, 5): ATTACKER, (6, 4): DEFENDER}, 1)
    game.make_move((2, 2, 0, 2))
    assert game.board[0, 1] == ATTACKER, "king captured against a corner"
    # A defender moves so that an attacker is between it and the king
    for rules, expect_captured in [(unarmed, False), (SIMPLE_7X7, True)]:
        game = set_position(Tafl(rules), {(0, 2): KING, (0, 3): ATTACKER, (2, 4): DEFENDER, (5, 5): ATTACKER}, 1)
        game.make_move((2, 4, 0, 4))
        assert (game.board[0, 3] == EMPTY) == expect_captured, f"king_armed={rules.king_armed}: wrong capture"
    # 9x9: a defender stepping next to its own king (attacker on the other side) must not end the game
    game = set_position(Tafl(replace(RULESETS["sami_tablut"], king_armed=False)),
                        {(4, 4): KING, (4, 5): ATTACKER, (6, 3): DEFENDER, (2, 4): ATTACKER}, 1)
    game.make_move((6, 3, 4, 3))
    assert not game.game_over, "defender captured its own king"
    print("  unarmed king: does not capture")
    return True


def test_rollout_value_sign():
    """The defender's only legal move wins, so a rollout from its turn must return +1."""
    game = set_position(Tafl("brandubh"), {(0, 1): KING, (0, 2): ATTACKER, (1, 1): ATTACKER}, 1)
    assert game.get_legal_moves() == [(0, 1, 0, 0)]
    assert RandomRolloutMCTS()._rollout(game) == 1.0
    print("  random rollout: value for the player to move")
    return True


def test_load_compatible_keeps_player_plane():
    """Loading a 13-plane checkpoint into a 4-plane network must keep the player-plane weights."""
    big, small = small_network(7, history_length=4), small_network(7)
    state_dict = big.state_dict()
    small.load_state_dict_compatible(state_dict)
    old, new = state_dict['stem.0.weight'], small.state_dict()['stem.0.weight']
    assert torch.equal(new[:, :3], old[:, :3]) and torch.equal(new[:, 3], old[:, 12])
    assert state_dict['stem.0.weight'].shape[1] == 13, "caller's state dict was modified"
    print("  load_state_dict_compatible: player plane kept")
    return True


def test_sim_balancer_band():
    """The simulation balancer with a win-rate band, and its special case 0.5 (the earlier rule)."""
    # min_win_rate 0.5 and no draws: same counts as the earlier rule
    def earlier_rule(attacker_sims, total, rate, attacker_wins, defender_wins):
        attacker_rate = attacker_wins / (attacker_wins + defender_wins)
        transfer = rate * abs((1 - attacker_rate) - attacker_rate) * total / 2
        attacker_sims += transfer if attacker_rate < 0.5 else -transfer
        return int(round(max(1, min(total - 1, attacker_sims))))

    rng = np.random.default_rng(4)
    balancer, expected = SimulationBalancer(1000, 1000, adjustment_rate=0.2), 1000
    for _ in range(50):
        attacker_wins = int(rng.integers(0, 513))
        balancer.update(attacker_wins, 512 - attacker_wins)
        expected = earlier_rule(expected, 2000, 0.2, attacker_wins, 512 - attacker_wins)
        assert (balancer.attacker_sims, balancer.defender_sims) == (expected, 2000 - expected)

    # Band [0.2, 0.8] with adjustment rate 0.1 and 2000 simulations in total
    balancer = SimulationBalancer(1000, 1000, adjustment_rate=0.1, min_win_rate=0.2)
    steps = [((90, 10, 0), 980),   # score 0.9: 0.1 * (0.9 - 0.8) * 2000 = 20 simulations to the defender
             ((70, 0, 30), 970),   # draws count half: score 0.85, 10 more to the defender
             ((50, 50, 0), 973),   # inside the band: 10 % of the way back to 1000
             ((10, 90, 0), 993)]   # score 0.1: 20 simulations to the attacker
    for (attacker_wins, defender_wins, draws), attacker_sims in steps:
        balancer.update(attacker_wins, defender_wins, draws)
        assert (balancer.attacker_sims, balancer.defender_sims) == (attacker_sims, 2000 - attacker_sims)

    # Settling in a toy game whose attacker score rises with the attacker's share f of the simulations:
    # score = 1 / (1 + exp(-(bias + log(f / (1 - f))))), so at an equal split the score is 1 / (1 + exp(-bias))
    for bias, min_win_rate, start, target in [(np.log(9), 0.2, 1000, 'edge'),   # 90 % at equal split
                                               (np.log(7 / 3), 0.2, 300, 'equal'),  # 70 %: no handicap needed
                                               (np.log(9), 0.5, 1000, 'half')]:   # earlier rule: towards 50 %
        balancer, scores = SimulationBalancer(start, 2000 - start, adjustment_rate=0.2, min_win_rate=min_win_rate), []
        for _ in range(300):
            f = balancer.attacker_sims / 2000
            scores.append(1 / (1 + np.exp(-(bias + np.log(f / (1 - f))))))
            attacker_wins = int(round(scores[-1] * 10000))
            balancer.update(attacker_wins, 10000 - attacker_wins)
        late_score = np.mean(scores[-50:])
        if target == 'edge':
            assert 0.8 <= late_score < 0.83, f"score {late_score:.3f} should stay just above the band edge 0.8"
        elif target == 'equal':
            assert abs(balancer.attacker_sims - 1000) <= 3, f"split {balancer.attacker_sims} should return to equal"
        else:
            assert abs(late_score - 0.5) < 0.01, f"score {late_score:.3f} should settle at 0.5"
    print("  simulation balancer: earlier rule kept at 0.5; band, drift back and draws work")
    return True


def test_replay_buffer_symmetries():
    """The symmetry tables used when a batch is drawn must give the boards and policies of
    augment_sample, and the ring buffer must keep the newest positions."""
    rng = np.random.default_rng(5)
    for rules in (RULESETS["brandubh"], RULESETS["sami_tablut"]):
        n = rules.board_size
        cell_perm, move_perm = symmetry_permutations(n)
        for _ in range(5):
            state = random_position(rules, rng).get_state()
            policy = rng.random(policy_size(n)).astype(np.float32)
            for k, (aug_state, aug_policy, _) in enumerate(augment_sample(state, policy, 0, n)):
                turned = state.reshape(4, -1)[:, cell_perm[k]].reshape(state.shape)
                assert np.array_equal(turned, aug_state)
                assert np.array_equal(policy[move_perm[k]], aug_policy)
    buffer = ReplayBuffer(5 * 8, (4, 7, 7), 1176, num_symmetries=8)  # room for 5 positions
    for values in (np.arange(3), np.arange(3, 7)):
        n = len(values)
        buffer.add(np.zeros((n, 4, 7, 7)), np.zeros((n, 1176)), values, np.zeros(n), np.zeros(n),
                   values)
    assert len(buffer) == 40 and sorted(buffer.values) == [2, 3, 4, 5, 6] and buffer.next_index == 2
    print("  replay buffer: symmetry tables equal augment_sample; ring buffer keeps the newest")
    return True


def test_train_network_small_buffers():
    """A too small buffer must give a clear error; small ones must train, with an empty validation
    set allowed (division by zero before). Draws and unequal loss boosts must work."""
    config = TrainingConfig()
    config.batch_size, config.batches_per_epoch, config.num_epochs = 4, 2, 1
    game = Tafl("brandubh")
    policy = np.zeros(1176, dtype=np.float32)
    policy[move_index(game.get_legal_moves()[0], 7)] = 1.0
    original_compile = torch.compile
    torch.compile = lambda model, **kwargs: model  # no compilation in this test
    try:
        cases = [(4, 1, False, None), (8, 1, True, False), (6, 8, True, True)]
        for num_positions, num_symmetries, trains, val_is_empty in cases:
            buffer = ReplayBuffer(64, (4, 7, 7), 1176, num_symmetries)
            buffer.add(np.repeat(game.get_state()[None], num_positions, axis=0),
                       np.repeat(policy[None], num_positions, axis=0), np.ones(num_positions),
                       np.arange(num_positions) % 3 - 1,  # draws, attacker wins and defender wins
                       np.array([0, 3, 4, 6])[np.arange(num_positions) % 4],  # allowed endings
                       np.arange(num_positions) * 10.0)  # plies left
            network = small_network(7)
            optimizer = torch.optim.AdamW(network.parameters())
            try:
                losses = train_network(network, buffer, optimizer, config,
                                       attacker_boost=2.0, defender_boost=0.5)
            except ValueError:
                assert not trains, f"{num_positions} positions: training should have run"
                continue
            assert trains, f"{num_positions} positions: a too small buffer must raise ValueError"
            assert all(np.isfinite(losses[key])
                       for key in ('policy_loss', 'ending_loss', 'moves_left_loss'))
            assert np.isnan(losses['val_policy_loss']) == val_is_empty
            # The per-epoch table: the same keys, one value per epoch
            assert set(losses) == set(LOSS_KEYS) and list(losses['epochs']) == LOSS_KEYS[:-1]
            assert all(len(values) == config.num_epochs for values in losses['epochs'].values())
    finally:
        torch.compile = original_compile
    print("  train_network: clear error for a too small buffer; small buffers and draws train")
    return True


def test_self_play_targets():
    """Self-play must give every position of a game the game's ending and the number of plies from
    the position to the end, and return a game record with the final board."""
    config = TrainingConfig()
    config.num_games_per_iteration, config.num_workers, config.temp_dir = 1, 1, None
    config.num_mcts_sims_attacker = config.num_mcts_sims_defender = 2
    config.network = NetworkConfig(channels=8, blocks=["res"])
    original_compile = torch.compile
    torch.compile = lambda model, **kwargs: model  # no compilation in this test
    try:
        for max_game_length in (6, 300):  # a draw by the move limit, then (usually) a real ending
            config.max_game_length = max_game_length
            positions, games, stats = generate_self_play_data(Agent(TaflNet(config.network, 7)), config)
            assert stats['unique_states'] is None and stats['mcts_timing']['network_eval'] > 0
            (states, policies, values, winners, endings, moves_left), record = positions, games[0]
            assert np.all(endings == ENDINGS.index(record['ending']))
            # The stored positions follow each other ply by ply (random opening plies come first and
            # are not stored), up to one ply before the end
            assert moves_left[-1] == 1 and np.all(np.diff(moves_left) == -1)
            assert len(record['board']) == 7 and record['type'] == 'selfplay'
            assert record['ending'] != 'move_limit' or record['plies'] > max_game_length
    finally:
        torch.compile = original_compile
    print("  self-play: ending and moves-left targets, game record")
    return True


def test_low_temperature_and_eval_mode():
    """A very low temperature must not round all probabilities to zero (NaN before), and a search
    must put the network into eval mode (a fresh network used batch statistics and changed its
    BatchNorm)."""
    game = Tafl("brandubh")
    moves = game.get_legal_moves()[:3]
    agent = Agent(small_network(7), num_simulations=2)
    agent.mcts.search = lambda g, **kwargs: dict(zip(moves, [0.3, 0.3, 0.4]))
    assert agent.select_move_with_stats(game, temperature=0.001)[0] == moves[2]
    assert agent.mcts.select_move(game, temperature=0.001) == moves[2]
    node = MCTSNode(game.clone())
    node.expand(np.ones(len(game.get_legal_moves())))
    for child, visits in zip(node.children.values(), [300, 300, 400]):
        child.visit_count = visits
    assert np.isclose(sum(node.get_visit_distribution(temperature=0.001).values()), 1.0)

    network = small_network(7)  # a new network is in training mode
    batches_tracked = int(network.stem[1].num_batches_tracked)
    Agent(network, num_simulations=3).select_move(game)
    assert not network.training and int(network.stem[1].num_batches_tracked) == batches_tracked
    print("  temperature 0.001 gives finite probabilities; search uses eval mode")
    return True


def test_larger_board_networks():
    """9x9 and 11x11 networks with history planes must search their games, and the games' piece
    planes must equal the first planes of their states."""
    for rules in (RULESETS["sami_tablut"], RULESETS["copenhagen"]):
        n = rules.board_size
        network = small_network(n, history_length=2).eval()
        game = Tafl(rules)
        assert np.array_equal(game.get_piece_planes(), game.get_state()[:3])
        state = StateHistory.from_game(game, 2).get_state_with_history(game.current_player)
        policy, value = network(torch.from_numpy(state)[None])[:2]
        assert state.shape == (7, n, n) and policy.shape == (1, policy_size(n))
        visit_probs = MCTS(network, num_simulations=3, history_length=2).search(game)
        assert np.isclose(sum(visit_probs.values()), 1.0)
    print("  9x9 and 11x11 networks with history planes search their games")
    return True


def test_legal_move_cache():
    """Cached legal moves must equal freshly generated ones: after moves, in clones, after edits."""
    rng = np.random.default_rng(6)
    for rules in (RULESETS["brandubh"], RULESETS["sami_tablut"]):
        for _ in range(10):
            game = Tafl(rules)
            for _ in range(40):
                moves = game.get_legal_moves()
                fresh = game.clone()
                fresh._legal_moves_key = None  # forget the cache
                assert moves == fresh.get_legal_moves() == game.get_legal_moves()
                no_cache = game.clone()
                no_cache._legal_moves_key = None
                assert no_cache.get_legal_moves(stop_at_first=True) == moves[:1]
                game.make_move(moves[rng.integers(len(moves))])
                if game.game_over:
                    break
        # Editing the board directly (as set_position does) must not return the moves of the old
        # board
        game = Tafl(rules)
        game.get_legal_moves()
        game.board[:] = EMPTY
        assert game.get_legal_moves() == []
    print("  legal-move cache: equals fresh generation after moves, in clones and after edits")
    return True


def test_checkpoint_save_and_resume_history():
    """Checkpoints must be written via a temporary file, hold all settings (also class defaults and
    the complete rules), and give back the training history saved with them (used on resume)."""
    import os
    import tempfile
    config = TrainingConfig()
    config.checkpoint_dir = tempfile.mkdtemp()
    network = small_network(7)
    history = {'iterations': [1, 2], 'sim_counts': [[900, 1100], [880, 1120]]}
    optimizer = torch.optim.AdamW(network.parameters())
    save_checkpoint(network, optimizer, 2, config, history, 'checkpoint_iter_2.pth')
    path = os.path.join(config.checkpoint_dir, 'checkpoint_iter_2.pth')
    assert os.listdir(config.checkpoint_dir) == ['checkpoint_iter_2.pth'], "temporary file left"
    saved = torch.load(path, weights_only=False)
    assert saved['config']['reward_discount'] == TrainingConfig.reward_discount, "default not saved"
    assert saved['config']['rules'] == RULESETS[config.ruleset].to_dict(), "rules not saved"
    network = small_network(7)
    assert load_checkpoint(path, network, torch.optim.AdamW(network.parameters())) == (2, history)
    print("  checkpoints: atomic write, all settings saved, training history returned for resume")
    return True


def test_gui_reads_checkpoint_settings():
    """The GUI must take the rule set, search settings and move limit from a 9x9 checkpoint, show
    the same raw policy and value as the search uses, and play moves."""
    import os
    import tempfile
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')  # no window needed
    from gui_play import TaflGUI
    config = TrainingConfig()
    config.checkpoint_dir = tempfile.mkdtemp()
    config.ruleset, config.max_game_length = "sami_tablut", 3
    # Constant planes, so a GUI that builds only the basic planes would fail
    config.network = NetworkConfig(channels=8, blocks=["line"],
                                   features=FeatureSpec(constant_planes=True))
    config.eval_mcts_sims_attacker = 5
    config.eval_c_puct, config.search_discount = 1.5, 0.99
    network = TaflNet(config.network, 9)
    save_checkpoint(network, torch.optim.AdamW(network.parameters()), 1, config, {}, 'tablut.pth')
    gui = TaflGUI(os.path.join(config.checkpoint_dir, 'tablut.pth'))
    assert gui.rules.name == "sami_tablut" and gui.game.current_player == 1
    assert gui.max_moves == 3
    assert (gui.simulations, gui.c_puct, gui.ai_mcts.search_discount) == (5, 1.5, 0.99)
    moves = gui.game.get_legal_moves()
    probs, value = gui.analysis_mcts._evaluate(gui.game, moves)
    assert np.allclose([gui.move_probs[m] for m in moves], probs, atol=1e-6), "policy differs"
    assert abs(gui.value - value) < 1e-6, "value differs"
    # A clicked move, then AI moves until the move limit ends the game in a draw
    gui._handle_click(gui._square_rect(moves[0][:2]).center, 1)
    gui._handle_click(gui._square_rect(moves[0][2:]).center, 1)
    while True:
        while gui.pending_move is not None:
            gui._update_animation()
        if gui.game.game_over:
            break
        gui._do('ai')
    assert gui.game.move_count == 4 and gui.game.winner is None, "move limit not applied"
    gui._draw_board()
    gui._draw_panel()
    print("  GUI: rule set and search settings from the checkpoint, moves, move limit")
    return True


if __name__ == "__main__":
    print("Testing training-pipeline fixes...")
    tests = [test_augmentation_consistent, test_from_game_matches_tracked_history,
             test_evaluate_matches_masked_softmax, test_play_game_uses_rules, test_fpu_uses_parent_value,
             test_search_input_matches_training_state, test_single_simulation_policy_is_finite,
             test_temperature_and_time_limit, test_play_game_counts_last_ply_win,
             test_unarmed_king_does_not_capture, test_rollout_value_sign,
             test_load_compatible_keeps_player_plane, test_sim_balancer_band, test_replay_buffer_symmetries,
             test_train_network_small_buffers, test_self_play_targets,
             test_low_temperature_and_eval_mode,
             test_larger_board_networks, test_legal_move_cache,
             test_checkpoint_save_and_resume_history, test_gui_reads_checkpoint_settings]
    all_passed = True
    for test in tests:
        try:
            all_passed &= test()
        except Exception as e:  # any error is a failure, not only a failed assert
            print(f"  FAILED {test.__name__}: {type(e).__name__}: {e}")
            all_passed = False
    print("All tests PASSED!" if all_passed else "Some tests FAILED!")
    sys.exit(0 if all_passed else 1)
