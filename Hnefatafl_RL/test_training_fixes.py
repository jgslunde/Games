"""
Tests for the training-pipeline fixes:
1. Symmetry augmentation: every augmented policy target matches its augmented board.
2. StateHistory.from_game rebuilds the same history planes as tracking them move by move.
3. MCTS._evaluate (softmax over legal moves) equals the old masked softmax over all moves.
4. play_game creates the game with the given rules.
5. FPU: an unvisited child is valued at the parent's value, seen from the player to move.
6. History planes: the network input during search has the same layout as the training states.
7. One MCTS simulation gives a finite policy (the network priors).
8. select_move and select_move_with_stats use the temperature value, and a time limit works.
9. play_game counts a win on the last allowed ply as a win.
10. king_can_capture=False: the king neither captures nor helps to capture (Brandubh and Tablut).
11. A capturing move is never refused as a repetition.
12. Random rollouts return the result for the player to move at the start of the rollout.
13. load_state_dict_compatible keeps the player plane when the history length differs.
14. SimulationBalancer: min_win_rate 0.5 reproduces the earlier rule; a band [m, 1-m] only pushes
    outside it, drifts back to an equal split inside it, and counts draws as half a win.
15. The replay buffer's symmetry tables equal augment_sample; the ring buffer keeps the newest data.
16. train_network: clear error for a too small buffer; small buffers, empty validation sets and
    draws work.
17. Temperature 0.001 gives finite probabilities, and a search puts the network into eval mode.
18. Tablut and Hnefatafl networks take value head size and history length, and load old checkpoints.
19. The cached legal moves equal freshly generated ones.
20. Checkpoints are written atomically, hold all settings and return their training history.

Run with: python test_training_fixes.py (exit code 1 if a test fails)
"""

import sys
import numpy as np
import torch

from brandubh import Brandubh, EMPTY, ATTACKER, DEFENDER, KING
from tablut import Tablut
from network import BrandubhNet, MoveEncoder, StateHistory
from network_tablut import TablutMoveEncoder
from mcts import MCTS, MCTSNode, RandomRolloutMCTS
from agent import Agent, RandomAgent, play_game
from train import (augment_sample, symmetry_permutations, ReplayBuffer, SimulationBalancer,
                   TrainingConfig, train_network, save_checkpoint, load_checkpoint)


def random_position(game_class, rng, max_plies=14):
    """Play a random number of random moves (no rules changed) and return a non-finished game."""
    while True:
        game = game_class(throne_enabled=False)
        for _ in range(rng.integers(1, max_plies)):
            moves = game.get_legal_moves()
            game.make_move(moves[rng.integers(len(moves))])
            if game.game_over:
                break
        if not game.game_over:
            return game


def test_augmentation_consistent():
    """Each of the 8 symmetries must put policy mass only on moves that are legal-looking on its board."""
    rng = np.random.default_rng(0)
    for game_class, encoder, size in [(Brandubh, MoveEncoder, 7), (Tablut, TablutMoveEncoder, 9)]:
        for _ in range(20):
            game = random_position(game_class, rng)
            state, player, legal = game.get_state(), game.current_player, game.get_legal_moves()
            policy = np.zeros(size * size * 4 * (size - 1), dtype=np.float32)
            for move in legal:
                policy[encoder.encode_move(move)] = 1.0 / len(legal)
            augmented = augment_sample(state, policy, 0.5, board_size=size)
            assert len(augmented) == 8
            for aug_state, aug_policy, aug_value in augmented:
                assert aug_value == 0.5 and np.isclose(aug_policy.sum(), 1.0)
                for idx in np.nonzero(aug_policy)[0]:
                    fr, fc, tr, tc = encoder.decode_move(idx)
                    # Plane 0 = attackers; planes 1 + 2 = defenders and king
                    own_piece = aug_state[0, fr, fc] if player == 0 else aug_state[1, fr, fc] + aug_state[2, fr, fc]
                    assert own_piece > 0, f"{game_class.__name__}: move {idx} does not start on own piece"
                    assert aug_state[0:3, tr, tc].sum() == 0, f"{game_class.__name__}: move {idx} ends on a piece"
    print("  augmentation: all 8 symmetries consistent for Brandubh and Tablut")
    return True


def test_from_game_matches_tracked_history():
    """from_game must give the same planes as pushing piece planes after every move (as in self-play)."""
    rng = np.random.default_rng(1)
    for history_length in (1, 4):
        game = Brandubh(throne_enabled=False)
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
    """Softmax over legal moves must equal the old mask-all-moves softmax restricted to legal moves."""
    torch.manual_seed(0)
    rng = np.random.default_rng(2)
    network = BrandubhNet(num_res_blocks=2, num_channels=16).eval()
    mcts = MCTS(network, num_simulations=10, move_encoder_class=MoveEncoder)
    for _ in range(10):
        game = random_position(Brandubh, rng)
        legal = game.get_legal_moves()
        probs, value = mcts._evaluate(game, legal)
        with torch.inference_mode():
            logits, _ = network(torch.from_numpy(game.get_state()).unsqueeze(0))
        logits = logits.numpy()[0]
        mask = MoveEncoder.get_legal_move_mask(game)
        old = MCTS._softmax(logits * mask + (1 - mask) * (-1e8))[[MoveEncoder.encode_move(m) for m in legal]]
        assert np.allclose(probs, old, atol=1e-6) and np.isclose(probs.sum(), 1.0)
    print("  _evaluate: legal-move softmax equals masked softmax")
    return True


def test_play_game_uses_rules():
    """play_game must pass the rules to the game class (evaluation used to run with default rules)."""
    created = []

    class RecordingBrandubh(Brandubh):
        def __init__(self, **kwargs):
            created.append(kwargs)
            super().__init__(**kwargs)

    rules = dict(king_capture_pieces=2, king_can_capture=True, throne_is_hostile=False, throne_enabled=False)
    play_game(RandomAgent(), RandomAgent(), RecordingBrandubh, display=False, rules=rules, max_moves=20)
    assert created == [rules]
    print("  play_game: game created with the given rules")
    return True


def set_position(game, pieces: dict, player: int):
    """Put `pieces` ({(row, col): piece}) on an empty board with `player` to move; returns the game."""
    game.board[:] = EMPTY
    for (r, c), piece in pieces.items():
        game.board[r, c] = piece
    game.current_player = player
    game.position_history = [game._get_position_hash()]
    return game


def test_fpu_uses_parent_value():
    """An unvisited child must be valued at parent value + fpu_reduction (the sign used to be flipped)."""
    game = Brandubh(throne_enabled=False)
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
    game = Brandubh(throne_enabled=False)
    history = StateHistory(history_length)
    history.push(game.get_piece_planes())
    for _ in range(3):  # track the history as self-play does
        game.make_move(game.get_legal_moves()[rng.integers(5)])
        history.push(game.get_piece_planes())
    train_state = history.get_state_with_history(game.current_player)

    for state_history in (history, None):  # self-play path, and the from_game path (Agent, play.py)
        net = RecordingNet(BrandubhNet(num_res_blocks=1, num_channels=8, history_length=history_length).eval())
        MCTS(net, num_simulations=4, history_length=history_length, move_encoder_class=MoveEncoder).search(
            game, state_history=state_history)
        assert np.array_equal(net.inputs[0], train_state), "root input differs from the training state"
        # First child: its own planes, then the root's planes, then the one before
        assert np.array_equal(net.inputs[1][3:9], train_state[0:6]), "child input has the wrong history"
        assert not np.array_equal(net.inputs[1][0:3], net.inputs[1][3:6]), "child position appears twice"
    print("  history planes: search input matches the training layout")
    return True


def test_single_simulation_policy_is_finite():
    """With one simulation no child is visited; the visit distribution must fall back to the priors."""
    mcts = MCTS(BrandubhNet(num_res_blocks=1, num_channels=8).eval(), num_simulations=1, move_encoder_class=MoveEncoder)
    probs = np.array(list(mcts.search(Brandubh(throne_enabled=False)).values()))
    assert np.all(np.isfinite(probs)) and np.isclose(probs.sum(), 1.0)
    print("  one simulation: finite policy")
    return True


def test_temperature_and_time_limit():
    """A low temperature must almost always pick the most visited move (it used to act like 1.0)."""
    game = Brandubh(throne_enabled=False)
    a, b = game.get_legal_moves()[:2]
    agent = Agent(BrandubhNet(num_res_blocks=1, num_channels=8).eval(), num_simulations=2, move_encoder_class=MoveEncoder)
    agent.mcts.search = lambda g, state_history=None, time_limit=None: {a: 0.9, b: 0.1}
    np.random.seed(0)
    assert sum(agent.mcts.select_move(game, temperature=0.1) == b for _ in range(2000)) < 20
    assert sum(agent.select_move_with_stats(game, temperature=0.1)[0] == b for _ in range(2000)) < 20

    # Time-limited search (used by play.py --time-per-move). One second, so that even a fully loaded
    # node runs more than num_simulations simulations
    agent = Agent(BrandubhNet(num_res_blocks=1, num_channels=8).eval(), num_simulations=2, move_encoder_class=MoveEncoder)
    move, _, _ = agent.select_move_with_stats(game, time_limit=1.0)
    assert move in game.get_legal_moves() and agent.mcts.root.visit_count > 2
    print("  temperature: low temperature picks the most visited move; time limit works")
    return True


def test_play_game_counts_last_ply_win():
    """A king capture on the last allowed ply must count as an attacker win, not as a draw."""

    class Position(Brandubh):
        def __init__(self, **rules):
            super().__init__(**rules)
            set_position(self, {(3, 3): KING, (3, 1): ATTACKER, (1, 4): ATTACKER, (5, 0): ATTACKER,
                                (6, 5): DEFENDER}, player=0)

    class Scripted:
        def __init__(self, moves):
            self.moves = list(moves)

        def select_move(self, game, temperature=0.0):
            return self.moves.pop(0)

    # Ply 3 (the first ply beyond max_moves=2) sandwiches the king between (3,2) and (3,4)
    attackers, defenders = Scripted([(3, 1, 3, 2), (1, 4, 3, 4)]), Scripted([(6, 5, 6, 4)])
    winner = play_game(attackers, defenders, Position, display=False, rules=dict(throne_enabled=False), max_moves=2)
    assert winner == 0, f"expected attacker win, got {winner}"
    print("  play_game: a win on the last allowed ply counts")
    return True


def test_king_cannot_capture():
    """With king_can_capture=False the king must not capture, also not with a corner or as second piece."""
    rules = dict(king_can_capture=False, throne_enabled=False)
    # King moves next to its own defender, which stands next to a corner
    game = set_position(Brandubh(**rules), {(0, 1): DEFENDER, (2, 2): KING, (5, 5): ATTACKER, (6, 4): ATTACKER}, 1)
    game.make_move((2, 2, 0, 2))
    assert game.board[0, 1] == DEFENDER, "king captured its own defender"
    # King moves next to an attacker that stands next to a corner
    game = set_position(Brandubh(**rules), {(0, 1): ATTACKER, (2, 2): KING, (5, 5): ATTACKER, (6, 4): DEFENDER}, 1)
    game.make_move((2, 2, 0, 2))
    assert game.board[0, 1] == ATTACKER, "king captured against a corner"
    # A defender moves so that an attacker is between it and the king
    for king_can_capture, expect_captured in [(False, False), (True, True)]:
        game = set_position(Brandubh(king_can_capture=king_can_capture, throne_enabled=False),
                            {(0, 2): KING, (0, 3): ATTACKER, (2, 4): DEFENDER, (5, 5): ATTACKER}, 1)
        game.make_move((2, 4, 0, 4))
        assert (game.board[0, 3] == EMPTY) == expect_captured, f"king_can_capture={king_can_capture}: wrong capture"
    # Tablut: a defender stepping next to its own king (attacker on the other side) must not end the game
    game = set_position(Tablut(**rules), {(4, 4): KING, (4, 5): ATTACKER, (6, 3): DEFENDER, (0, 4): ATTACKER}, 1)
    game.make_move((6, 3, 4, 3))
    assert not game.game_over, "Tablut: defender captured its own king"
    print("  king_can_capture=False: king does not capture (Brandubh, Tablut)")
    return True


def test_capture_is_never_repetition():
    """A capturing move must stay legal even if its board before the capture occurred twice before."""
    pieces = {(0, 2): ATTACKER, (2, 3): DEFENDER, (2, 4): ATTACKER, (5, 5): KING, (6, 1): ATTACKER}
    for move, captures in [((0, 2, 2, 2), True), ((2, 4, 1, 4), False)]:
        # A new game for each case: the legal-move cache assumes the history only grows through
        # make_move
        game = set_position(Brandubh(throne_enabled=False), pieces, 0)
        board = game.board.copy()
        board[move[2], move[3]], board[move[0], move[1]] = board[move[0], move[1]], EMPTY
        game.position_history = [game._get_position_hash()] + 2 * [(board.tobytes(), 1)]
        assert (move in game.get_legal_moves()) == captures, f"move {move}: wrong repetition decision"
    print("  repetition: capturing moves are never repetitions")
    return True


def test_rollout_value_sign():
    """The defender's only legal move wins, so a rollout from its turn must return +1."""
    game = set_position(Brandubh(throne_enabled=False), {(0, 1): KING, (0, 2): ATTACKER, (1, 1): ATTACKER}, 1)
    assert game.get_legal_moves() == [(0, 1, 0, 0)]
    assert RandomRolloutMCTS()._rollout(game) == 1.0
    print("  random rollout: value for the player to move")
    return True


def test_load_compatible_keeps_player_plane():
    """Loading a 13-plane checkpoint into a 4-plane network must keep the player-plane weights."""
    big, small = BrandubhNet(1, 8, history_length=4), BrandubhNet(1, 8, history_length=1)
    state_dict = big.state_dict()
    small.load_state_dict_compatible(state_dict)
    old, new = state_dict['conv_input.weight'], small.state_dict()['conv_input.weight']
    assert torch.equal(new[:, :3], old[:, :3]) and torch.equal(new[:, 3], old[:, 12])
    assert state_dict['conv_input.weight'].shape[1] == 13, "caller's state dict was modified"
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
    for game_class, size in [(Brandubh, 7), (Tablut, 9)]:
        cell_perm, move_perm = symmetry_permutations(size)
        for _ in range(5):
            state = random_position(game_class, rng).get_state()
            policy = rng.random(size * size * 4 * (size - 1)).astype(np.float32)
            for k, (aug_state, aug_policy, _) in enumerate(augment_sample(state, policy, 0, size)):
                turned = state.reshape(4, -1)[:, cell_perm[k]].reshape(state.shape)
                assert np.array_equal(turned, aug_state)
                assert np.array_equal(policy[move_perm[k]], aug_policy)
    buffer = ReplayBuffer(5 * 8, (4, 7, 7), 1176, num_symmetries=8)  # room for 5 positions
    for values in (np.arange(3), np.arange(3, 7)):
        n = len(values)
        buffer.add(np.zeros((n, 4, 7, 7)), np.zeros((n, 1176)), values, np.zeros(n))
    assert len(buffer) == 40 and sorted(buffer.values) == [2, 3, 4, 5, 6] and buffer.next_index == 2
    print("  replay buffer: symmetry tables equal augment_sample; ring buffer keeps the newest")
    return True


def test_train_network_small_buffers():
    """A too small buffer must give a clear error; small ones must train, with an empty validation
    set allowed (division by zero before). Draws and unequal loss boosts must work."""
    config = TrainingConfig()
    config.batch_size, config.batches_per_epoch, config.num_epochs = 4, 2, 1
    game = Brandubh()
    policy = np.zeros(1176, dtype=np.float32)
    policy[MoveEncoder.encode_move(game.get_legal_moves()[0])] = 1.0
    original_compile = torch.compile
    torch.compile = lambda model, **kwargs: model  # no compilation in this test
    try:
        cases = [(4, 1, False, None), (8, 1, True, False), (6, 8, True, True)]
        for num_positions, num_symmetries, trains, val_is_empty in cases:
            buffer = ReplayBuffer(64, (4, 7, 7), 1176, num_symmetries)
            buffer.add(np.repeat(game.get_state()[None], num_positions, axis=0),
                       np.repeat(policy[None], num_positions, axis=0), np.ones(num_positions),
                       np.arange(num_positions) % 3 - 1)  # draws, attacker wins and defender wins
            network = BrandubhNet(1, 8)
            optimizer = torch.optim.AdamW(network.parameters())
            try:
                losses = train_network(network, buffer, optimizer, config,
                                       attacker_boost=2.0, defender_boost=0.5)
            except ValueError:
                assert not trains, f"{num_positions} positions: training should have run"
                continue
            assert trains, f"{num_positions} positions: a too small buffer must raise ValueError"
            assert np.isfinite(losses['policy_loss'])
            assert np.isnan(losses['val_policy_loss']) == val_is_empty
    finally:
        torch.compile = original_compile
    print("  train_network: clear error for a too small buffer; small buffers and draws train")
    return True


def test_low_temperature_and_eval_mode():
    """A very low temperature must not round all probabilities to zero (NaN before), and a search
    must put the network into eval mode (a fresh network used batch statistics and changed its
    BatchNorm)."""
    game = Brandubh(throne_enabled=False)
    moves = game.get_legal_moves()[:3]
    agent = Agent(BrandubhNet(1, 8), num_simulations=2, move_encoder_class=MoveEncoder)
    agent.mcts.search = lambda g, **kwargs: dict(zip(moves, [0.3, 0.3, 0.4]))
    assert agent.select_move_with_stats(game, temperature=0.001)[0] == moves[2]
    assert agent.mcts.select_move(game, temperature=0.001) == moves[2]
    node = MCTSNode(game.clone())
    node.expand(np.ones(len(game.get_legal_moves())))
    for child, visits in zip(node.children.values(), [300, 300, 400]):
        child.visit_count = visits
    assert np.isclose(sum(node.get_visit_distribution(temperature=0.001).values()), 1.0)

    network = BrandubhNet(1, 8)  # a new network is in training mode
    batches_tracked = int(network.bn_input.num_batches_tracked)
    Agent(network, num_simulations=3, move_encoder_class=MoveEncoder).select_move(game)
    assert not network.training and int(network.bn_input.num_batches_tracked) == batches_tracked
    print("  temperature 0.001 gives finite probabilities; search uses eval mode")
    return True


def test_larger_board_networks():
    """Tablut and Hnefatafl networks must take the training script's arguments (value head size,
    history length) and load old 4-plane checkpoints, and their games must give piece planes."""
    from network_tablut import TablutNet
    from network_hnefatafl import HnefataflNet, HnefataflMoveEncoder
    from hnefatafl import Hnefatafl
    cases = [(TablutNet, Tablut, TablutMoveEncoder, 9),
             (HnefataflNet, Hnefatafl, HnefataflMoveEncoder, 11)]
    for network_class, game_class, encoder, size in cases:
        network = network_class(num_res_blocks=1, num_channels=8, value_head_hidden_size=16,
                                history_length=2)
        # A checkpoint with 4 input planes (as the existing ones), loaded into a 7-plane network
        four_planes = network_class(1, 8, value_head_hidden_size=16)
        network.load_state_dict_compatible(four_planes.state_dict())
        game = game_class(throne_enabled=False)
        assert np.array_equal(game.get_piece_planes(), game.get_state()[:3])
        state = StateHistory.from_game(game, 2).get_state_with_history(game.current_player)
        policy, value = network.eval()(torch.from_numpy(state)[None])
        assert state.shape == (7, size, size) and policy.shape == (1, size * size * 4 * (size - 1))
        mcts = MCTS(network, num_simulations=3, move_encoder_class=encoder, history_length=2)
        visit_probs = mcts.search(game)
        assert np.isclose(sum(visit_probs.values()), 1.0)
    print("  Tablut/Hnefatafl networks: value head size, history length and old checkpoints work")
    return True


def test_legal_move_cache():
    """Cached legal moves must equal freshly generated ones: after moves, in clones, after edits."""
    rng = np.random.default_rng(6)
    for game_class in (Brandubh, Tablut):
        for _ in range(10):
            game = game_class(throne_enabled=False)
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
        game = game_class(throne_enabled=False)
        game.get_legal_moves()
        game.board[game.board == ATTACKER] = EMPTY
        assert game.get_legal_moves() == []
    print("  legal-move cache: equals fresh generation after moves, in clones and after edits")
    return True


def test_checkpoint_save_and_resume_history():
    """Checkpoints must be written via a temporary file, hold all settings (also class defaults),
    and give back the training history saved with them (used on resume)."""
    import os
    import tempfile
    config = TrainingConfig()
    config.checkpoint_dir = tempfile.mkdtemp()
    network = BrandubhNet(1, 8)
    history = {'iterations': [1, 2], 'sim_counts': [[900, 1100], [880, 1120]]}
    optimizer = torch.optim.AdamW(network.parameters())
    save_checkpoint(network, optimizer, 2, config, history, 'checkpoint_iter_2.pth')
    path = os.path.join(config.checkpoint_dir, 'checkpoint_iter_2.pth')
    assert os.listdir(config.checkpoint_dir) == ['checkpoint_iter_2.pth'], "temporary file left"
    saved = torch.load(path, weights_only=False)
    assert saved['config']['reward_discount'] == TrainingConfig.reward_discount, "default not saved"
    network = BrandubhNet(1, 8)
    assert load_checkpoint(path, network, torch.optim.AdamW(network.parameters())) == (2, history)
    print("  checkpoints: atomic write, all settings saved, training history returned for resume")
    return True


def test_gui_reads_checkpoint_settings():
    """The GUI must take the variant, rules, search settings and move limit from a Tablut checkpoint
    (no --game needed), show the same raw policy and value as the search uses, and play moves."""
    import os
    import tempfile
    os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')  # no window needed
    from network_tablut import TablutNet
    from gui_play import TaflGUI
    from network_config import NetworkConfig
    config = TrainingConfig()
    config.checkpoint_dir = tempfile.mkdtemp()
    config.game, config.throne_enabled, config.max_game_length = "tablut", False, 3
    config.network = NetworkConfig(arch="legacy", channels=8, blocks=1, policy_head="dense", value_head="dense")
    config.eval_mcts_sims_attacker = 5
    config.eval_c_puct, config.search_discount = 1.5, 0.99
    network = TablutNet(1, 8)
    save_checkpoint(network, torch.optim.AdamW(network.parameters()), 1, config, {}, 'tablut.pth')
    gui = TaflGUI(os.path.join(config.checkpoint_dir, 'tablut.pth'))
    assert gui.game_class is Tablut and not gui.game.throne_enabled and gui.max_moves == 3
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
    print("  GUI: Tablut variant, rules and search settings from the checkpoint, moves, move limit")
    return True


if __name__ == "__main__":
    print("Testing training-pipeline fixes...")
    tests = [test_augmentation_consistent, test_from_game_matches_tracked_history,
             test_evaluate_matches_masked_softmax, test_play_game_uses_rules, test_fpu_uses_parent_value,
             test_search_input_matches_training_state, test_single_simulation_policy_is_finite,
             test_temperature_and_time_limit, test_play_game_counts_last_ply_win, test_king_cannot_capture,
             test_capture_is_never_repetition, test_rollout_value_sign, test_load_compatible_keeps_player_plane,
             test_sim_balancer_band, test_replay_buffer_symmetries,
             test_train_network_small_buffers, test_low_temperature_and_eval_mode,
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
