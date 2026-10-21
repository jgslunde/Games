"""
Tests of the configurable networks (tafl_network.py), the input planes (features.py), the config
system (run_config.py, network_config.py) and a short end-to-end training run.

Run with: python test_tafl_network.py
"""

import json
import os
import tempfile

import numpy as np
import torch

from dataclasses import replace

from convert_to_onnx import OUTPUT_NAMES, export_to_onnx
from features import FeatureSpec, build_features, constant_planes, defender_region_plane, reach_planes
from network_config import NetworkConfig
from run_config import TrainingConfig, apply_overrides, from_dict, load_config
from tafl import ATTACKER, DEFENDER, ENDINGS, KING, RULESETS, Tafl, move_index, policy_size
from tafl_network import (BilinearPolicyHead, ConvPolicyHead, TaflNet, count_parameters, load_network,
                          load_weights, move_index_tables)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def random_position(rules, rng, max_plies=40):
    """A position reached by random play (or the start position)."""
    game = Tafl(rules)
    for _ in range(rng.integers(0, max_plies + 1)):
        moves = game.get_legal_moves()
        if game.game_over or not moves:
            break
        game.make_move(moves[rng.integers(len(moves))])
    return game


def test_reach_planes_match_move_generator():
    """The vectorised reach planes must equal the game's own move generator for both sides, in
    every rule set (corners or not, throne or not), and the defender-region plane must equal the
    encirclement check."""
    rng = np.random.default_rng(0)
    rule_sets = list(RULESETS.values()) + [replace(RULESETS["brandubh"], throne=False)]
    for rules in rule_sets:
        n = rules.board_size
        for _ in range(20):
            game = random_position(rules, rng, max_plies=150)
            expected = np.zeros((3, n, n), dtype=np.float32)
            for player in (0, 1):
                # The moves of each side, without the repetition rule
                side = game.clone()
                side.current_player, side._legal_moves_key = player, None
                side.position_history = [side._position_key()]
                for fr, fc, tr, tc in side.get_legal_moves():
                    expected[player, tr, tc] = 1
                    if game.board[fr, fc] == KING:
                        expected[2, tr, tc] = 1
            assert np.array_equal(reach_planes(game), expected), f"reach planes differ ({rules.name})"
    for _ in range(30):
        game = random_position("copenhagen", rng, max_plies=80)
        region = defender_region_plane(game)
        touches_edge = region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()
        assert touches_edge != game._defenders_encircled(), "region plane disagrees with encirclement"
        assert region[game.board >= DEFENDER].all() and not region[game.board == ATTACKER].any()
    # A hand-made ring of attackers around the lone king: the region is the inside of the ring,
    # until a defender stands outside it
    game = Tafl("copenhagen")
    game.board[:] = 0
    game.board[5, 5] = KING
    for r, c in [(3, 5), (7, 5), (5, 3), (5, 7), (4, 4), (4, 6), (6, 4), (6, 6)]:
        game.board[r, c] = ATTACKER
    assert defender_region_plane(game).sum() == 5 and game._defenders_encircled()
    game.board[0, 2] = DEFENDER
    assert defender_region_plane(game).sum() == 121 - 8 and not game._defenders_encircled()  # both areas
    print("  reach planes equal the move generator; defender-region plane equals the encirclement test")
    return True


def test_feature_layout():
    """Plane names, counts and contents of every feature option."""
    game = Tafl("brandubh")
    full = FeatureSpec(history_length=2, constant_planes=True, mobility_planes=True, region_plane=True)
    names = full.plane_names()
    assert names == ["attackers_t0", "defenders_t0", "king_t0", "attackers_t1", "defenders_t1", "king_t1",
                     "player", "ones", "corners", "throne", "reach_attackers", "reach_defenders",
                     "reach_king", "defender_region"]
    state = build_features(game, full)
    assert state.shape == (14, 7, 7) and state.dtype == np.float32
    assert np.array_equal(state[:3], game.get_state()[:3]) and not state[3:6].any()  # no earlier position
    assert np.array_equal(state[6], game.get_state()[3])
    assert state[7].all() and state[8].sum() == 4 and state[8, 0, 0] == 1 and state[9].sum() == 1 and state[9, 3, 3] == 1
    assert np.array_equal(state[10:13], reach_planes(game)) and np.array_equal(state[13], defender_region_plane(game))
    assert set(np.unique(state)) <= {0.0, 1.0}
    no_throne = Tafl(replace(RULESETS["brandubh"], throne=False))
    assert not constant_planes(no_throne)[2].any(), "no throne must give an empty throne plane"
    assert not constant_planes(Tafl("sami_tablut"))[1].any(), "edge goal must give an empty corner plane"
    # The default spec gives the classic 4 planes
    assert np.array_equal(build_features(game, FeatureSpec()), game.get_state())
    print("  feature planes: names, order and contents as documented")
    return True


def test_policy_head_index_order():
    """Both policy heads must put move (from, direction, distance) where tafl.move_index puts it."""
    for ruleset in ("brandubh", "sami_tablut", "copenhagen"):
        game = Tafl(ruleset)
        n = game.size
        head = ConvPolicyHead(8, n)
        with torch.no_grad():
            head.conv.weight.zero_()
            head.conv.bias.copy_(torch.arange(4 * (n - 1), dtype=torch.float32))
        logits = head(torch.zeros(1, 8, n, n))[0]
        to_square, on_board = move_index_tables(n)
        for move in game.get_legal_moves():
            idx = move_index(move, n)
            direction, distance = (idx % (4 * (n - 1))) // (n - 1), idx % (n - 1) + 1
            assert logits[idx].item() == direction * (n - 1) + distance - 1
            assert on_board[idx] and to_square[idx] == move[2] * n + move[3]
            assert idx // (4 * (n - 1)) == move[0] * n + move[1]
        bilinear = BilinearPolicyHead(8, n, dim=4)
        logits = bilinear(torch.randn(2, 8, n, n))
        assert logits.shape == (2, policy_size(n))
        assert (logits[:, ~torch.from_numpy(on_board)] == -1e4).all() and torch.isfinite(logits).all()
    print("  conv and bilinear policy heads use the policy index order")
    return True


def test_line_block_sees_whole_lines():
    """One line block must connect every square to every square of its row and column, also from a
    corner (a kernel only N long would reach just N//2 squares to each side)."""
    from tafl_network import LineBlock
    torch.manual_seed(0)
    for n in (7, 9, 11):
        for kind in ("line", "thick"):
            block = LineBlock(16, n, kind)
            inner = block.row.in_channels
            for conv, square in ((block.row, lambda k: (0, k)), (block.col, lambda k: (k, 0))):
                for k in range(n):
                    x = torch.zeros(1, inner, n, n)
                    x[0, :, square(k)[0], square(k)[1]] = 1.0
                    seen = conv(x)[0, :, 0, 0].abs().sum() > 0
                    assert seen, f"{kind} {n}x{n}: corner misses square {k}"
    print("  line blocks: every square sees its whole row and column in one block")
    return True


def test_attention_block_sees_whole_board():
    """One attention block must connect opposite corners (no convolution block can), and its
    position bias must depend only on the offset between two squares."""
    from tafl_network import AttentionBlock
    torch.manual_seed(0)
    for n in (7, 9, 11):
        block = AttentionBlock(16, n, heads=4).eval()
        with torch.no_grad():
            block.position_bias.normal_()
            x = torch.randn(1, 16, n, n)
            changed = x.clone()
            changed[0, :, n - 1, n - 1] += torch.randn(16)  # not a constant: layer norm removes it
            assert not torch.allclose(block(x)[0, :, 0, 0], block(changed)[0, :, 0, 0]), f"{n}x{n}"
        index = block.offset_index.view(n, n, n, n)
        # Square (1, 2) -> (3, 5) has the same offset as (0, 0) -> (2, 3)
        assert index[1, 2, 3, 5] == index[0, 0, 2, 3] and index[0, 0, 2, 3] != index[0, 0, 3, 2]
    print("  attention blocks: one block connects opposite corners; position bias by offset")
    return True


ARCHITECTURES = [
    dict(channels=16, blocks=["res"] * 2, squeeze_excitation=True, policy_head="dense", value_head="dense"),
    dict(channels=16, blocks=["line"] * 2, policy_head="conv", value_head="pooled",
         features=dict(constant_planes=True)),
    dict(channels=16, blocks=["line3", "res"], global_pool="every_other", policy_head="conv",
         value_head="pooled", features=dict(history_length=3, mobility_planes=True, region_plane=True)),
    dict(channels=16, blocks=["thick"] * 2, global_pool="all", policy_head="bilinear", policy_bilinear_dim=8),
    dict(channels=16, blocks=["line"] * 3, tie_blocks=True, global_pool="all", policy_head="conv"),
    dict(channels=16, blocks=["line", "attention", "line"], attention_heads=4,
         global_pool="every_other", policy_head="conv", value_head="pooled",
         features=dict(constant_planes=True)),
]


def test_architectures_forward_and_onnx():
    """Every block kind, head and feature option must build, train (batch forward/backward), run at
    batch 1 in eval mode, export to ONNX (opset 12, as convert_to_onnx.py) and give the same outputs
    in onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        ort = None
    for n in (7, 9, 11):
        for arch in ARCHITECTURES:
            config = NetworkConfig.from_dict(arch)
            network = TaflNet(config, n)
            planes = config.features.num_planes
            x = (torch.rand(4, planes, n, n) < 0.2).float()
            network.train()
            policy, value, ending, moves_left = network(x)
            assert policy.shape == (4, policy_size(n)) and value.shape == (4, 1)
            assert ending.shape == (4, len(ENDINGS)) and moves_left.shape == (4, 1)
            assert (moves_left >= 0).all()
            (policy.logsumexp(1).mean() + value.mean() + ending.logsumexp(1).mean()
             + moves_left.mean()).backward()
            network.eval()
            with torch.no_grad():
                policy1, value1 = network(x[:1])[:2]
            assert policy1.shape == (1, policy_size(n)) and -1 <= value1.item() <= 1
            assert count_parameters(network) > 0
            if config.tie_blocks:
                assert len(network.blocks) == 1 and len(network.block_order) == 3
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "net.onnx")
                export_to_onnx(network, path, n, planes)
                if ort is not None:
                    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                    onnx_outputs = session.run(OUTPUT_NAMES, {"board_state": x[:2].numpy()})
                    with torch.no_grad():
                        torch_outputs = network(x[:2])
                    for name, onnx_out, torch_out in zip(OUTPUT_NAMES, onnx_outputs, torch_outputs):
                        assert np.allclose(onnx_out, torch_out.numpy(), atol=1e-4, rtol=1e-4), \
                            f"ONNX output {name} differs"
    print("  all block kinds, heads and features: forward, backward, batch-1 eval, ONNX export match")
    return True


def test_checkpoint_loading_and_stem_adaptation():
    """Networks must come back from checkpoints, and weights must transfer to a network with more
    input planes: the shared planes keep their weights. Deprecated checkpoints (no network config)
    must be refused with a clear error."""
    with tempfile.TemporaryDirectory() as tmp:
        config = NetworkConfig(channels=16, blocks=["line"] * 2, features=FeatureSpec(constant_planes=True))
        network = TaflNet(config, 9).eval()
        x = (torch.rand(1, config.features.num_planes, 9, 9) < 0.2).float()
        path = os.path.join(tmp, "new.pth")
        torch.save({"model_state_dict": network.state_dict(), "network_config": config.to_dict(),
                    "config": {"board_size": 9}}, path)
        loaded, checkpoint = load_network(path)
        assert isinstance(loaded, TaflNet) and loaded.board_size == 9
        with torch.no_grad():
            assert torch.equal(loaded(x)[0], network(x)[0])
        # Transfer into a network with extra planes: equal outputs while the extra planes are zero
        bigger = TaflNet(NetworkConfig(channels=16, blocks=["line"] * 2, features=FeatureSpec(
            constant_planes=True, mobility_planes=True)), 9).eval()
        load_weights(bigger, checkpoint)
        x_big = torch.cat([x, torch.zeros(1, 3, 9, 9)], dim=1)
        with torch.no_grad():
            assert torch.allclose(bigger(x_big)[0], network(x)[0], atol=1e-6)
        # A deprecated checkpoint (old flat config without a network config)
        old_path = os.path.join(tmp, "old.pth")
        torch.save({"model_state_dict": network.state_dict(), "config": {"num_channels": 16}}, old_path)
        try:
            load_network(old_path)
            assert False, "a checkpoint without network config was loaded"
        except ValueError as e:
            assert "deprecated" in str(e)
    print("  checkpoints load; weights transfer to networks with more input planes; old ones refused")
    return True


def test_run_configs():
    """All shipped configs must load, round-trip through their JSON form, accept overrides and
    reject unknown keys. Each lives in configs/<rule set>/, plays that named rule set and writes to
    checkpoints/<rule set>/<run>. Within a rule set the experiments may differ from the baseline
    only in the network (so a comparison measures the network), and the search must discount like
    the value targets."""
    found = 0
    for ruleset in sorted(os.listdir(CONFIG_DIR)):
        assert ruleset in RULESETS, f"configs/{ruleset} is not a rule set"
        with open(os.path.join(CONFIG_DIR, ruleset, "baseline.json")) as f:
            baseline_data = json.load(f)
        for name in sorted(os.listdir(os.path.join(CONFIG_DIR, ruleset))):
            path = os.path.join(CONFIG_DIR, ruleset, name)
            config = load_config(path)
            found += 1
            assert config.ruleset == ruleset and config.rules is RULESETS[ruleset]
            assert config.checkpoint_dir == f"checkpoints/{ruleset}/{name[:-5]}", f"{name}: checkpoint_dir"
            assert from_dict(config.to_dict()).to_dict() == config.to_dict(), f"round trip changed {name}"
            assert config.to_flat_dict()["rules"] == RULESETS[ruleset].to_dict()
            assert config.search_discount == config.reward_discount, f"{name}: discounts differ"
            TaflNet(config.network, config.board_size)  # every network builds
            with open(path) as f:
                data = json.load(f)
            assert data["system"]["num_workers"] is None, f"{name}: num_workers must be null"
            for section in ("game", "self_play", "training", "replay_buffer", "evaluation"):
                assert data[section] == baseline_data[section], f"{ruleset}/{name}: {section} differs"
    assert found == 13
    baseline = load_config(os.path.join(CONFIG_DIR, "brandubh", "baseline.json"))
    assert (baseline.num_mcts_sims_attacker, baseline.batch_size, baseline.network.value_hidden,
            baseline.network.squeeze_excitation, baseline.rules.throne) == (1000, 256, 256, True, True)
    hnef = load_config(os.path.join(CONFIG_DIR, "copenhagen", "baseline.json"))
    assert (hnef.rules.king_capture, hnef.value_loss_weight, hnef.network.channels,
            hnef.network.num_blocks, hnef.reward_discount) == ("strong", 1.5, 64, 8, 0.995)
    changed = apply_overrides(baseline, ["self_play.num_iterations=3", "network.channels=48",
                                         "network.features.history_length=4", 'network.blocks=["line","line"]',
                                         "num_workers=2", "name=test"])
    assert (changed.num_iterations, changed.network.channels, changed.network.features.history_length,
            changed.network.blocks, changed.num_workers, changed.name) == (3, 48, 4, ["line", "line"], 2, "test")
    assert baseline.num_iterations == 1000, "apply_overrides must not change the original"
    for bad in (["self_play.no_such_key=1"], ["nosuchsection.x=1"], ['network.blocks=["cube"]']):
        try:
            apply_overrides(baseline, bad)
            assert False, f"{bad} was accepted"
        except ValueError:
            pass
    try:
        from_dict({"training": {"batch_siz": 8}})
        assert False, "typo accepted"
    except ValueError as e:
        assert "batch_siz" in str(e)
    config = TrainingConfig()
    assert config.network is not TrainingConfig.network, "instances must not share the class-level network"
    print(f"  {found} configs load, round-trip and build; overrides and typo detection work")
    return True


def test_training_end_to_end():
    """A very short run of the full pipeline (self-play with reach planes, training, gating,
    checkpoint) with a line-block network, then loading the checkpoint back and exporting it."""
    from train import train
    from convert_to_onnx import export_to_onnx
    original_compile = torch.compile
    torch.compile = lambda model, **kwargs: model  # no compilation in this test
    try:
        with tempfile.TemporaryDirectory() as tmp:
            config = load_config(os.path.join(CONFIG_DIR, "brandubh", "line_4x32_mobility.json"))
            config = apply_overrides(config, [
                "self_play.num_iterations=2", "self_play.num_games_per_iteration=2",
                "self_play.num_mcts_sims_attacker=4", "self_play.num_mcts_sims_defender=4",
                "self_play.track_unique_states=true", "network.channels=16", 'network.blocks=["line","line3"]',
                "network.global_pool=every_other", "network.features.history_length=2",
                "training.batch_size=8", "training.batches_per_epoch=2", "training.num_epochs=1",
                "training.use_stepping_lr=false", "replay_buffer.replay_buffer_size=4000",
                "replay_buffer.min_buffer_size=64", "evaluation.eval_games=2", "evaluation.eval_frequency=1",
                "evaluation.eval_mcts_sims_attacker=2", "evaluation.eval_mcts_sims_defender=2",
                "evaluation.eval_vs_random_games=1", "evaluation.eval_vs_random_frequency=1",
                "system.num_workers=1", f"system.checkpoint_dir={tmp}", "system.temp_dir=null"])
            network, best, history = train(config)
            assert os.path.exists(os.path.join(tmp, "run_config.json"))
            with open(os.path.join(tmp, "run_config.json")) as f:
                assert json.load(f) == config.to_dict()
            assert len(history["iterations"]) == 2 and np.isfinite(history["policy_loss"][-1])
            assert np.isfinite(history["ending_loss"][-1])
            assert np.isfinite(history["moves_left_loss"][-1])
            assert sum(history["selfplay_results"]["endings"][-1].values()) == 2
            # games.jsonl: every self-play and evaluation game of both iterations, with its ending
            # and final board
            with open(os.path.join(tmp, "games.jsonl")) as f:
                games = [json.loads(line) for line in f]
            types = [g["type"] for g in games]
            assert types.count("selfplay") == 4 and {"vs_random", "vs_best"} <= set(types)
            assert all(g["ending"] in ENDINGS and len(g["board"]) == 7 for g in games)
            assert all(g["network_side"] in ("attackers", "defenders")
                       for g in games if g["type"] != "selfplay")
            path = os.path.join(tmp, "checkpoint_iter_2.pth")
            loaded, checkpoint = load_network(path)
            assert checkpoint["network_config"] == config.network.to_dict()
            assert checkpoint["config"]["ruleset"] == "brandubh" and checkpoint["run_config"] == config.to_dict()
            assert checkpoint["config"]["rules"] == RULESETS["brandubh"].to_dict()
            assert isinstance(loaded, TaflNet) and loaded.feature_spec.mobility_planes
            game = Tafl("brandubh")
            x = torch.from_numpy(build_features(game, loaded.feature_spec))[None]
            with torch.no_grad():
                assert torch.allclose(loaded(x)[0], network.eval()(x)[0])
            export_to_onnx(loaded, os.path.join(tmp, "model.onnx"), 7, loaded.input_channels)
            # Resuming works and keeps the config file
            train(apply_overrides(config, ["self_play.num_iterations=3"]), resume_from=path)
            assert os.path.exists(os.path.join(tmp, "checkpoint_iter_3.pth"))
    finally:
        torch.compile = original_compile
    print("  end-to-end: self-play, training, gating, checkpoint, reload, ONNX export, resume")
    return True


if __name__ == "__main__":
    print("Testing the configurable networks, features and configs...")
    tests = [test_reach_planes_match_move_generator, test_feature_layout, test_policy_head_index_order,
             test_line_block_sees_whole_lines, test_attention_block_sees_whole_board,
             test_architectures_forward_and_onnx,
             test_checkpoint_loading_and_stem_adaptation, test_run_configs,
             test_training_end_to_end]
    all_passed = True
    for test in tests:
        try:
            all_passed &= test()
        except Exception as e:  # any error is a failure, not only a failed assert
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {test.__name__}: {e}")
            all_passed = False
    print("All tests passed!" if all_passed else "Some tests FAILED")
    raise SystemExit(0 if all_passed else 1)
