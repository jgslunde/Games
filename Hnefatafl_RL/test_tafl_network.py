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

from brandubh import Brandubh
from features import FeatureSpec, build_features, constant_planes, king_region_plane, reach_planes
from games import GAMES, game_spec
from hnefatafl import Hnefatafl
from network import MoveEncoder
from network_config import NetworkConfig
from run_config import TrainingConfig, apply_overrides, from_dict, load_config
from tafl_network import (BilinearPolicyHead, ConvPolicyHead, TaflNet, build_network, count_parameters,
                          load_network, load_weights, move_index_tables, network_config_from_checkpoint)

CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")


def random_position(game_class, rng, rules=None, max_plies=40):
    """A position reached by random play (or the start position)."""
    game = game_class(**(rules or {}))
    for _ in range(rng.integers(0, max_plies + 1)):
        moves = game.get_legal_moves()
        if game.game_over or not moves:
            break
        game.make_move(moves[rng.integers(len(moves))])
    return game


def test_reach_planes_match_move_generator():
    """The vectorised reach planes must equal the game's own move generator for both sides, with
    and without the throne, and the king-region plane must agree with the encirclement check."""
    rng = np.random.default_rng(0)
    for spec in GAMES.values():
        n = spec.board_size
        for throne_enabled in (False, True):
            for _ in range(12):
                game = random_position(spec.game_class, rng, dict(throne_enabled=throne_enabled))
                expected = np.zeros((3, n, n), dtype=np.float32)
                for r in range(n):
                    for c in range(n):
                        piece = game.board[r, c]
                        if piece == 0:
                            continue
                        for _, _, tr, tc in game._get_piece_moves(r, c):
                            expected[0 if piece == 1 else 1, tr, tc] = 1
                            if piece == 3:
                                expected[2, tr, tc] = 1
                assert np.array_equal(reach_planes(game), expected), f"reach planes differ ({spec.name})"
    for _ in range(30):
        game = random_position(Hnefatafl, rng, max_plies=80)
        region = king_region_plane(game)
        if not (game.board == 3).any():
            assert not region.any()
            continue
        touches_edge = region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()
        assert touches_edge == (not game._is_king_encircled()), "region plane disagrees with encirclement"
        assert region[game.board == 3].all() and not region[game.board == 1].any()
    # A hand-made ring of attackers: the region is the inside of the ring
    game = Hnefatafl()
    game.board[:] = 0
    game.board[5, 5] = 3
    for r, c in [(3, 5), (7, 5), (5, 3), (5, 7), (4, 4), (4, 6), (6, 4), (6, 6)]:
        game.board[r, c] = 1
    region = king_region_plane(game)
    assert region.sum() == 5 and game._is_king_encircled()
    print("  reach planes equal the move generator; king-region plane equals the encirclement flood fill")
    return True


def test_feature_layout():
    """Plane names, counts and contents of every feature option."""
    game = Brandubh(throne_enabled=True)
    full = FeatureSpec(history_length=2, constant_planes=True, mobility_planes=True, region_plane=True)
    names = full.plane_names()
    assert names == ["attackers_t0", "defenders_t0", "king_t0", "attackers_t1", "defenders_t1", "king_t1",
                     "player", "ones", "corners", "throne", "reach_attackers", "reach_defenders",
                     "reach_king", "king_region"]
    state = build_features(game, full)
    assert state.shape == (14, 7, 7) and state.dtype == np.float32
    assert np.array_equal(state[:3], game.get_state()[:3]) and not state[3:6].any()  # no earlier position
    assert np.array_equal(state[6], game.get_state()[3])
    assert state[7].all() and state[8].sum() == 4 and state[8, 0, 0] == 1 and state[9].sum() == 1 and state[9, 3, 3] == 1
    assert np.array_equal(state[10:13], reach_planes(game)) and np.array_equal(state[13], king_region_plane(game))
    assert set(np.unique(state)) <= {0.0, 1.0}
    assert not constant_planes(Brandubh(throne_enabled=False))[2].any(), "disabled throne must give an empty plane"
    # The default spec gives the classic 4 planes
    assert np.array_equal(build_features(game, FeatureSpec()), game.get_state())
    print("  feature planes: names, order and contents as documented")
    return True


def test_policy_head_index_order():
    """Both new policy heads must put move (from, direction, distance) where MoveEncoder puts it."""
    for name, spec in GAMES.items():
        n = spec.board_size
        encoder = spec.move_encoder_class
        head = ConvPolicyHead(8, n)
        with torch.no_grad():
            head.conv.weight.zero_()
            head.conv.bias.copy_(torch.arange(4 * (n - 1), dtype=torch.float32))
        logits = head(torch.zeros(1, 8, n, n))[0]
        to_square, on_board = move_index_tables(n)
        game = spec.game_class(throne_enabled=False)
        for move in game.get_legal_moves():
            idx = encoder.encode_move(move)
            direction, distance = (idx % (4 * (n - 1))) // (n - 1), idx % (n - 1) + 1
            assert logits[idx].item() == direction * (n - 1) + distance - 1
            assert on_board[idx] and to_square[idx] == move[2] * n + move[3]
            assert encoder.decode_move(idx) == move
        assert on_board.sum() == sum(1 for i in range(len(on_board)) if on_board[i])
        bilinear = BilinearPolicyHead(8, n, dim=4)
        logits = bilinear(torch.randn(2, 8, n, n))
        assert logits.shape == (2, n * n * 4 * (n - 1))
        assert (logits[:, ~torch.from_numpy(on_board)] == -1e4).all() and torch.isfinite(logits).all()
    print("  conv and bilinear policy heads use the MoveEncoder index order")
    return True


ARCHITECTURES = [
    dict(channels=16, blocks=["res"] * 2, squeeze_excitation=True, policy_head="dense", value_head="dense"),
    dict(channels=16, blocks=["line"] * 2, policy_head="conv", value_head="pooled",
         features=dict(constant_planes=True)),
    dict(channels=16, blocks=["line3", "res"], global_pool="every_other", policy_head="conv",
         value_head="pooled", features=dict(history_length=3, mobility_planes=True, region_plane=True)),
    dict(channels=16, blocks=["thick"] * 2, global_pool="all", policy_head="bilinear", policy_bilinear_dim=8),
    dict(channels=16, blocks=["line"] * 3, tie_blocks=True, global_pool="all", policy_head="conv"),
]


def test_architectures_forward_and_onnx():
    """Every block kind, head and feature option must build, train (batch forward/backward), run at
    batch 1 in eval mode, export to ONNX (opset 12, as convert_to_onnx.py) and give the same outputs
    in onnxruntime."""
    try:
        import onnxruntime as ort
    except ImportError:
        ort = None
    for spec in GAMES.values():
        n = spec.board_size
        for arch in ARCHITECTURES:
            config = NetworkConfig.from_dict(arch)
            network = build_network(config, n)
            planes = config.features.num_planes
            x = (torch.rand(4, planes, n, n) < 0.2).float()
            network.train()
            policy, value = network(x)
            assert policy.shape == (4, spec.policy_size) and value.shape == (4, 1)
            (policy.logsumexp(1).mean() + value.mean()).backward()
            network.eval()
            with torch.no_grad():
                policy1, value1 = network(x[:1])
            assert policy1.shape == (1, spec.policy_size) and -1 <= value1.item() <= 1
            assert count_parameters(network) > 0
            if config.tie_blocks:
                assert len(network.blocks) == 1 and len(network.block_order) == 3
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "net.onnx")
                torch.onnx.export(network, x[:1], path, opset_version=12, input_names=["board_state"],
                                  output_names=["policy_logits", "value"],
                                  dynamic_axes={"board_state": {0: "b"}, "policy_logits": {0: "b"}, "value": {0: "b"}},
                                  dynamo=False)
                if ort is not None:
                    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
                    onnx_policy, onnx_value = session.run(None, {"board_state": x[:2].numpy()})
                    with torch.no_grad():
                        torch_policy, torch_value = network(x[:2])
                    assert np.allclose(onnx_policy, torch_policy.numpy(), atol=1e-4), "ONNX policy differs"
                    assert np.allclose(onnx_value, torch_value.numpy(), atol=1e-4), "ONNX value differs"
    print("  all block kinds, heads and features: forward, backward, batch-1 eval, ONNX export match")
    return True


def test_checkpoint_loading_and_stem_adaptation():
    """Networks must come back from checkpoints (new and legacy format), and weights must transfer
    to a network with more input planes: the shared planes keep their weights."""
    from network import BrandubhNet
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
        # Legacy checkpoint (old flat config, old class)
        legacy = BrandubhNet(num_res_blocks=1, num_channels=8).eval()
        legacy_path = os.path.join(tmp, "legacy.pth")
        torch.save({"model_state_dict": legacy.state_dict(),
                    "config": {"num_res_blocks": 1, "num_channels": 8, "board_size": 7,
                               "value_head_hidden_size": 64, "history_length": 1}}, legacy_path)
        loaded, checkpoint = load_network(legacy_path)
        assert isinstance(loaded, BrandubhNet)
        assert network_config_from_checkpoint(checkpoint).arch == "legacy"
        x7 = (torch.rand(1, 4, 7, 7) < 0.2).float()
        with torch.no_grad():
            assert torch.equal(loaded(x7)[1], legacy(x7)[1])
        # A legacy checkpoint without a board size is recognised by its policy head
        torch.save({"model_state_dict": legacy.state_dict()}, legacy_path)
        assert isinstance(load_network(legacy_path)[0], BrandubhNet)
    print("  checkpoints load (new and legacy); weights transfer to networks with more input planes")
    return True


def test_run_configs():
    """All shipped configs must load, round-trip through their JSON form, accept overrides and
    reject unknown keys; the baselines must match the old training-script defaults."""
    found = 0
    for game in ("brandubh", "tablut", "hnefatafl"):
        for name in sorted(os.listdir(os.path.join(CONFIG_DIR, game))):
            config = load_config(os.path.join(CONFIG_DIR, game, name))
            found += 1
            assert config.game == game and config.board_size == game_spec(game).board_size
            assert from_dict(config.to_dict()).to_dict() == config.to_dict(), f"round trip changed {name}"
            assert config.to_flat_dict()["num_channels"] == config.network.channels
            build_network(config.network, config.board_size)  # every network builds
    assert found == 12
    baseline = load_config(os.path.join(CONFIG_DIR, "brandubh", "baseline.json"))
    assert (baseline.num_mcts_sims_attacker, baseline.batch_size, baseline.network.value_hidden,
            baseline.network.squeeze_excitation, baseline.throne_enabled) == (1000, 256, 256, True, False)
    hnef = load_config(os.path.join(CONFIG_DIR, "hnefatafl", "baseline.json"))
    assert (hnef.king_capture_pieces, hnef.value_loss_weight, hnef.network.channels) == (4, 40.0, 96)
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
            path = os.path.join(tmp, "checkpoint_iter_2.pth")
            loaded, checkpoint = load_network(path)
            assert checkpoint["network_config"] == config.network.to_dict()
            assert checkpoint["config"]["king_capture_pieces"] == 2 and checkpoint["run_config"] == config.to_dict()
            assert isinstance(loaded, TaflNet) and loaded.feature_spec.mobility_planes
            game = Brandubh(throne_enabled=False)
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
             test_architectures_forward_and_onnx, test_checkpoint_loading_and_stem_adaptation,
             test_run_configs, test_training_end_to_end]
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
