"""
Do the networks see immediate wins? This script builds random positions in which the side to move
wins with one move, one kind of win at a time, and checks what the networks make of them:

    escape          the king reaches a corner (or the edge, in edge-goal rule sets)
    capture         the attackers capture the king away from the throne
    capture_throne  the attackers capture the king beside the empty throne
    encircle        the attackers close the last gap in a ring around all defenders
    fort            the defenders close an exit fort around the king on the edge

The engine decides which moves win and how, so a position is never mislabelled. A position is kept
only if every winning move is of its kind (a fort position with an open corner tests nothing).

Printed per kind:
    legal    the mean number of legal moves
    uniform  the share of the legal moves that win (what a policy without knowledge gives them)
    p_win    the mean policy share of the winning moves
    top1     how often the policy's first choice wins
    rank     the median rank of the best winning move in the policy (1 = first choice)
    value    the mean value of the position for the side to move (+1 = sure win)
    search   with --sims: how often a search with that many simulations plays a winning move

Usage:
    python probe_tactics.py checkpoints/copenhagen/*/checkpoint_iter_30.pth --positions 200 --sims 100
"""

import argparse
from typing import List, Optional, Set, Tuple

import numpy as np
import scipy.ndimage
import torch

from agent import Agent
from features import build_features, special_squares
from tafl import (ATTACKER, ATTACKER_PLAYER, DEFENDER, DEFENDER_PLAYER, DIRECTIONS, EMPTY, KING,
                  Tafl, get_rules, move_index)
from tafl_network import load_network

KINDS = ("escape", "capture", "capture_throne", "encircle", "fort")


def scatter(board: np.ndarray, piece: int, count: int, allowed: np.ndarray, rng):
    """Put `count` pieces on random empty squares of `allowed` (fewer if there is no room)."""
    squares = np.argwhere(allowed & (board == EMPTY))
    for r, c in squares[rng.permutation(len(squares))[:count]]:
        board[r, c] = piece


def random_position(kind: str, rules, rng) -> Optional[Tafl]:
    """
    A random position built so that a win of `kind` is likely for the side to move, or None if the
    attempt failed. Whether the win is really there is checked afterwards by find_wins.
    """
    game = Tafl(rules)
    n, m = game.size, game.size // 2
    corners, throne = special_squares(game)
    free = ~corners & ~throne  # where soldiers may stand
    edge = np.ones((n, n), dtype=bool)
    edge[1:-1, 1:-1] = False
    goal = corners if rules.king_goal == "corner" else edge
    board = np.zeros((n, n), dtype=game.board.dtype)
    player = ATTACKER_PLAYER

    if kind == "escape":
        scatter(board, KING, 1, ~goal, rng)
        scatter(board, ATTACKER, rng.integers(8, 25), free, rng)
        scatter(board, DEFENDER, rng.integers(2, 13), free, rng)
        player = DEFENDER_PLAYER

    elif kind in ("capture", "capture_throne"):
        # The king off the edge with attackers beside him on all sides but one; beside the throne,
        # the empty throne takes the place of one attacker
        beside_throne = [(m + dr, m + dc) for dr, dc in DIRECTIONS]
        squares = [(r, c) for r in range(1, n - 1) for c in range(1, n - 1)
                   if (r, c) != (m, m) and ((r, c) in beside_throne) == (kind == "capture_throne")]
        kr, kc = squares[rng.integers(len(squares))]
        board[kr, kc] = KING
        sides = [(kr + dr, kc + dc) for dr, dc in DIRECTIONS if (kr + dr, kc + dc) != (m, m)]
        for i in rng.permutation(len(sides))[1:]:  # all sides but a random one
            board[sides[i]] = ATTACKER
        scatter(board, ATTACKER, rng.integers(6, 22), free, rng)
        scatter(board, DEFENDER, rng.integers(2, 13), free, rng)

    elif kind == "encircle":
        # A random area around the centre, grown one square at a time and kept off the edge. The
        # squares beside it (sharing a side) are the ring: attackers on all but one. The king and
        # the defenders go inside, the other attackers outside.
        area = np.zeros((n, n), dtype=bool)
        area[m + rng.integers(-2, 3), m + rng.integers(-2, 3)] = True
        for _ in range(rng.integers(8, 50)):
            frontier = np.argwhere(scipy.ndimage.binary_dilation(area) & ~area & ~edge)
            area[tuple(frontier[rng.integers(len(frontier))])] = True
        ring = scipy.ndimage.binary_dilation(area) & ~area
        if ring.sum() > 24 or (ring & throne).any():
            return None
        board[ring] = ATTACKER
        board[tuple(np.argwhere(ring)[rng.integers(ring.sum())])] = EMPTY  # the gap
        scatter(board, KING, 1, area, rng)
        scatter(board, DEFENDER, rng.integers(1, 13), area & free, rng)
        scatter(board, ATTACKER, rng.integers(1, 26 - ring.sum()), free & ~area & ~ring, rng)

    elif kind == "fort":
        # A fort on the top edge: the king, 1-3 empty squares he can walk to (the inside), defenders
        # on all squares beside the inside (the wall) and up to 3 more nearby. If that is a fort,
        # one wall defender moves away along its row or column, and the board is turned to a
        # random edge.
        kc = rng.integers(2, n - 2)
        inside = np.zeros((n, n), dtype=bool)
        inside[0, kc] = True
        rows = np.arange(n)[:, None] * np.ones(n, dtype=int)
        for _ in range(rng.integers(1, 4)):
            frontier = np.argwhere(scipy.ndimage.binary_dilation(inside) & ~inside & (rows < 2) & free)
            inside[tuple(frontier[rng.integers(len(frontier))])] = True
        wall = scipy.ndimage.binary_dilation(inside) & ~inside
        if (wall & ~free).any():
            return None
        board[wall] = DEFENDER
        board[0, kc] = KING
        near = scipy.ndimage.binary_dilation(wall, iterations=2) & ~inside
        scatter(board, DEFENDER, rng.integers(0, 4), near & free, rng)
        game.board = board
        if not game._king_in_exit_fort():
            return None
        gr, gc = np.argwhere(wall)[rng.integers(wall.sum())]
        board[gr, gc] = EMPTY
        line = (board == EMPTY) & free & ~inside & ((rows == gr) | (rows.T == gc))
        line[gr, gc] = False
        scatter(board, DEFENDER, 1, line, rng)
        scatter(board, ATTACKER, rng.integers(8, 25), free & ~near, rng)
        board = np.rot90(board, rng.integers(4)).copy()
        player = DEFENDER_PLAYER

    game.board = board
    game.current_player = player
    game.position_history = [game._position_key()]
    return game


def win_kind(game: Tafl, move: Tuple[int, int, int, int]) -> Optional[str]:
    """How `move` wins for the side to move ("other" for a win of no listed kind), or None."""
    after = game.clone()
    after.make_move(move)
    if not after.game_over or after.winner != game.current_player:
        return None
    if after.winner == DEFENDER_PLAYER:
        if game.board[move[:2]] == KING and after._is_goal(*move[2:]):
            return "escape"
        return "fort" if after.rules.exit_fort and after._king_in_exit_fort() else "other"
    if not (after.board == KING).any():
        kr, kc = np.argwhere(game.board == KING)[0]
        beside_throne = game.rules.throne and abs(kr - game.size // 2) + abs(kc - game.size // 2) == 1
        return "capture_throne" if beside_throne else "capture"
    return "encircle" if after._defenders_encircled() else "other"


def make_probes(kind: str, rules, count: int, rng, max_tries: int = 100_000):
    """Up to `count` positions in which the side to move wins with a move of `kind`, and only so.
    Returns a list of (game, set of winning moves)."""
    probes = []
    for _ in range(max_tries):
        game = random_position(kind, rules, rng)
        if game is None or game._defenders_encircled():
            continue
        moves = game.get_legal_moves()
        kinds = [win_kind(game, move) for move in moves]
        if kind in kinds and set(kinds) == {kind, None}:
            probes.append((game, {move for move, k in zip(moves, kinds) if k}))
            if len(probes) == count:
                break
    return probes


def probe_network(path: str, probes: List[Tuple[Tafl, Set]], sims: int) -> dict:
    """The policy share and rank of the winning moves, the value, and (if sims > 0) whether the
    search plays a winning move, for every probe position."""
    network, checkpoint = load_network(path)
    states = np.stack([build_features(game, network.feature_spec) for game, _ in probes])
    with torch.inference_mode():
        logits, values = network(torch.from_numpy(states))
    config = checkpoint['config']
    agent = Agent(network, num_simulations=sims, c_puct=config['c_puct'],
                  fpu_reduction=config['fpu_reduction'], search_discount=config['search_discount'])
    results = {key: [] for key in ("legal", "uniform", "p_win", "top1", "rank", "value", "search")}
    for (game, winning), position_logits, value in zip(probes, logits.numpy(), values.numpy()[:, 0]):
        moves = game.get_legal_moves()
        legal_logits = position_logits[[move_index(move, game.size) for move in moves]]
        probs = np.exp(legal_logits - legal_logits.max())
        probs /= probs.sum()
        wins = np.array([move in winning for move in moves])
        results["legal"].append(len(moves))
        results["uniform"].append(wins.mean())
        results["p_win"].append(probs[wins].sum())
        results["top1"].append(wins[np.argmax(probs)])
        results["rank"].append(1 + np.flatnonzero(wins[np.argsort(-probs)])[0])
        results["value"].append(value)
        results["search"].append(agent.select_move(game, temperature=0.0) in winning if sims else np.nan)
    return {key: np.array(values) for key, values in results.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("checkpoints", nargs="+", help="training checkpoints (.pth) of one rule set")
    parser.add_argument("--positions", type=int, default=200, help="positions per kind of win")
    parser.add_argument("--sims", type=int, default=0, help="also run a search with this many simulations")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show", action="store_true", help="print one position of each kind")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoints[0], map_location="cpu", weights_only=False)
    rules = get_rules(checkpoint['config']['rules'])
    kinds = [kind for kind in KINDS if (kind != "capture_throne" or rules.throne)
             and (kind != "encircle" or rules.encirclement) and (kind != "fort" or rules.exit_fort)]
    rng = np.random.default_rng(args.seed)
    probes = {kind: make_probes(kind, rules, args.positions, rng) for kind in kinds}
    print(f"Rule set {rules.name}: " + ", ".join(f"{len(p)} {k}" for k, p in probes.items()))
    if args.show:
        for kind, kind_probes in probes.items():
            game, winning = kind_probes[0]
            print(f"\n{kind}: winning moves {sorted(winning)}\n{game}")

    all_probes = [probe for kind in kinds for probe in probes[kind]]
    for path in args.checkpoints:
        if get_rules(torch.load(path, map_location="cpu", weights_only=False)['config']['rules']) != rules:
            raise ValueError(f"{path} was trained under other rules than {args.checkpoints[0]}")
        results = probe_network(path, all_probes, args.sims)
        print(f"\n{path}")
        print(f"  {'kind':<15} {'legal':>5} {'uniform':>7} {'p_win':>6} {'top1':>5} {'rank':>5} "
              f"{'value':>6} {'search':>6}")
        start = 0
        for kind in kinds:
            part = {key: values[start:start + len(probes[kind])] for key, values in results.items()}
            start += len(probes[kind])
            search = f"{np.mean(part['search']):6.0%}" if args.sims else "     -"
            print(f"  {kind:<15} {part['legal'].mean():5.0f} {part['uniform'].mean():7.3f} "
                  f"{part['p_win'].mean():6.3f} {part['top1'].mean():5.0%} {np.median(part['rank']):5.0f} "
                  f"{part['value'].mean():+6.2f} {search}")


if __name__ == "__main__":
    main()
