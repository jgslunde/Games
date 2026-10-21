# Rule sets

Every training run plays one named rule set from `tafl.RULESETS`. The rule set fixes the board,
the starting position and every rule. Configs live in `configs/<rule set>/<run>.json` and write to
`checkpoints/<rule set>/<run>/`, so the directory of a network says which rules it plays. Each
checkpoint also stores the complete rules (`config['rules']`), and `convert_to_onnx.py` copies them
into the model's `.json` file.

The named sets are the nine most-played variants, as defined card by card in the implementation
reference of the tafl overview (October 2026, parts A-C). The historical 11x11 card has three
setups, so there are eleven names. The engine has more switches (see `Rules` in `tafl.py`), which
tests use to build other combinations, but runs must name a rule set.

## The named rule sets

Defender wins are from human games on aagenielsen.dk: a draw counts as half a win, and 50 % is
fair. The number of games is in brackets.

| Name | Card | Board | First | King wins | King captured | Throne | Extra rules | Repetition | Defender wins | Configs |
|---|---|---|---|---|---|---|---|---|---|---|
| `brandubh` | C1 Brandubh 2 | 7x7 | attackers | corner | two; four on the throne | hostile | encirclement | 3rd: attackers win | 62 % (2,318) | `configs/brandubh/` |
| `walker_brandub` | C2 Walker's Brandub | 7x7 | attackers | corner | two everywhere | not hostile | encirclement | 3rd: attackers win | 55 % (468) | |
| `brandubh_2_5w` | C3 Brandubh 2.5W | 7x7 | defenders | corner | two; four on the throne | hostile, only the king may pass | encirclement | 5th: attackers win | 59 % (354) | |
| `sami_tablut` | C4 Sami Tablut | 9x9 | defenders | edge | two; four on the throne, three beside it | hostile | encirclement | 3rd: attackers win | 49 % (966) | `configs/sami_tablut/` |
| `sea_battle_9x9` | C5 Sea Battle | 9x9 | attackers | edge | four; unarmed king | none | encirclement | 3rd: attackers win | 51 % (352) | |
| `market_hnefatafl` | C6 Market Hnefatafl | 9x9 | defenders | edge | two everywhere | none | none | 3rd: attackers win | 46 % (222) | |
| `copenhagen` | C7 Copenhagen | 11x11 | attackers | corner or exit fort | four (three beside the throne), never on the edge | hostile | encirclement, shieldwall | 3rd: attackers win | 58 % (13,640); top players 52 % | `configs/copenhagen/` |
| `historical_lewis` | C8 Historical, Lewis cross | 11x11 | defenders | edge | as Sami Tablut | hostile | encirclement | 3rd: attackers win | 52 % (664) | |
| `historical_bell` | C8 Historical, Bell | 11x11 | defenders | edge | as Sami Tablut | hostile | encirclement | 3rd: attackers win | 51 % (314) | |
| `simple_tafl` | C8 Simple Tafl | 11x11 | defenders | edge | two everywhere (no throne) | none | encirclement | 3rd: attackers win | 50 % (142) | |
| `fetlar` | C9 Fetlar | 11x11 | attackers | corner | as Copenhagen | hostile | encirclement | 3rd: draw | 59 % (1,564) | |

"Hostile throne" means: hostile to attackers always, to defenders when empty, never to the king.
"Throne: none" means the centre is an ordinary square. Corners are restricted and hostile to all
pieces in the corner-goal sets, and ordinary squares in the edge-goal sets. Training ends a game as
a draw after `max_game_length` plies; the published rules have no such limit.

## How the engine follows the reference

- Part A (shared rules) and the order of checks after a move (A7) are in `Tafl.make_move`.
- Encirclement (A8): `_defenders_encircled`, after every attackers' move.
- Repetition (A9): every position is counted, the start included. The Nth occurrence ends the
  game, whichever side made the move.
- Shieldwall (B1): `_shieldwall_captures`, including the **[Ours]** choice that the king counts as
  a front piece.
- Exit fort (B2): `_king_in_exit_fort`, the reference's safe-set test, after every defenders' move.
- The cards' **[Ours]** choices are all taken as written. Two of them deserve a check:
  - the start position of Brandubh 2.5W, assumed to be that of Brandubh 2;
  - Walker's Brandub with encirclement.
- Not implemented: draws by agreement (A10) and two-game match scoring (A11). Neither matters for
  self-play.
- A finished game records how it ended in `Tafl.end_reason`, one of `tafl.ENDINGS`:
  `king_escaped`, `exit_fort`, `attackers_blocked`, `king_captured`, `encircled`,
  `defenders_blocked` (a side without legal moves loses), `repetition`, or `move_limit` (the draw
  that training imposes after `max_game_length` plies). Training prints these counts for its
  self-play and evaluation games, and appends every game's ending and final board to
  `checkpoints/<rule set>/<run>/games.jsonl`.

## Changes since the first version of the rule sets (commit 2c6f2f5)

- Copenhagen:
  - the exit fort test is now the reference's B2 test, which also accepts walls protected by
    soldiers outside the wall;
  - it now runs only after defenders' moves (before, it ran after every move);
  - everything else is unchanged.
- Brandubh 2: the third occurrence of a position is now an attacker win, not a forbidden move.
- `sami_tablut` now means the defenders-first championship version (the former
  `sami_tablut_defenders_first`); the attackers-first version is gone.
- `old_hnefatafl` is gone: it is not one of the nine variants.
- New: `walker_brandub`, `brandubh_2_5w`, `sea_battle_9x9`, `market_hnefatafl`,
  `historical_lewis`, `historical_bell` and `simple_tafl`.

## Deprecated runs

Everything trained before the rule sets is in `deprecated_checkpoints/`, and the current code
cannot load it. None of these runs used a published rule set:

| Folder | Board | Rules (verified where stated) |
|---|---|---|
| `checkpoints_brandubh`, `checkpoints_brandubh_new`, `checkpoints_brandubh_SE` | 7x7 | No throne, king captured by two attackers everywhere, armed king, hostile corners, corner goal, no encirclement, third repetition forbidden (verified for `littleboy_decay_fixed`, `littleboy_decay`, `fatman_nothrone` and `jack_nothrone`). Closest published set: Walker's Brandub, which has a throne. |
| `checkpoints_tablut` | 9x9 | The same rules on the Tablut board (verified for `test_6x48`). Closest published set: Corner Tablut (38 % defender wins), which has a throne. |
| `checkpoints_hnefatafl` | 11x11 | King captured on four sides with the edge and corners counting, hostile throne, corner goal, encirclement of the king alone (verified for `baseline` and `line_10x64_gpool`). This is "Old Hnefatafl" with a different encirclement rule. |
| `checkpoints_old`, `checkpoints` | all | Early experiments with various rules. |

The ONNX example files in git (`deprecated_checkpoints/*/.../checkpoint_iter_*.onnx`) belong to
these runs. Their `.json` files list the old rule switches.

`deprecated_checkpoints_2/copenhagen/` holds the first Copenhagen runs under the named rule sets
(`baseline`, `line_10x64_gpool`, `line_10x64_gpool_mobility_region`, to iteration 17-30;
`attention_8x64` crashed in its first training step). Their rules equal today's `copenhagen`, except
that runs started before commit 460c6c1 and not resumed since used the exit fort test of 2c6f2f5
(see above). Their networks lack the ending
and moves-left outputs, so the current code cannot load them.
