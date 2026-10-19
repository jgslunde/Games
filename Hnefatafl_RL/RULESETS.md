# Rule sets

Every training run plays one named rule set from `tafl.RULESETS`. The rule set fixes the board,
the starting position and every rule. Configs live in `configs/<rule set>/<run>.json` and write to
`checkpoints/<rule set>/<run>/`, so the directory of a network says which rules it plays. Each
checkpoint also stores the complete rules (`config['rules']`), and `convert_to_onnx.py` copies them
into the model's `.json` file.

Only commonly played, published rule sets are named. The engine has more switches (see `Rules` in
`tafl.py`), which tests use to build other combinations, but runs must name a rule set.

## The named rule sets

Defender wins in human games on aagenielsen.dk: a draw counts as half a win, and 50 % is fair. The
number of games is in brackets. The source is the October 2026 tafl overview.

| Name | Board | Published rules | First | King wins | King captured | Throne | Extra rules | Repetition | Defender wins | Configs |
|---|---|---|---|---|---|---|---|---|---|---|
| `brandubh` | 7x7 | Brandubh 2 (WTF, 2020), the most played 7x7 rules | attackers | corner | two sides, four on the throne | hostile | encirclement | not allowed | 62 % (2,318); top players 69 % | `configs/brandubh/` |
| `sami_tablut` | 9x9 | Sami Tablut, historical rules after Linnaeus | attackers | edge | two sides; four on the throne, three beside it | hostile | encirclement | defenders lose | 46 % (~2,700) | none |
| `sami_tablut_defenders_first` | 9x9 | Sami Tablut as in the WTF championship | defenders | edge | as above | hostile | encirclement | defenders lose | 49 % (966) | `configs/sami_tablut_defenders_first/` |
| `copenhagen` | 11x11 | Copenhagen (WTF, 2012), the most played 11x11 rules | attackers | corner | four sides (three beside the throne), never on the edge | hostile | encirclement, shieldwall, exit fort | defenders lose | 58 % (13,640); top players 52 % | `configs/copenhagen/` |
| `fetlar` | 11x11 | Fetlar (2007) | attackers | corner | as Copenhagen | hostile | encirclement | draw | 59 % (1,564) | none |
| `old_hnefatafl` | 11x11 | "Old Hnefatafl": Fetlar, but the king can be captured on the edge | attackers | corner | four sides, where the edge, a corner and the empty throne count | hostile | encirclement | not allowed | 53 % (1,222); strong players 49 % | none |

Shared by all: pieces move like rooks; a piece is captured between two enemies, or an enemy and a
hostile square, by the move that closes the sandwich; the king is armed; only the king may stop on
the throne and the corners, and every piece may pass over the empty throne; the corners are
hostile to all pieces; "hostile throne" means hostile to attackers always and to defenders when
empty; a player who cannot move loses. Training ends a game as a draw after `max_game_length`
plies, which the published rules do not have.

Setups: Brandubh and Tablut use their standard crosses. All three 11x11 sets use the Copenhagen
setup (rows of five attackers plus one in front of each edge, twelve defenders in a diamond).

Sources: the October 2026 tafl overview, the official WTF Brandubh rules sheet
(`aagenielsen.dk/brandubh2_rules_en.pdf`) and the rules pages of aagenielsen.dk for Copenhagen,
Fetlar and the historical rules.

## How the engine reads the rules

Some rule texts leave room for interpretation. The engine does this:

- **Repetition, "not allowed":** a move that gives a position for the third time is not legal. A
  player who has only such moves therefore loses, as Brandubh 2 says ("the player who had the
  choice of repeating ... loses"). Any earlier occurrence counts, not only the last few moves.
- **Repetition, "defenders lose":** the third occurrence of any position ends the game as an
  attacker win, whoever made the move. This is the literal Copenhagen and historical text.
- **Encirclement:** the attackers win when no defender (king included) can reach the board edge
  through squares without attackers. This is checked after every attacker move.
- **Shieldwall:** only the piece that moves closes the bracket; the other end needs a piece of its
  side or a corner.
- **Exit fort:** the fort must be unbreakable in a strict sense. Every wall piece must, along both
  lines through it, have a neighbour that attackers can never occupy (off the board, inside the
  fort, or another wall piece). With the shieldwall rule, every wall piece on the edge must also
  have such a square in front of it. Some forts that are safe only through tactics are not
  recognised. A breakable fort is never declared safe.
- **Old Hnefatafl repetition:** the overview's source says "repetitions not allowed". It is read
  as the "not allowed" rule above.
- **Fetlar, king on the edge:** the Fetlar text mentions an exception for a lone, fully surrounded
  king on the edge, with a broken cross-reference. The engine never captures the king on the edge
  in Fetlar.

## Not yet available

Missing details, each of which the overview could fill in:

- **Walker's Brandub** (7x7): the overview gives the king capture (two sides everywhere) and the
  throne (not hostile), but not the encirclement and repetition rules. The engine supports it as
  `king_capture="weak", throne_hostile=False` once these are known.
- **Brandubh 2.5W** (7x7, draft): only the king may cross the throne. This needs a new switch
  (throne passable for the king only); it would also change the reach planes in `features.py`.
- **Historical 11x11 rules** (Lewis cross and Bell setups, Simple Tafl): the exact starting
  positions are not given.
- **Sea Battle** (9x9 and 11x11): the starting positions are not given; the rules themselves
  (unarmed strong king, edge goal, no special squares) are supported.
- **Market Hnefatafl, Corner Tablut, Ard Ri, Magpie, Ashton and Foteviken Tablut:** rarely played,
  or they need rules the engine lacks (single-step king, closed camps).

## Deprecated runs

Everything trained before the rule sets is in `deprecated_checkpoints/`, and the current code
cannot load it. None of these runs used a published rule set:

| Folder | Board | Rules (verified where stated) |
|---|---|---|
| `checkpoints_brandubh`, `checkpoints_brandubh_new`, `checkpoints_brandubh_SE` | 7x7 | No throne, king captured by two attackers everywhere, armed king, hostile corners, corner goal, no encirclement (verified for `littleboy_decay_fixed`, `littleboy_decay`, `fatman_nothrone` and `jack_nothrone`). Closest published set: Walker's Brandub, which has a throne. |
| `checkpoints_tablut` | 9x9 | The same rules on the Tablut board (verified for `test_6x48`). Closest published set: Corner Tablut (38 % defender wins), which has a throne. |
| `checkpoints_hnefatafl` | 11x11 | King captured on four sides with the edge counting, hostile throne, corner goal, encirclement of the king alone (verified for `baseline` and `line_10x64_gpool`): `old_hnefatafl` with a different encirclement rule. |
| `checkpoints_old`, `checkpoints` | all | Early experiments with various rules. |

The ONNX example files in git (`deprecated_checkpoints/*/.../checkpoint_iter_*.onnx`) belong to
these runs. Their `.json` files list the old rule switches.
