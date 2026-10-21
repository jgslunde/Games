# Proposal: training on decisive positions that self-play rarely reaches

Status: proposal, written October 2026, while the first Copenhagen runs were at iteration 17-30.
On hold: we first check whether the networks learn these ideas by themselves, since generated
positions would break the "zero" approach of learning from self-play alone. Implemented so far:
the logging of how games end (step 1 of section 5) and the extra ending output (section 4.3).

## 1. The problem

In the Copenhagen runs the defenders win 70-75 % of the self-play games, although strong human
players are close to even (52 %). One likely cause is that the attackers' main winning ideas need
many coordinated moves. Examples are closing a ring around all defenders (encirclement) and
blocking the corners. A young network almost never plays such a sequence, so it never sees what
the ending is worth, and it has no reason to aim for it. The defenders' exit fort has the same
problem from the other side.

`probe_tactics.py` measures this. It builds random positions where the side to move wins in one
move, of one kind at a time. It then checks how much of the policy goes to the winning moves, and
what value the network gives the position. It also checks whether a search with 100 simulations
plays a winning move. Results for `line_10x64_gpool`, iteration 30, 200 positions per kind:

| Kind of win | Policy share of the winning moves (random policy) | First choice wins | Value for the side to move | Search plays a win |
|---|---|---|---|---|
| king escapes to a corner | 0.94 (0.02) | 99 % | +0.95 | 100 % |
| attackers capture the king | 0.30 (0.01) | 98 % | −0.37 | 100 % |
| capture beside the throne | 0.11 (0.01) | 75 % | −0.51 | 98 % |
| attackers close a ring | 0.05 (0.02) | 13 % | −0.05 | 60 % |
| defenders close an exit fort | 0.06 (0.05) | 7 % | +0.87 | 52 % |

The other two runs give the same picture. `line_10x64_gpool_mobility_region` sees king captures
better (policy share 0.62), but not rings or forts.

- **Wins in one move.** The search finds escapes and king captures. It misses about 40 % of the
  ring closures and half of the forts, because the policy gives these moves so little weight that
  100 simulations often never try them. For forts, the search success has fallen from 82 % at
  iteration 5 to 52 %. This matters in evaluation games and in play, but not in self-play: there,
  every legal move is checked for an immediate win before the search, and a winning move is
  played and becomes the policy target.
- **Positions before the win.** These matter more. The search only steers towards an ending a
  few moves deep, and only if the value head already rates the positions on the way as good. So
  the important gap is the value of positions 3-30 moves before the decisive move. Self-play
  reaches them too rarely to learn from.

The high value in the fort positions does not mean the network sees the fort. These positions put
the king on the edge among his defenders, which the network rates as good for the defenders
whether or not a fort can be closed.

We let the runs reach at least iteration 100 before we decide whether this is a real problem.
Section 6 lists what to measure then.

## 2. What the positions must be

- **Decisive or nearly so.** One side wins by force, or is close to it. Close means a known
  small number of moves from a decisive structure (section 3.4).
- **Realistic.** The rest of the board looks like a real game at that stage: piece counts,
  where the pieces stand, which side is to move. Tafl pieces move freely, so nearly any
  arrangement can be reached in a game. The real question is whether reasonable play leads there,
  not whether the position is legal.
- **Diverse.** Many sizes, shapes and locations for each structure, both sides to move, and all
  eight board symmetries. Otherwise the network learns the generator, not the idea.
- **Rare in self-play.** Easy kinds, like a king one move from a corner, are already learned and
  get a small share.
- **Both sides learn from them.** Each structure has a winning side and a defending side. A
  near-closed ring teaches the attackers to close it and the defenders to break out. This keeps
  the data from simply pushing the balance towards one side.

## 3. How to produce the positions

Three sources, from the most artificial to the most realistic. I propose to build them in this
order, since each one reuses parts of the one before.

### 3.1 Built positions

This extends the generators in `probe_tactics.py`. First build the decisive structure, then fill
the rest of the board.

Structures to build, for Copenhagen:

| Structure | Built as | Winning side |
|---|---|---|
| Encirclement | a random area (any size, not only around the centre), the squares around it filled with attackers, all defenders inside | attackers |
| Exit fort | the king on the edge, 1-4 empty squares he can walk to, a wall of defenders that cannot be captured | defenders |
| King trap | a strong king with attackers on 2-3 sides, beside the throne or not | attackers |
| Corner blockade | attackers on the squares that guard the corners, so the king cannot reach any corner | attackers (not decisive alone, but the usual way in) |
| Shieldwall | a row of defenders on the edge, with attackers ready to close both ends | attackers (wins material) |

The win-in-one positions in the probe are random everywhere outside the structure. To make them
realistic:
- **Piece counts:** draw them from the counts in real self-play positions of the same game stage.
  The stage is roughly the total number of pieces captured.
- **Placement of the other pieces:** draw squares from the piece density of real self-play
  positions, per side, instead of uniformly.
- **Consistency:** reject positions that are decided in an unintended way. The engine finds every
  winning move, as `probe_tactics.make_probes` does already.

### 3.2 Built positions, walked backwards (the main source)

A position one move from a win is too easy. Positions 2-10 moves away are the ones the value head
needs. To get them, start from a finished structure and move pieces backwards:

1. Build a decisive structure (3.1), or take it from a real game (3.3).
2. Take back k moves, alternating the sides: slide a piece back along its line to an empty
   square. A taken-back move of the winning side opens the structure: it removes a ring attacker
   or a fort wall defender. A taken-back move of the losing side moves its pieces to where they
   might have come from. Where the reverse move could have captured, sometimes put the captured
   piece back.
3. Use k = 1-10, with more weight on small k early in training.

The engine can check that the structure is still reachable for k = 1-2, by a search of all moves.
For larger k we do not prove that the position is decisive. The game outcome labels it instead
(section 4.1), so an unproved position does no harm. It is just a position where one side has a
clear plan.

### 3.3 Real positions that come near a decisive structure

This is the most realistic source. It works as soon as self-play sometimes comes close, even
if it never finishes. Scan the self-play games, and score each position by how far it is from each
structure:
- **Encirclement:** the smallest number of empty squares the attackers must still fill to cut
  all defenders off from the edge. This is a minimum cut on the grid of non-attacker squares, with
  defender squares not cuttable. It is a fast flow computation on 121 squares.
- **Exit fort:** the smallest number of defender moves, ignoring attacker replies, that
  complete a fort. The engine's fort test can check this for 1-2 moves.
- **King trap:** the number of free sides of the king, and how many moves the king has.

Keep the positions with a small distance, for example a ring with at most 3 squares missing. Two
ways to use them:
- **Directly:** these positions are what we want, and the pool grows as play improves.
- **As a base for editing:** move 1-3 pieces to bring the distance down by one, and keep the rest
  of the board. This gives positions that are realistic around the structure and still decisive.

Training keeps only the final board of each game (`games.jsonl`), not the moves. Positions along
the way would be produced by playing new games with a checkpoint, which costs nothing in the
training run.

### 3.4 Optional: human games

Expert games from aagenielsen.dk would be the best source of realistic decisive positions:
13,640 Copenhagen games. How many end by encirclement or a fort is not known. The games could feed
3.3 directly. The site's firewall blocked my earlier download, so we would need an export or
permission. Do you know whether the game records are available?

## 4. How to train on them

### 4.1 Recommended: self-play games that start from these positions

A share f of the self-play games starts from a generated position instead of the normal start.
Everything else stays as now: the same search, the same Dirichlet noise and the same training
targets (visit shares for the policy, the game outcome for the value).

Why this and not hand-made labels:
- **The labels are honest.** The value target is what actually happened when the current network
  played the position out. Unproved positions from 3.2 and 3.3 therefore do no harm.
- **Both sides learn the structure.** The attackers learn to close the ring, and the defenders
  learn to break out early.
- **It is cheap.** These games are short, often 5-40 moves, and use the existing pipeline.

Details:
- **Settings in the run config**, a new `start_positions` section: the share f, the weight of
  each source and kind, and the range of k for 3.2. Start with f = 0.2 and lower it to about 0.05
  by iteration 200. The networks should then reach these structures in normal games.
- **Side to move:** as generated. Each structure is generated both with the winning side and
  with the defending side to move.
- **No random opening moves** for these games. The position history starts with the start
  position, so repetitions are counted from there.
- **Statistics:** exclude these games from the win rates that drive the dynamic simulation
  balancing, and from the evaluation games. Report them separately, per kind: who won, and how
  quickly. They then also measure progress (section 6).
- **Code:** in `train.py`, the self-play worker builds `Tafl(rules)` and then sets the board and
  player from the start position. The probe already does this. The replay buffer and the training
  loop do not change.

### 4.2 Also: a fixed set of solved positions in every batch

The engine solves the win-in-one positions, and some win-in-two positions, exactly. For these,
the policy target is the winning moves and the value target is +1. A small fixed set (about 10^5
positions with all symmetries) can take a small share of each training batch, for example 2 %.
This keeps the network from forgetting finishing moves that normal games seldom show.

A pre-training on this set alone, before iteration 1, is cheap but probably not worth it. The
first iterations of self-play data would wash it out.

### 4.3 An extra training target about how games end (implemented)

Every network now has an output that predicts how the game will end (`tafl.ENDINGS`), trained on
the actual ending of each self-play game, and one that predicts the plies left. This gives the
network a richer signal than win or lose, and stays within the "zero" approach, since the targets
come from self-play alone. It is like KataGo's ownership target, which tells the network which side
will own each area at the end. A stronger version would predict the final `defender_region` plane
square by square, so that the network learns where a ring will close; the final boards in
`games.jsonl` would give its targets.

## 5. Order of work

1. **Log how each game ends** in `train.py`, with counts per kind in the iteration summary.
   This shows directly whether self-play ever reaches encirclements and forts. (Done; the final
   boards are in `games.jsonl`.)
2. **Positions along the way** (for 3.3 and for checking realism): play new games with a
   checkpoint and keep their positions.
3. **Generators and distance measures** (3.1, 3.2, the encirclement cut of 3.3) in one module,
   with tests that the engine confirms each built structure.
4. **Start positions in self-play** (4.1), behind the new config section.
5. **An A/B test.** Copy a run at the same checkpoint (for example iteration 100). Train one copy
   with start positions and one without, for the same number of iterations. Compare them as below.
6. If it helps: the fixed solved set (4.2).

## 6. How to tell whether it works

- **The probe:** `probe_tactics.py` on the new checkpoints. Extend it with positions 2-5 moves
  from a decisive structure, so it measures the value head where it matters.
- **Normal games:** the attackers' share of wins from the standard start, and the share of games
  that end by encirclement or fort (printed every iteration, and stored in
  `training_history.json` under `selfplay_results.endings`).
- **Playing strength:** Elo of the two A/B runs against each other and against the same reference
  checkpoints (`evaluate_elo.py`). This is the deciding test. Better probe numbers without more
  strength mean the data taught the generator, not the game.
- **Realism:** for each source, compare piece counts and the piece density per square with
  real self-play positions of the same stage.

## 7. Risks

- **Shortcut learning.** The network may learn traits of the generator, for example a ring
  always being one piece thick. More diverse built positions and more weight on 3.3 reduce this.
  The Elo test in section 6 detects it.
- **Wasted network capacity** on positions that never occur in real play. Lowering f over time
  limits this.
- **A shift in the balance of the sides.** The attackers may win more of the generated games
  than real play would justify, or fewer. The value targets of normal games are not affected,
  but the mix changes what the value head sees. Generate each structure with both sides to move,
  and report the generated games separately.
