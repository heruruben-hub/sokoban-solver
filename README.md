# Push-optimal Sokoban solver

Entry for make.fun `sokoban-solver-01-run-20261121` — "Build the best Sokoban
solver". `main.ts` (Node 22, erasable TypeScript, no dependencies) reads each
`/input/<id>/level.txt` and writes `/output/<id>/solution.txt`.

## Approach

The rules give the level structure away: every level is a one-wide hallway
with 1–4 rooms off it, each room in a 6×6 slot with a single door, and a box
pushed out of a room through its door can never be pushed back. So in any
optimal solution every box stays in its own room, and the level optimum P*
is the sum of the rooms' optima. The solver exploits that:

1. **Decompose.** Split the level into rooms: connected floor components off
   the hallway row, each with its door cell and the hallway cell outside it.
2. **Solve each room push-optimally.** A* over pushes. A state is the sorted
   box cells plus the player's reachable region (canonicalised to its lowest
   cell, since free walking costs nothing). The heuristic is the minimum
   assignment of boxes to goals over wall-only BFS distances (Hungarian, by
   brute-force permutation over at most 4 goals) — admissible, because each
   push moves one box one cell and can lower that sum by at most one. States
   are pruned by static dead squares, corner deadlocks, and 2×2 freeze
   blocks. A solved state is only accepted when the player can still walk
   out through the door. Boxes are never allowed onto the hallway entry
   cell, so they can never leave their room.
3. **Stitch.** Visit rooms nearest-first from the player's current cell and
   join the rooms' move strings with hallway walks, which add zero pushes.
   The final move string is replayed on the whole level as a safety net; if
   anything is off, an empty (valid, zero-point) answer is emitted instead.

## Determinism

All orders are fixed (sorted cells, fixed neighbour order, insertion-order
heap tie-breaks) and all budgets are counted in search expansions, never in
wall-clock time — no `Date.now`, no randomness — so two runs produce
byte-identical outputs.

## Results (practice set)

| instance | pushes | P\* | points |
|----------|--------|----|--------|
| i001 | 13 | 13 | 1000 |
| i002 | 16 | 16 | 1000 |
| i003 | 29 | 29 | 1000 |
| i004 | 47 | 47 | 1000 |
| i005 | 60 | 60 | 1000 |

Practice score **1,000,000 / 1,000,000**, total runtime ~0.4 s (limit 60 s).
The same solver also scores 1,000,000 on three freshly drawn unseen sets
(15 instances) generated with the kit's own generator, so the approach
generalises to the scored set rather than fitting the practice levels.

## Local run

```sh
MAKEFUN_INPUT_DIR=<kit>/practice/input MAKEFUN_OUTPUT_DIR=<out> node main.ts
```
