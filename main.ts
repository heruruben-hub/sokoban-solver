// Push-optimal Sokoban solver for the make.fun "sokoban-solver-01" bounty.
//
// Idea (exploits the level structure given in the rules): every level is a
// one-wide hallway with 1-4 rooms off it, each room in a 6x6 slot with a
// single door. A box pushed out of a room through its door can never be
// pushed back, so in any optimal solution every box stays in its own room
// and the level optimum is the sum of the rooms' optima. This solver:
//
//   1. splits the level into its rooms (connected floor components off the
//      hallway row), each with its door cell and the hallway cell outside it;
//   2. solves each room exactly and push-optimally with A* over pushes:
//      a state is the sorted box cells plus the player's reachable region
//      (canonicalised to its lowest cell), the heuristic is the minimum
//      assignment (Hungarian, by brute-force permutation) of boxes to goals
//      over wall-only BFS distances, and states are pruned by static dead
//      squares, corner deadlocks and 2x2 freeze blocks;
//   3. stitches the rooms together with hallway walks that add zero pushes
//      (rooms are visited nearest-first from the player's current cell),
//      and verifies the final move string by replaying it on the whole level.
//
// Determinism: every order is fixed (sorted cells, fixed neighbour order,
// insertion-order heap tie-breaks) and all budgets are counted in search
// expansions, never in wall-clock time, so two runs are byte-identical.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

// Directions: 0 = up, 1 = right, 2 = down, 3 = left.
const DX = [0, 1, 0, -1];
const DY = [-1, 0, 1, 0];
const CH = ['u', 'r', 'd', 'l'];
const OPP = [2, 3, 0, 1];

type Level = {
  w: number;
  h: number;
  wall: Uint8Array; // 1 = wall
  goal: Uint8Array; // 1 = goal square
  box: Uint8Array; // 1 = box here (initial)
  player: number; // cell index of the player
};

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goal = new Uint8Array(w * h);
  const box = new Uint8Array(w * h);
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const i = y * w + x;
      const c = row[x];
      if (c === '#') continue;
      wall[i] = 0;
      if (c === '.' || c === '*' || c === '+') goal[i] = 1;
      if (c === '$' || c === '*') box[i] = 1;
      if (c === '@' || c === '+') player = i;
    }
  });
  return { w, h, wall, goal, box, player };
}

// Breadth-first walk on the whole board. `blocked(i)` says whether cell i
// cannot be stepped on. Returns the direction list from `from` to `to`, or
// null when unreachable. Neighbour order is fixed, so the path is unique.
function walkPath(
  w: number,
  h: number,
  from: number,
  to: number,
  blocked: (i: number) => boolean,
): number[] | null {
  if (from === to) return [];
  const n = w * h;
  const prev = new Int32Array(n).fill(-2);
  const prevDir = new Int8Array(n);
  const queue = new Int32Array(n);
  let qh = 0;
  let qt = 0;
  queue[qt++] = from;
  prev[from] = -1;
  while (qh < qt) {
    const cur = queue[qh++];
    const cx = cur % w;
    const cy = Math.floor(cur / w);
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d];
      const ny = cy + DY[d];
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const ni = ny * w + nx;
      if (prev[ni] !== -2 || blocked(ni)) continue;
      prev[ni] = cur;
      prevDir[ni] = d;
      if (ni === to) {
        const out: number[] = [];
        let c = ni;
        while (c !== from) {
          out.push(prevDir[c]);
          c = prev[c];
        }
        out.reverse();
        return out;
      }
      queue[qt++] = ni;
    }
  }
  return null;
}

// Walking distance (in steps) on the whole board, or -1 when unreachable.
function walkDist(
  w: number,
  h: number,
  from: number,
  to: number,
  blocked: (i: number) => boolean,
): number {
  const p = walkPath(w, h, from, to, blocked);
  return p === null ? -1 : p.length;
}

// ---------------------------------------------------------------------------
// Room decomposition
// ---------------------------------------------------------------------------

type Room = {
  cells: number[]; // global cell indices of the room floor, door cell included
  goals: number[]; // global indices of goals inside the room
  boxes: number[]; // global indices of boxes inside the room (initial)
  door: number; // global index of the door cell (room side)
  entry: number; // global index of the hallway cell outside the door
};

function findRooms(level: Level): Room[] {
  const { w, h, wall, goal, box } = level;
  const hy = Math.floor(level.player / w);
  const seen = new Uint8Array(w * h);
  const rooms: Room[] = [];
  for (let start = 0; start < w * h; start++) {
    if (wall[start] || seen[start]) continue;
    if (Math.floor(start / w) === hy) continue; // hallway row is not a room
    // Flood-fill one connected component of non-hallway floor.
    const comp: number[] = [];
    const stack = [start];
    seen[start] = 1;
    while (stack.length > 0) {
      const cur = stack.pop() as number;
      comp.push(cur);
      const cx = cur % w;
      const cy = Math.floor(cur / w);
      for (let d = 0; d < 4; d++) {
        const nx = cx + DX[d];
        const ny = cy + DY[d];
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const ni = ny * w + nx;
        if (wall[ni] || seen[ni] || Math.floor(ni / w) === hy) continue;
        seen[ni] = 1;
        stack.push(ni);
      }
    }
    comp.sort((a, b) => a - b);
    // The door is the component cell touching the hallway row.
    let door = -1;
    let entry = -1;
    for (const c of comp) {
      const cy = Math.floor(c / w);
      const ny = cy < hy ? cy + 1 : cy - 1;
      if (ny !== hy) continue;
      const ni = c + (cy < hy ? w : -w);
      if (!wall[ni]) {
        door = c;
        entry = ni;
        break;
      }
    }
    if (door < 0) continue; // a stray pocket with no door; not a room
    rooms.push({
      cells: comp,
      goals: comp.filter((c) => goal[c] === 1),
      boxes: comp.filter((c) => box[c] === 1),
      door,
      entry,
    });
  }
  return rooms;
}

// ---------------------------------------------------------------------------
// Push-optimal room solver (A* over pushes)
// ---------------------------------------------------------------------------

type RoomProblem = {
  n: number; // local cells (room floor + the hallway entry cell)
  neigh: Int32Array; // n * 4, local neighbour id or -1
  dist: Int32Array; // n * n wall-only distances, -1 = unreachable
  isGoal: Uint8Array;
  dead: Uint8Array; // static dead squares (non-goal, cannot reach a goal / corner)
  blocks: { cells: number[]; hasGoal: boolean }[]; // 2x2 freeze blocks
  goals: number[]; // local goal ids
  boxes0: number[]; // local initial box ids, sorted
  start: number; // local id of the entry cell (player starts here)
  door: number; // local id of the door cell
  entry: number; // local id of the entry cell
  perms: number[][]; // permutations of goal ids (assignment orders)
  k: number; // number of boxes
};

function buildRoomProblem(room: Room, level: Level): RoomProblem {
  // Local board: the room's floor plus the single hallway cell outside the
  // door. Everything else is a wall for this sub-problem.
  const all = [...room.cells];
  if (!all.includes(room.entry)) all.push(room.entry);
  all.sort((a, b) => a - b);
  const local = new Map<number, number>();
  all.forEach((g, i) => local.set(g, i));
  const n = all.length;
  const neigh = new Int32Array(n * 4).fill(-1);
  for (let i = 0; i < n; i++) {
    const g = all[i];
    const gx = g % level.w;
    const gy = Math.floor(g / level.w);
    for (let d = 0; d < 4; d++) {
      const nx = gx + DX[d];
      const ny = gy + DY[d];
      if (nx < 0 || nx >= level.w || ny < 0 || ny >= level.h) continue;
      const ni = ny * level.w + nx;
      const li = local.get(ni);
      if (li !== undefined) neigh[i * 4 + d] = li;
    }
  }
  // Wall-only BFS distances between every pair of local cells.
  const dist = new Int32Array(n * n).fill(-1);
  const queue = new Int32Array(n);
  for (let s = 0; s < n; s++) {
    let qh = 0;
    let qt = 0;
    queue[qt++] = s;
    dist[s * n + s] = 0;
    while (qh < qt) {
      const cur = queue[qh++];
      const cd = dist[s * n + cur];
      for (let d = 0; d < 4; d++) {
        const ni = neigh[cur * 4 + d];
        if (ni < 0 || dist[s * n + ni] >= 0) continue;
        dist[s * n + ni] = cd + 1;
        queue[qt++] = ni;
      }
    }
  }
  const isGoal = new Uint8Array(n);
  const goals: number[] = [];
  for (const g of room.goals) {
    const li = local.get(g) as number;
    isGoal[li] = 1;
    goals.push(li);
  }
  goals.sort((a, b) => a - b);
  // Static dead squares: a non-goal cell from which no goal is reachable
  // over floor alone, or a corner (two perpendicular neighbouring walls).
  const dead = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (isGoal[i]) continue;
    let reach = false;
    for (const g of goals) if (dist[i * n + g] >= 0) { reach = true; break; }
    if (!reach) { dead[i] = 1; continue; }
    const up = neigh[i * 4 + 0] < 0;
    const right = neigh[i * 4 + 1] < 0;
    const down = neigh[i * 4 + 2] < 0;
    const left = neigh[i * 4 + 3] < 0;
    if ((up && left) || (up && right) || (down && left) || (down && right)) dead[i] = 1;
  }
  // 2x2 freeze blocks: four cells that are all wall-or-box with no goal in
  // them freeze any box they hold. Precompute from global coordinates.
  const blocks: { cells: number[]; hasGoal: boolean }[] = [];
  for (let i = 0; i < n; i++) {
    const gx = all[i] % level.w;
    const gy = Math.floor(all[i] / level.w);
    // Only emit the block once, at its top-left cell.
    if (gx + 1 >= level.w || gy + 1 >= level.h) continue;
    const cells = [
      i,
      local.get(gy * level.w + gx + 1) ?? -1,
      local.get((gy + 1) * level.w + gx) ?? -1,
      local.get((gy + 1) * level.w + gx + 1) ?? -1,
    ];
    if (cells.every((c) => c < 0)) continue;
    blocks.push({ cells, hasGoal: cells.some((c) => c >= 0 && isGoal[c] === 1) });
  }
  // All permutations of the goals, for the minimum-assignment heuristic.
  const perms: number[][] = [];
  const used = new Array(goals.length).fill(false);
  const cur: number[] = [];
  const genPerm = (): void => {
    if (cur.length === goals.length) { perms.push([...cur]); return; }
    for (let i = 0; i < goals.length; i++) {
      if (used[i]) continue;
      used[i] = true;
      cur.push(goals[i]);
      genPerm();
      cur.pop();
      used[i] = false;
    }
  };
  genPerm();
  const boxes0 = room.boxes.map((g) => local.get(g) as number).sort((a, b) => a - b);
  return {
    n,
    neigh,
    dist,
    isGoal,
    dead,
    blocks,
    goals,
    boxes0,
    start: local.get(room.entry) as number,
    door: local.get(room.door) as number,
    entry: local.get(room.entry) as number,
    perms,
    k: room.boxes.length,
  };
}

// Minimum-assignment lower bound on remaining pushes: the cheapest way to
// match every box to a distinct goal over wall-only distances. Each push
// moves one box one cell, so this sum can drop by at most one per push.
function heuristic(prob: RoomProblem, boxes: number[]): number {
  let best = Infinity;
  for (const perm of prob.perms) {
    let sum = 0;
    for (let i = 0; i < prob.k; i++) {
      const d = prob.dist[boxes[i] * prob.n + perm[i]];
      if (d < 0) { sum = Infinity; break; }
      sum += d;
    }
    if (sum < best) best = sum;
  }
  return best;
}

// True when any box sits on a static dead square or inside a frozen 2x2.
function isDead(prob: RoomProblem, boxMask: Uint8Array): boolean {
  for (let i = 0; i < prob.n; i++) {
    if (boxMask[i] && prob.dead[i]) return true;
  }
  for (const b of prob.blocks) {
    if (b.hasGoal) continue;
    let allBlocked = true;
    let anyBox = false;
    for (const c of b.cells) {
      if (c < 0) continue;
      if (boxMask[c]) { anyBox = true; continue; }
      allBlocked = false;
    }
    if (allBlocked && anyBox) return true;
  }
  return false;
}

// Player's reachable region (local cells) from `from`, boxes as obstacles.
function region(prob: RoomProblem, boxMask: Uint8Array, from: number): Uint8Array {
  const seen = new Uint8Array(prob.n);
  const queue = [from];
  seen[from] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const cur = queue[qi];
    for (let d = 0; d < 4; d++) {
      const ni = prob.neigh[cur * 4 + d];
      if (ni < 0 || seen[ni] || boxMask[ni]) continue;
      seen[ni] = 1;
      queue.push(ni);
    }
  }
  return seen;
}

// Walk inside the room (local ids), boxes as obstacles. Fixed neighbour
// order, so the path is deterministic.
function roomWalk(prob: RoomProblem, boxMask: Uint8Array, from: number, to: number): number[] | null {
  if (from === to) return [];
  const prev = new Int32Array(prob.n).fill(-2);
  const prevDir = new Int8Array(prob.n);
  const queue = [from];
  prev[from] = -1;
  for (let qi = 0; qi < queue.length; qi++) {
    const cur = queue[qi];
    for (let d = 0; d < 4; d++) {
      const ni = prob.neigh[cur * 4 + d];
      if (ni < 0 || prev[ni] !== -2 || boxMask[ni]) continue;
      prev[ni] = cur;
      prevDir[ni] = d;
      if (ni === to) {
        const out: number[] = [];
        let c = ni;
        while (c !== from) {
          out.push(prevDir[c]);
          c = prev[c];
        }
        out.reverse();
        return out;
      }
      queue.push(ni);
    }
  }
  return null;
}

// Boxes are stored sorted as a base-64 code (local ids are < 64; a missing
// fourth box is coded 63, which no real cell uses in a 6x6 slot).
function encodeBoxes(boxes: number[]): number {
  let code = 63;
  for (let i = boxes.length - 1; i >= 0; i--) code = code * 64 + boxes[i];
  return code;
}
function decodeBoxes(code: number, k: number): number[] {
  const out: number[] = [];
  let c = code;
  for (let i = 0; i < k; i++) {
    out.push(c % 64);
    c = Math.floor(c / 64);
  }
  return out;
}

type RoomResult = { moves: string; pushes: number; expansions: number };

// A* over pushes. Returns the move string from the entry cell (player
// starts on the hallway cell outside the door) back to the entry cell, with
// every box on a goal, in the fewest pushes possible; null when the budget
// runs out or the room is unsolvable.
function solveRoom(prob: RoomProblem, maxExpansions: number): RoomResult | null {
  const solved0 = prob.boxes0.every((b) => prob.isGoal[b] === 1);
  if (solved0) return { moves: '', pushes: 0, expansions: 0 };
  for (const b of prob.boxes0) if (prob.dead[b]) return null;
  const h0 = heuristic(prob, prob.boxes0);
  if (!Number.isFinite(h0)) return null;

  // Node storage (parallel arrays).
  const codeArr: number[] = [];
  const playerArr: number[] = [];
  const parentArr: number[] = [];
  const pushBoxArr: number[] = [];
  const pushDirArr: number[] = [];
  const gArr: number[] = [];
  const addNode = (
    code: number, player: number, parent: number, pushBox: number, pushDir: number, g: number,
  ): number => {
    codeArr.push(code);
    playerArr.push(player);
    parentArr.push(parent);
    pushBoxArr.push(pushBox);
    pushDirArr.push(pushDir);
    gArr.push(g);
    return codeArr.length - 1;
  };
  const root = addNode(encodeBoxes(prob.boxes0), prob.start, -1, -1, -1, 0);

  // Binary heap ordered by (f, -g, insertion order). No clock, no random.
  type HeapItem = { node: number; f: number; g: number; seq: number };
  const heap: HeapItem[] = [];
  let seq = 0;
  const heapLess = (a: HeapItem, b: HeapItem): boolean =>
    a.f !== b.f ? a.f < b.f : a.g !== b.g ? a.g > b.g : a.seq < b.seq;
  const heapPush = (item: HeapItem): void => {
    heap.push(item);
    let i = heap.length - 1;
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!heapLess(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const heapPop = (): HeapItem => {
    const top = heap[0];
    const last = heap.pop() as HeapItem;
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && heapLess(heap[l], heap[m])) m = l;
        if (r < heap.length && heapLess(heap[r], heap[m])) m = r;
        if (m === i) break;
        [heap[i], heap[m]] = [heap[m], heap[i]];
        i = m;
      }
    }
    return top;
  };

  // Best known g for (boxes, canonical player) -> state key.
  const best = new Map<number, number>();
  const startMask = new Uint8Array(prob.n);
  for (const b of prob.boxes0) startMask[b] = 1;
  const startRegion = region(prob, startMask, prob.start);
  let canon = prob.n;
  for (let i = 0; i < prob.n; i++) if (startRegion[i]) { canon = i; break; }
  best.set(encodeBoxes(prob.boxes0) * 64 + canon, 0);
  heapPush({ node: root, f: h0, g: 0, seq: seq++ });

  const boxMask = new Uint8Array(prob.n);
  let expansions = 0;
  let goalNode = -1;
  while (heap.length > 0) {
    const item = heapPop();
    const node = item.node;
    if (item.g !== gArr[node]) continue; // a better copy of this state exists
    expansions++;
    if (expansions > maxExpansions) return null;
    const boxes = decodeBoxes(codeArr[node], prob.k);
    const solved = boxes.every((b) => prob.isGoal[b] === 1);
    if (solved) {
      // Accept only when the player can still walk out through the door;
      // a solved-but-trapped state can never lead to an optimal solution
      // (leaving it costs extra pushes), so it is a dead end, not a goal.
      boxMask.fill(0);
      for (const b of boxes) boxMask[b] = 1;
      const reg = region(prob, boxMask, playerArr[node]);
      if (reg[prob.door]) { goalNode = node; break; }
      continue;
    }
    boxMask.fill(0);
    for (const b of boxes) boxMask[b] = 1;
    const reach = region(prob, boxMask, playerArr[node]);
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let d = 0; d < 4; d++) {
        const dest = prob.neigh[b * 4 + d];
        if (dest < 0 || boxMask[dest]) continue;
        if (dest === prob.entry) continue; // a box may never leave the room
        const behind = prob.neigh[b * 4 + OPP[d]];
        if (behind < 0 || !reach[behind]) continue;
        const nextBoxes = [...boxes];
        nextBoxes[bi] = dest;
        nextBoxes.sort((x, y) => x - y);
        const nextMask = new Uint8Array(prob.n);
        for (const nb of nextBoxes) nextMask[nb] = 1;
        if (isDead(prob, nextMask)) continue;
        const h = heuristic(prob, nextBoxes);
        if (!Number.isFinite(h)) continue;
        const nextRegion = region(prob, nextMask, b);
        let nextCanon = prob.n;
        for (let i = 0; i < prob.n; i++) if (nextRegion[i]) { nextCanon = i; break; }
        const nextCode = encodeBoxes(nextBoxes);
        const key = nextCode * 64 + nextCanon;
        const g2 = item.g + 1;
        const prevBest = best.get(key);
        if (prevBest !== undefined && prevBest <= g2) continue;
        best.set(key, g2);
        const nn = addNode(nextCode, b, node, b, d, g2);
        heapPush({ node: nn, f: g2 + h, g: g2, seq: seq++ });
      }
    }
  }
  if (goalNode < 0) return null;

  // Reconstruct the push list, then turn it into walks + pushes.
  const pushes: { box: number; dir: number }[] = [];
  let cur = goalNode;
  while (parentArr[cur] >= 0) {
    pushes.push({ box: pushBoxArr[cur], dir: pushDirArr[cur] });
    cur = parentArr[cur];
  }
  pushes.reverse();
  const moves: number[] = [];
  let player = prob.start;
  let curBoxes = [...prob.boxes0];
  const mask = new Uint8Array(prob.n);
  for (const p of pushes) {
    for (let i = 0; i < prob.n; i++) mask[i] = 0;
    for (const b of curBoxes) mask[b] = 1;
    const behind = prob.neigh[p.box * 4 + OPP[p.dir]];
    const walk = roomWalk(prob, mask, player, behind);
    if (walk === null) return null; // cannot happen: reachability was checked
    moves.push(...walk);
    moves.push(p.dir);
    const dest = prob.neigh[p.box * 4 + p.dir];
    curBoxes = curBoxes.map((b) => (b === p.box ? dest : b)).sort((a, b) => a - b);
    player = p.box;
  }
  for (let i = 0; i < prob.n; i++) mask[i] = 0;
  for (const b of curBoxes) mask[b] = 1;
  const exit = roomWalk(prob, mask, player, prob.door);
  if (exit === null) return null;
  moves.push(...exit);
  // Step out of the door back onto the hallway entry cell.
  const outDir = prob.neigh[prob.door * 4 + 0] === prob.entry ? 0
    : prob.neigh[prob.door * 4 + 1] === prob.entry ? 1
    : prob.neigh[prob.door * 4 + 2] === prob.entry ? 2 : 3;
  moves.push(outDir);
  return { moves: moves.map((d) => CH[d]).join(''), pushes: pushes.length, expansions };
}

// ---------------------------------------------------------------------------
// Whole level
// ---------------------------------------------------------------------------

function solveLevel(text: string): string {
  const level = parseLevel(text);
  const rooms = findRooms(level);
  if (rooms.length === 0) return '';
  // Live box positions, updated as rooms are solved.
  const boxNow = new Uint8Array(level.w * level.h);
  for (let i = 0; i < boxNow.length; i++) boxNow[i] = level.box[i];
  let player = level.player;
  const moves: string[] = [];
  const remaining = [...rooms];
  while (remaining.length > 0) {
    // Visit the nearest unvisited room first (hallway walking distance);
    // ties break by door position, so the order is deterministic.
    let bestIdx = -1;
    let bestKey: number[] = [];
    for (let i = 0; i < remaining.length; i++) {
      const r = remaining[i];
      const blocked = (c: number): boolean => level.wall[c] === 1 || boxNow[c] === 1;
      const d = walkDist(level.w, level.h, player, r.entry, blocked);
      const key = [d < 0 ? 1e9 : d, r.door];
      if (bestIdx < 0 || key[0] < bestKey[0] || (key[0] === bestKey[0] && key[1] < bestKey[1])) {
        bestIdx = i;
        bestKey = key;
      }
    }
    const room = remaining.splice(bestIdx, 1)[0];
    const prob = buildRoomProblem(room, level);
    let res = solveRoom(prob, 1500000);
    if (res === null) res = solveRoom(prob, 6000000);
    if (res === null) return ''; // give up on this level rather than emit junk
    // Walk to the hallway cell outside the door, then play the room moves.
    const blocked = (c: number): boolean => level.wall[c] === 1 || boxNow[c] === 1;
    const walk = walkPath(level.w, level.h, player, room.entry, blocked);
    if (walk === null) return '';
    moves.push(walk.map((d) => CH[d]).join(''));
    moves.push(res.moves);
    // Update the world: the room's boxes are now on its goals, and the
    // player stands on the hallway cell outside the door.
    for (const b of room.boxes) boxNow[b] = 0;
    for (const g of room.goals) boxNow[g] = 1;
    player = room.entry;
  }
  const allMoves = moves.join('');
  // Safety net: replay the whole move string on the level; if anything is
  // off, emit an empty (valid, zero-point) answer instead of an illegal one.
  if (!replayOk(level, allMoves)) return '';
  return allMoves;
}

function replayOk(level: Level, moves: string): boolean {
  const box = new Uint8Array(level.box);
  let player = level.player;
  for (const ch of moves) {
    const d = CH.indexOf(ch);
    if (d < 0) return false;
    const next = player + DY[d] * level.w + DX[d];
    if (level.wall[next]) return false;
    if (box[next]) {
      const beyond = next + DY[d] * level.w + DX[d];
      if (level.wall[beyond] || box[beyond]) return false;
      box[next] = 0;
      box[beyond] = 1;
    }
    player = next;
  }
  for (let i = 0; i < box.length; i++) {
    if (box[i] && !level.goal[i]) return false;
  }
  return true;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

const IN_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

for (const id of readdirSync(IN_DIR).sort()) {
  let moves = '';
  try {
    moves = solveLevel(readFileSync(join(IN_DIR, id, 'level.txt'), 'utf8'));
  } catch {
    moves = '';
  }
  mkdirSync(join(OUT_DIR, id), { recursive: true });
  writeFileSync(join(OUT_DIR, id, 'solution.txt'), `${moves}\n`);
}
