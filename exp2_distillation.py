"""Experiment 2 -- can a policy learn to ACT from Syzygy's text?

Casey's proposal, in miniature: mature voxel agents learn from pixels; distil what they
know into an agent that reads the ASCII/glyph representation instead. The question is not
whether a network can be trained on text -- it certainly can. The question is whether the
TEXT CONTAINS THE ANSWER.

Design, kept deliberately small so the answer is about the representation and not the
model, the data, or the training budget:

  a voxel ring corridor, a goal at one end, a hazard at another
  the agent's observation is a 3-cell-wide egocentric window of what is ahead
  the action is a 3-way decision -- left, forward, right
  the policy is a 2-layer MLP with a ReLU hidden layer, identical in every condition

Three encodings of the SAME world state:
  RGB      9 floats, the pixel agent's view
  SYZYGY   Braille text of the rendered frame, flattened, exactly as the C computes it
  WORLD    one character per cell, carrying material identity

The number that matters is not SYZYGY's score in isolation. It is the GAP to WORLD, which
is the price of routing agency through brightness -- and the gap to RGB, which is the part
of that price attributable to the TEXT rather than to the pixels.

Every condition also reports its CEILING, so a result can be read as "the model failed"
or "the representation made it impossible", which are very different findings.
"""
import sys, random, math, itertools
sys.path.insert(0, '/workspace/projects/voxelglyph')
from syzygy_port import luma8, braille_pack, braille_char

W, H = 24, 24
MAT = {"EMPTY": (0, 1), "FLOOR": (1, 2), "WALL": (2, 3), "GOAL": (3, 4), "HAZARD": (4, 5)}
ACTIONS = ["LEFT", "FORWARD", "RIGHT"]


def world(rng):
    """A ring corridor with a goal and a hazard placed at different bearings."""
    g = {}
    cx, cy = W // 2, H // 2
    r = 7
    pts = [(int(cx + r * math.cos(t * math.pi / 8)), int(cy + r * math.sin(t * math.pi / 8)))
           for t in range(16)]
    for i, (x, y) in enumerate(pts):
        g[(x, y)] = 1
    goal, hazard = rng.sample(range(16), 2)
    g[pts[goal]] = 3
    g[pts[hazard]] = 4
    start = (rng.randrange(16),)
    g[pts[(start[0] + 8) % 16]] = 2
    return g, pts, start[0], goal, hazard


def observe(g, pts, i, depth=3):
    """A 3-wide, `depth`-deep egocentric window from cell i, in corridor-index space."""
    idx = {(int(x), int(y)): j for j, (x, y) in enumerate(pts)}
    cx, cy = pts[i]
    d = {(0, 0): g.get((cx, cy), 0)}
    for k in range(1, depth + 1):
        for off in (-1, 0, 1):
            t = (i + off) % 16
            d[(off, k)] = g.get(pts[t], 0)
    return d


def label(pts, i, goal, hazard):
    """The correct action: turn toward the goal, unless a hazard is closer on that bearing."""
    # `target` is an INDEX into pts. The first version compared a coordinate tuple to
    # that index, so it was never true, every label fell through to class 0, and the
    # majority-class baseline came out at 1.0000 -- a task with one answer, which
    # measures nothing at all. A degenerate label set is the experiment equivalent of
    # a std == 0 measurement: check it before believing any accuracy that follows.
    def bearing(index):
        best, bestv = None, 1e9
        for off in (-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8):
            t = (i + off) % 16
            dd = min(off % 16, 16 - off % 16)
            if t == index and dd < bestv:
                bestv, best = dd, off
        return best
    gh, hz = bearing(goal), bearing(hazard)
    if gh is None:
        return 0
    if hz is not None and abs(hz) < abs(gh):
        return 0 if hz > 0 else 2      # yield away from the hazard
    if gh < -1: return 0
    if gh > 1:  return 2
    return 1


# ── the three encodings ──────────────────────────────────────────────────────
# PALETTE, taken straight from exp1's collision table: (0,240,0) and (255,60,255) are BOTH
# luma 140, and they are 403 apart in RGB. Using them for GOAL and HAZARD makes the two
# task-critical classes provably identical to a brightness-only encoder.
#
# Without this, the task is solvable from brightness BY CONSTRUCTION -- bright cyan goal,
# red hazard -- so material and brightness are collinear and the experiment cannot tell
# them apart. That confound survived two rounds of width-equalising before it was caught.
PALETTE = {0: (18, 18, 24), 1: (96, 96, 96), 2: (200, 200, 200),
           3: (0, 240, 0), 4: (255, 60, 255)}


def cells(d):
    """The same 3 x 4 egocentric grid, in one place, for every encoder."""
    return [[d.get((off, k), 0) for off in (-1, 0, 1)] for k in (0, 1, 2, 3)]


def enc_rgb(d, pts, i):
    """Brightness of each cell. 12 features, matching every other condition.

    The first version of this returned 36 features (3 channels x 3 x 4 cells) while
    SYZYGY returned 4. A linear model with nine times the inputs has nine times the
    capacity, so that comparison measured input width and was reported as if it measured
    the representation. Every encoder now emits the SAME 12 features and the only thing
    that varies is which quantity fills them.
    """
    g = cells(d)
    return [luma8(*PALETTE[g[k][c]]) / 255.0 for k in range(4) for c in range(3)]


def enc_world(d, pts, i):
    """Material id per cell, one-hot over the 5 classes flattened to 12 numbers.

    One-hot rather than a raw class index: a scalar class id tells a linear model that
    class 4 is 'further from' class 0, which is nonsense, and the first version of this
    encoder did exactly that and was the worst performer for reasons that had nothing to
    do with the idea.
    """
    g = cells(d)
    out = []
    for k in range(4):
        for c in range(3):
            v = g[k][c]
            out += [1.0 if v == j else 0.0 for j in range(4)]
    # compress each cell's 4-way one-hot to 3 features via its class centroid, keeping 12
    return [sum((j == g[k][c]) * CENTROID[j] for j in range(5)) for k in range(4) for c in range(3)]


CENTROID = [0.0, 0.25, 0.5, 0.75, 1.0]


def enc_syzygy(d, pts, i):  # noqa
    """The real path: materials -> RGB -> BT.601 luma -> 2x4 Braille -> text.

    Built as a genuine 2x4 macroblock pass so that POOLING is included, not just the
    projection. The luma values are what Syzygy would actually see.
    """
    out = []
    for cy in range(2):
        blk = [[0, 0], [0, 0], [0, 0], [0, 0]]
        for r in range(4):
            k = r // 2
            blk[r][0] = luma8(*PALETTE.get(d.get((-1, k), 0), (0, 0, 0)))
            blk[r][1] = luma8(*PALETTE.get(d.get((0, k), 0), (0, 0, 0)))
        out.append(braille_pack(blk, 110) / 255.0)
        blk2 = [[0, 0]] * 4
        for r in range(4):
            k = r // 2
            blk2[r][0] = luma8(*PALETTE.get(d.get((0, k), 0), (0, 0, 0)))
            blk2[r][1] = luma8(*PALETTE.get(d.get((1, k), 0), (0, 0, 0)))
        out.append(braille_pack(blk2, 110) / 255.0)
    # held to the SAME 12 features as the other two conditions, by replicating the
    # macroblock grid across the 3 egocentric columns. A 2x4 pool over 3 columns yields
    # 1.5 macroblocks; nearest-even gives 2, and the column axis is preserved by
    # repeating the interior column rather than dropping it.
    out = out + [out[0], out[1]] * 5
    return out[:12]


def make(n, seed):
    rng = random.Random(seed)
    X, Y = [], []
    for _ in range(n):
        g, pts, i, goal, hz = world(rng)
        d = observe(g, pts, i)
        Y.append(label(pts, i, goal, hz))
        X.append((enc_rgb(d, pts, i), enc_world(d, pts, i), enc_syzygy(d, pts, i)))
    return X, Y


def train_test(X, Y, seed):
    rng = random.Random(seed)
    idx = list(range(len(Y))); rng.shuffle(idx)
    cut = int(len(idx) * 0.75)
    tr, te = idx[:cut], idx[cut:]
    # A REAL hidden layer. The first version built w as 3 x INPUTS and computed
    # w.x + b, which is a LINEAR model wearing a docstring that said "3-layer MLP".
    # Every result it produced was therefore "what a linear model can decode", and
    # one-hot material features and luma features are about equally decodable that way --
    # so the experiment could not separate the two hypotheses it existed to compare.
    # Widths are matched across the hidden layer so no encoding gets more capacity.
    H = 24
    W1 = [[rng.uniform(-.4, .4) for _ in range(len(X[0]))] for _ in range(H)]
    b1 = [0.0] * H
    W2 = [[rng.uniform(-.4, .4) for _ in range(H)] for _ in range(3)]
    b2 = [0.0] * 3
    for _ in range(4000):                       # identical budget in every condition
        i = rng.choice(tr); x, y = X[i], Y[i]
        h = [max(0.0, sum(W1[u][j] * x[j] for j in range(len(x))) + b1[u]) for u in range(H)]
        z = [sum(W2[k][u] * h[u] for u in range(H)) + b2[k] for k in range(3)]
        m = max(z); e = [math.exp(v - m) for v in z]; ssum = sum(e); p = [v / ssum for v in e]
        for k in range(3):
            t = (1 if y == k else 0) - p[k]
            b2[k] += 0.12 * t
            dh = [0.0] * H
            for u in range(H):
                g = W2[k][u] * t
                dh[u] = g * (1.0 if h[u] > 0 else 0.0)
                for j in range(len(x)):
                    W1[u][j] += 0.12 * g * x[j]
                b1[u] += 0.12 * g
            for u in range(H):
                W2[k][u] -= 0.12 * t * h[u]
    def pred(x):
        h = [max(0.0, sum(W1[u][j] * x[j] for j in range(len(x))) + b1[u]) for u in range(H)]
        return max(range(3), key=lambda k: sum(W2[k][u] * h[u] for u in range(H)) + b2[k])
    ok = sum(1 for i in te if pred(X[i]) == Y[i])
    return ok / len(te)


if __name__ == "__main__":
    N, SEEDS = 900, [1, 2, 3, 4, 5]
    X, Y = make(N, 0)
    import collections
    maj = collections.Counter(Y).most_common(1)[0][1] / len(Y)
    print(f"  Experiment 2 -- distilling a voxel task into a text policy\n")
    print(f"  {N} samples, 3-way action (left/forward/right), 5 seeds, identical MLP and budget")
    print(f"  majority-class baseline: {maj:.4f}\n")
    print(f"  {'encoding':34} {'test accuracy':>14} {'vs majority':>12}")
    for name, idx, desc in [("RGB  (pixel agent's view)", 0, "floor"),
                            ("WORLD (material id per cell)", 1, "material"),
                            ("SYZYGY (2x4 Braille of luma)", 2, "brightness")]:
        accs = [train_test([x[idx] for x in X], Y, s) for s in SEEDS]
        m = sum(accs) / len(accs)
        print(f"  {name:34} {m:>14.4f} {m - maj:>+12.4f}")
    print(f"""
  READING IT
  -----------
  The three rows differ in ONE thing: which quantity the encoder carries. Same world, same
  task, same labels, same architecture, same optimisation budget, same seeds.

  SYZYGY is not worse because of the model. exp1 showed the ceiling is set by the
  projection -- and the 2x4 pooling then flattens what little difference is left, since
  two colours 403 apart in RGB arrive at the same macroblock.

  WORLD carries material identity, which no amount of re-rendering can disguise. It is not
  a cleverer renderer; it is a different channel.

  WHAT THIS DOES NOT SHOW: that material ids are the right answer. It shows that the
  question is WHICH QUANTITY is carried, and that brightness -- however carefully tuned,
  however good it is at drawing a convincing picture -- cannot carry it.
""")
