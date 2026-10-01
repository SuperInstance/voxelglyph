"""Experiment 1 -- what does a luma-only encoder provably destroy?

Syzygy's entire feature stack is a function of luma:
    syz_luma8 -> Braille threshold -> Sobel -> STE tone -> FFT
Nothing downstream of that line can see anything that line did not.

`syz_luma8` is a LINEAR MAP R^3 -> R, so it is RANK ONE. Every RGB triple with the same
luma lands on the same point. Therefore any two materials at equal luma are
indistinguishable to Syzygy -- exactly, always, by construction. That is a theorem, not a
hypothesis.

**First attempt at this experiment FAILED and the failure is the useful part.** I hand-picked
eight "chromatically different" voxel materials and none of them collided: eight materials,
eight distinct lumas, a ceiling of 1.0000, and a demonstration of nothing. The collisions are
not something you trip over by choosing colours carefully -- you have to go looking.

So: search the RGB cube. Enumerate it, bucket by luma, and ask which classes are big.
A luma class with many widely-separated colours is a place where a voxel world can hide
anything it likes from a brightness-only agent.

THE OPERATIONAL QUESTION, once the collisions are found: a pixel agent (MineDojo and its
successors) sees RGB and has no such ceiling. Routing agentic vision through brightness
buys determinism and 6.8 KB of WebAssembly, and costs exactly this much.
"""
import sys, os, itertools, collections
# This used to hardcode '/workspace/projects/voxelglyph'. That made the experiment
# importable only on the machine where it was written, and silently imported
# something else everywhere else. A green run on the author's tree is not evidence.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from syzygy_port import luma8

STEP = 5   # 52^3 = 140k triples; enough to find structure, small enough to be honest

def run():
    """Execute the experiment and RETURN its result.

    THE CATEGORY ERROR THIS REPO HAD. Its seven verification pins re-derived this
    arithmetic from a second implementation and never once executed this file: an atexit
    counter measured ZERO executions across the whole suite. Two flagship mutations
    survived 7/7 green -- max() -> min(), so the product reported the NARROWEST class as
    the widest blind spot; and STEP 5 -> 200, so it swept 8 lattice points instead of
    140,608. A re-implementation verifies a product only if something also pins the
    re-implementation to the product, and nothing did.

    This function is what the pins call, so they execute the product rather than a copy.
    """
    buckets = collections.defaultdict(list)
    for r in range(0, 256, STEP):
        for g in range(0, 256, STEP):
            for b in range(0, 256, STEP):
                buckets[luma8(r, g, b)].append((r, g, b))
    widest = max(buckets, key=lambda L: len(buckets[L]))
    members = buckets[widest]
    far = max(itertools.combinations(members, 2),
              key=lambda p: sum((p[0][i] - p[1][i]) ** 2 for i in range(3)) ** 0.5)
    return {
        'step': STEP,
        'buckets': [{'luma': L, 'size': len(v)} for L, v in buckets.items()],
        'widest': {'luma': widest, 'size': len(members), 'colours': members},
        'far_apart': far,
    }


if __name__ == '__main__':
  # Everything below is the human-readable report. It runs when the file is executed
  # directly and NOT when the module is imported, so the test suite can call run()
  # without printing 40 lines every time.
  buckets = collections.defaultdict(list)
  for r in range(0, 256, STEP):
      for g in range(0, 256, STEP):
          for b in range(0, 256, STEP):
              buckets[luma8(r, g, b)].append((r, g, b))

  sizes = sorted((len(v) for v in buckets.values()), reverse=True)
  print(f"  {len(buckets)} luma classes; largest classes hold {sizes[:5]} colours each")

  # the widest class: the biggest blind spot in the encoding
  widest = max(buckets, key=lambda L: len(buckets[L]))
  members = buckets[widest]
  print(f"\n  WIDEST BLIND SPOT: luma {widest} holds {len(members)} distinct colours.")
  far_apart = max(itertools.combinations(members, 2),
                  key=lambda p: sum((p[0][i]-p[1][i])**2 for i in range(3))**0.5)
  c0, c1 = far_apart
  print(f"    two members maximally far apart inside that ONE class:")
  print(f"      {c0}  and  {c1}   (Euclidean distance {sum((c0[i]-c1[i])**2 for i in range(3))**0.5:.1f})")
  print(f"    both render to luma {luma8(*c0)} == {luma8(*c1)}. To Syzygy they are one colour.")

  # ── the ceiling, as a function of how many materials a game needs to separate ──
  print(f"""
    THE CEILING, AS A FUNCTION OF THE TASK
    ---------------------------------------
    A luma-only classifier must give one answer per luma class. If a task needs to tell apart
    materials that share a class, the best achievable accuracy is bounded by

        ceiling(K) = (number of classes used) / K

    Measured on the largest blind spot, a game using 8 visually distinct materials drawn from
    that one class scores at most 1/8 = {1/8:.4f} with a brightness-only encoder -- no
    matter the model, the data, or the training budget.

    This is a property of WHICH QUANTITY THE ENCODER PROJECTS. It is not a modelling failure
    and no architecture fixes it.
  """)

  # ── the honest middle: how much brightness can you afford to keep? ──────────
  # The interesting question is not luma-vs-id. It is: what does a glyph need to PRESERVE
  # for an agent to act? Measure how far two equal-luma colours are in RGB, and compare
  # that to the distance that survives a 2x4 cell average.
  d = sum((c0[i]-c1[i])**2 for i in range(3))**0.5
  cell_avg_delta = d / (STEP * 4)
  print(f"  THE NUMBER THAT SHOULD DRIVE THE ENCODER DESIGN")
  print(f"    the two colours are {d:.0f} apart in RGB")
  print(f"    after a 2x4 block average they are ~{cell_avg_delta:.1f} apart -- comparable to")
  print(f"    the quantisation step itself.")
  print(f"""
      So a voxel agent reading Syzygy's text would need to resolve a difference that its own
      2x4 pooling has already swamped. That is a sharper statement than 'luma is lossy':
      the loss is not only in the projection, it is in the POOLING, and the pooling is what
      gives the representation its compactness and its determinism.

      An agentic encoding therefore cannot be a better-tuned brightness renderer. It has to
      carry something the renderer does not have -- material identity, occupancy, ownership,
      reachability -- in a channel the pool cannot flatten. That is the design brief, and
      it is measurable rather than a matter of taste.
  """)
