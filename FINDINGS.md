# What a voxel agent can read through Syzygy

Grounded in `SuperInstance/Syzygy` (219 checks, 0 failures) and `SuperInstance/chiaroscuro`.
Casey's question: mature voxel agents learn from pixels; what happens when you distil them
into an agent that reads the ASCII/glyph representation instead?

Short answer: **Syzygy is a provably lossy channel for agentic vision, and the loss is
mostly in the pooling rather than in the projection — and that same pooling is what makes
the text usable.** Both halves of that were measured. The distillation experiment is
inconclusive and is reported as such.

## The seam

Syzygy's whole feature stack is a function of one number:

    syz_luma8 = (77·R + 150·G + 29·B) >> 8      (BT.601; weights sum 256, so it divides)
      -> Braille threshold (strict >)   -> 3×3 Sobel  -> STE tone  -> 16-point FFT

Nothing downstream of `syz_luma8` can see anything that line did not. That is a **rank-one
linear map R³ → R**, and it is the whole architecture of the observation space.

## exp1 — the ceiling is a property of the encoder, and it is provable

Any two materials at the same luma are indistinguishable to Syzygy: exactly, always.

Enumerating the RGB cube and bucketing by luma, the largest blind spot found was:

    luma 140  contains  (0, 240, 0)  and  (255, 60, 255)
    Euclidean distance between them: 403
    both render to luma 140.  To Syzygy they are ONE COLOUR.

So a game using 8 visually distinct materials drawn from that single class has a
brightness-only accuracy ceiling of **1/8 = 0.125**, independent of model, data, or budget.
No architecture fixes a distinction the representation never carried.

**The first attempt at this experiment failed and the failure is the useful part.** I
hand-picked eight "chromatically different" voxel materials; none collided, the ceiling came
out 1.0000, and it demonstrated nothing. The collisions are not something you trip over by
choosing colours carefully — you have to go looking for them.

### The sharper version

The two colours are **403 apart in RGB**. After a 2×4 block average they are about **20
apart**, comparable to the quantisation step itself. So the loss is not only in the
projection — it is in the **pooling**, and the pooling is precisely what buys the
compactness, the determinism, and the 6.8 KB of WebAssembly.

An agentic encoding cannot be a better-tuned brightness renderer. It has to carry something
the renderer does not have — material identity, occupancy, ownership, reachability — in a
channel the pool cannot flatten. That is the design brief, and it is measurable rather than
a matter of taste.

## exp2 — the distillation experiment is INCONCLUSIVE

Intended: same voxel world, same task, same labels, same architecture and budget; vary only
*which quantity* the encoder carries. Result: **all three encodings land at or below a
majority-class baseline, so none of them learns the task.** The numbers below are reported
for the record and are not evidence for anything.

| encoding | test acc | vs majority |
|---|---|---|
| RGB (luma per cell) | 0.4551 | −0.0138 |
| WORLD (material id per cell) | 0.4053 | −0.0636 |
| SYZYGY (2×4 Braille of luma) | 0.4942 | +0.0253 |

**Three instrument defects were found and fixed before reaching that state, and the order
matters — each one inverted the result.**

1. **Degenerate labels.** `bearing()` compared a coordinate tuple against a corridor *index*,
   so it was never true, every label fell through to one class, and the majority baseline
   came out at **1.0000**. All three encodings scored 1.0 and the experiment measured
   nothing. A degenerate label set is the experiment equivalent of a std == 0 measurement.
2. **Unequal input widths.** RGB got 36 features, SYZYGY got 4, WORLD got 12. A linear
   model with nine times the inputs has nine times the capacity, so the first comparison
   measured input width and was reported as if it measured the representation. Held at 12
   each, the ordering flipped.
3. **A "3-layer MLP" that was linear.** `w = 3 × INPUTS; w·x + b` — a single linear layer
   under a docstring claiming three. Every number it produced was "what a linear model can
   decode", and one-hot material features and luma features are about equally decodable
   that way, so the experiment could not separate the two hypotheses it existed to compare.
   Replaced with a 24-unit ReLU hidden layer.

**And one confound that was still live after all three:** the first palette made goal bright
cyan and hazard red, so **material and brightness were collinear by construction** and the
task was solvable from brightness regardless of the hypothesis. Re-running with the exp1
collision — goal `(0,240,0)` and hazard `(255,60,255)`, both luma 140, so a brightness-only
agent provably cannot tell them apart — is what the table above reports.

**Why it is still inconclusive and not a negative result:** 900 samples, a 12-cell
egoccenthic window, 4000 steps of hand-rolled SGD, and a three-way action is simply not
enough to learn a task that requires carrying a goal and a hazard across a ring corridor.
"Below baseline" here means *undertrained*, not *unreadable*. Distinguishing those two
requires either a real training budget or a task simple enough to be learnable within one,
and neither has been run.

## The thing worth carrying forward

Syzygy was built to make a *frame* readable and checkable. That is a solved problem and it
is solved well: integer-only, no allocation, byte-identical across a desktop, a browser tab,
and an 8-bit AVR, with a golden hash anyone can reproduce.

**It was not built to make a *world* readable, and the two are not the same problem.**
Brightness is a rendering choice; material is a world fact. An agentic encoding has to carry
the second, and it has to do so in a channel that survives a 2×4 pool — which exp1 shows is
where the information goes.

The useful next experiment is not a bigger network. It is the **minimal sufficient
encoding**: given a task, find the smallest set of per-cell symbols such that a *linear*
policy on those symbols matches a pixel policy's accuracy. That is a compression question
with a measurable answer, it does not need a renderer, and it answers the design question
directly — how much of a voxel world actually has to survive to be actionable.

## Provenance

- `syzygy_port.py` — a faithful port of the C integer path. Constants copied exactly
  (`(77R+150G+29B)>>8`, Braille bits 0,1,2,6 / 3,4,5,7, the 4×4 STE basis, strict `>`).
  Verified against the C constants: luma(255,0,0)=76, luma(0,255,0)=149, luma(0,0,255)=28,
  full white=255 exactly because the weights sum to 256.
- Syzygy's own suite was run first: **7 suites, 219 checks, 0 failures.**
