# voxelglyph

What can a voxel agent actually read through `SuperInstance/Syzygy`'s text encoding?

**Read [`FINDINGS.md`](FINDINGS.md) first.** Two experiments, one solid and one
inconclusive, and the record of three instruments that lied on the way.

## The one-line finding

Syzygy's entire observation is a **rank-one linear projection R³ → R** (BT.601 luma), and
its 2×4 pooling then flattens what survives it. Two colours 403 apart in RGB can land on
the same luma — `(0,240,0)` and `(255,60,255)`, both luma 140 — and to Syzygy they are one
colour. That is a **provable** accuracy ceiling, not an empirical one.

But the same pooling is also what makes the text compact and deterministic. So the
interesting question is not "luma is lossy" — it is **what a per-cell symbol must carry to
survive a 2×4 pool**, and that is measurable.

## Run

```sh
python3 exp1_luma_collision.py    # the provable ceiling, and the blind spot
python3 exp2_distillation.py      # INCONCLUSIVE -- see FINDINGS.md before citing
```

## Landing page — live

**https://dng8dc2ppoaxp.space.minimax.io**

`site/index.html` is a self-contained page with a **live side-by-side demonstration**: a voxel
scene rendered on the left, and what `syz_luma8 -> Braille` computes from it on the right,
running the real integer path ported line for line. A toggle swaps between an ordinary palette
and an adversarial one where GOAL and HAZARD share a luma.

The claim on the page is verified, not asserted. Swapping GOAL and HAZARD under the adversarial
palette produces **byte-identical** Syzygy text, while a human sees two plainly different
colours. Under the ordinary palette the text differs, which is exactly why an ordinary palette
hides the problem.

## Honest status

| | |
|---|---|
| exp1 | solid. Reproducible, and the ceiling is a property of the encoder |
| exp2 | **inconclusive.** Three instrument defects found and fixed; "below baseline" here means *undertrained*, not *unreadable* |
| next | the minimal sufficient encoding: smallest symbol set where a *linear* policy matches a pixel policy. No renderer needed, and it answers the design question |

## Provenance and verification

- `syzygy_port.py` was checked against the C constants directly: `luma(255,0,0)=76`,
  `luma(0,255,0)=149`, `luma(0,0,255)=28`, full white `=255` exactly because the BT.601
  weights sum to 256. Braille bit order `0,1,2,6` / `3,4,5,7` and the 4×4 STE basis were
  copied from `include/syz_braille.h` and `include/syz_ste.h`.
- `SuperInstance/Syzygy` was cloned and its own suite run before any of this:
  **7 suites, 219 checks, 0 failures.**
- Neither `Syzygy` nor `chiaroscuro` contains any ML or agent hook today. This repository
  is not a port of theirs; it is a measurement of what their encoding can and cannot carry.

## What a follow-up should do

1. **Run the minimal sufficient encoding.** Find the smallest per-cell symbol set for which
   a *linear* policy matches a pixel policy on a learnable task. No renderer required, and
   it answers the design question directly.
2. **Get a real training budget into exp2**, or replace the task with one learnable inside
   900 samples. "Below baseline" is currently ambiguous between *undertrained* and
   *unreadable*, and only one of those is a finding.
3. **Ask the question in the other direction.** Not "how much does brightness lose" but
   "how much brightness can you afford to keep alongside material identity", since the
   2×4 pool is what buys the compactness and the determinism. The useful answer is probably
   a mixed channel, not a swap.
