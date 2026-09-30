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

## Honest status

| | |
|---|---|
| exp1 | solid. Reproducible, and the ceiling is a property of the encoder |
| exp2 | **inconclusive.** Three instrument defects found and fixed; "below baseline" here means *undertrained*, not *unreadable* |
| next | the minimal sufficient encoding: smallest symbol set where a *linear* policy matches a pixel policy. No renderer needed, and it answers the design question |
