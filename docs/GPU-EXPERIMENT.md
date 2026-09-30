# GPU experiment brief — `voxelglyph`

**Glyph and braille as an agentic observation**

This is the slice of the master queue that concerns this repository. The full document,
with all ten experiments and their decision trees, is at
[`SuperInstance/fleet-triage` → `docs/GPU-EXPERIMENTS.md`](https://github.com/SuperInstance/fleet-triage/blob/main/docs/GPU-EXPERIMENTS.md).

1. **State whether CUDA or CPU actually ran.** A CPU fallback reported as a GPU result is a
   fabricated measurement. Record the device string with the number.
2. **Report the variance, not just the mean.** If the measured quantity has `std == 0` over
   the sample, the result is **INCONCLUSIVE, never PASSED**. This is a hard rule: an earlier
   sharding experiment scored `PASSED` on the literal comparison `0 < 0`.
3. **The ground truth must be computed, not asserted.** Every game number below is exact.
   If a label is approximate, say by how much.
4. **Non-degeneracy is a precondition, not a result.** Assert that the data has variance
   *before* evaluating any relational claim about it.
5. **A control must vary the thing it audits, by a different path than the audited thing.**
   The most expensive mistake in this project's history: a control built from the same call
   path as the probes, which therefore confirmed a fault instead of auditing it.
6. **A ratio whose denominator can be zero is a construction, not a measurement.** It always
   produces a finding, and the finding is always false.
7. **Seed everything and record the seed.** Two runs of the same experiment that differ are
   a finding about the seed, not about the method.
8. **Cross-port arithmetic must preserve the reference loop/summation order.** Float addition
   is not associative; a port that vectorises changes the answer.

---

### Experiment 6 — Glyph and braille as an agentic observation

**Repo:** `voxelglyph`. **Result: exp1 is solid, exp2 is INCONCLUSIVE, exp3 is at chance.**

The proven fact: Syzygy's luma is a rank-1 projection, so `(0,240,0)` and `(255,60,255)` are
**403 RGB units apart and both luma 140**. A rank-one luma ceiling means a genuine
aliasing collision, not a near-miss.

exp2 (distillation into a linear model) is **inconclusive after four corrections**: degenerate
labels, unequal input widths, a falsely described linear model, and palette/material
collinearity. exp3 (minimal encoding) is **at chance for every brightness alphabet** —
alphabet *alignment*, not alphabet size, controls collisions.

**There is no authorised claim of successful agentic distillation.** Do not report one.

| Result | What it means |
|---|---|
| A non-linear model beats the linear one on the same corrected setup | The corrected null was a real null and the function is genuinely nonlinear. Report it as a new result, not as a reversal. |
| The linear model still ties | **Alignment, not capacity, is the binding constraint.** Any encoder that does not canonicalise the palette first is measuring nothing. This is the more interesting claim and it is already supported by exp3. |
| A canonicalising encoder (map each glyph to its byte) succeeds | Trivially true and **not a result** — it is a lookup table. Say so if you find yourself reporting it. |

---

## 4. Reporting format

Every experiment reports, in this order:

1. **Device.** Did CUDA actually run? Paste the device string.
2. **Data provenance.** Which commit, which digest, how many states/positions, and the
   FNV-1a 64 of the input file.
3. **The ceiling.** What is perfect, and what did you get as a fraction of it.
4. **Variance.** Mean ± std over N seeds, with the seeds listed. **std == 0 means INCONCLUSIVE.**
5. **The branch.** Which row of which decision tree above you landed on, quoted.
6. **Controls.** What ran that could have failed. If nothing could have failed, say that —
   it is the finding.
