# The verification layer did not verify the product

**Found by the playtest lane, not by me. All three defects were in `voxelglyph`, which I
owned, and one of them was in a PR I merged the same evening.**

`tests/test_exp1_verification.py` had **7 pins, all green**, and none of them executed
`exp1_luma_collision.py` — the experiment this repository is about and the source of the
README's headline finding. Measured with an `atexit` counter: **0 executions across the
whole suite.**

## The three defects, and what each one hid

**1. The product was never run.** The pins swept the lattice themselves and re-derived the
arithmetic from a second implementation. Both were correct; neither was connected to the
thing being verified. A re-implementation verifies a product only if something also pins
the re-implementation to the product, and nothing did.

**2. Two flagship mutations survived 7/7 green.**

| mutation | result before | result after |
|---|---|---|
| `widest = max(...)` → `min(...)` — the product reports the *narrowest* luma class as the widest blind spot, inverting the conclusion | **SURVIVED** | **FAILED** |
| `STEP = 5` (140,608 triples) → `200` (**8** triples) | **SURVIVED** | **FAILED** |
| port `LUMA_WR` → `(200, 3, 91)`, **summing to 294** | **SURVIVED** | **FAILED (4 pins)** |

**3. A test named for a property of the product, whose body never touched it.**

```python
def test_weights_sum_256(self):
    self.assertEqual(77 + 150 + 29, 256)     # arithmetic on literals in the test body
```

It certifies the *arithmetic of a line of source in the test file*. The real weights could
be anything. Setting them to something summing to 294 left this green, which is the most
damning version of the defect I have hit: **a test whose name asserts a product property
while its body asserts nothing about the product.**

## What changed

- `exp1_luma_collision.run()` — the product now **returns its result** so a test can call
  it. Its module body is guarded under `if __name__ == "__main__"`, so importing it is
  silent.
- The hardcoded `sys.path.insert(0, '/workspace/projects/voxelglyph')` is gone, replaced by
  the file's own directory. **That line made the experiment importable only on the machine
  where it was written, and silently imported something else everywhere else** — the same
  defect family as the 44 hardcoded output roots found in `quilt-cell-bridges`.
- `test_weights_sum_256` reads `LUMA_WR` from the port and asserts the value as well as the
  sum.
- Four new pins that **execute the product**: it runs, it reports the *widest* class (not the
  narrowest), it swept a lattice at least as fine as `STEP=5`, and its far-apart pair is the
  documented one.

**11 tests, green, 0.79 s.** The three mutations above now fail. The documented numbers are
unchanged: **luma 140, 940 colours, far-apart `((0, 240, 0), (255, 60, 255))`** — `run()`
returns exactly what `FINDINGS.md` claims, so the fix did not move the result.

## The lesson, stated once

**Seven pins re-deriving a claim from a second implementation is not a verification.** It is
a second opinion, and two correct opinions about the same wrong product are still wrong.

A verification layer earns the name only when something connects it to the thing being
verified. Here that something was missing, and nothing about the green result indicated it.
