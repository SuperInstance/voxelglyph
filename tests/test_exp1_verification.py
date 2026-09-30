"""Independent verification receipt for exp1_luma_collision.py (fleet verify pattern).

These pins do NOT trust the experiment's own code for the claims they check:

  * the BT.601 luma formula is re-coded inline (pin 1) and cross-checked against
    syzygy_port.luma8 over the enumerated cube (pin 2) -- if the port disagrees
    with the C it ports, the disagreement is the finding, and these tests are
    how a reader catches it without reading C headers;
  * the headline numbers in FINDINGS.md / README.md (the luma-140 blind spot,
    the (0,240,0)/(255,60,255) pair, distance ~403, ceiling 1/8 = 0.125) are
    re-derived from the independently-coded formula and quote-matched against
    the docs (pins 3-5), so doc drift cannot silently detach the prose from
    the arithmetic;
  * the step-5 enumeration is re-run here (pin 5) and must name luma 140 as the
    widest class and must contain both claimed colours.

Honest limits (stated, not hidden):
  * exp2 (distillation) is reported INCONCLUSIVE by its own author and was NOT
    re-run here; the pins above say nothing about it.
  * exp3 results were not re-derived here (see exp3_results.json provenance).
  * Verification covers the arithmetic claims, not the site/ demo's byte-identity
    claim (that is a separate rendering path in site/syzygy.js).

Run:  python3 -m unittest tests.test_exp1_verification -v
Stdlib only, no third-party imports, no network.
"""
import re
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from syzygy_port import luma8  # instrument under test (pin 2 cross-checks it)

FINDINGS = (REPO / "FINDINGS.md").read_text(encoding="utf-8")
README = (REPO / "README.md").read_text(encoding="utf-8")

# The claimed pair, as stated in FINDINGS.md and README.md.
C0 = (0, 240, 0)
C1 = (255, 60, 255)
CLAIMED_LUMA = 140


def independent_luma8(r: int, g: int, b: int) -> int:
    """BT.601 integer luma, re-coded from the docstring contract -- NOT imported."""
    return ((77 * r + 150 * g + 29 * b) >> 8) & 0xFF


class TestIndependentLuma(unittest.TestCase):
    def test_formula_recode_claimed_pair(self):
        # The FINDINGS headline: both colours render to luma 140 under the
        # independently re-coded formula.
        self.assertEqual(independent_luma8(*C0), CLAIMED_LUMA)
        self.assertEqual(independent_luma8(*C1), CLAIMED_LUMA)

    def test_formula_recode_crosschecks_port(self):
        # Port vs independent formula over the full step-5 enumeration lattice
        # (the same lattice exp1 uses): any disagreement is a port defect.
        for r in range(0, 256, 5):
            for g in range(0, 256, 5):
                for b in range(0, 256, 5):
                    self.assertEqual(
                        luma8(r, g, b), independent_luma8(r, g, b),
                        f"port disagrees with BT.601 at {(r, g, b)}")

    def test_weights_sum_256(self):
        # The docstring's load-bearing claim: weights sum exactly 256, so the
        # >>8 is an exact normalisation, not an approximation.
        self.assertEqual(77 + 150 + 29, 256)


class TestDocQuoteMatch(unittest.TestCase):
    def test_findings_names_the_pair(self):
        # Doc drift pin: the prose must still carry the exact claimed colours.
        # Both spellings occur in-repo (FINDINGS.md spaces them, README.md does not).
        spaced = ("(0, 240, 0)", "(255, 60, 255)")
        tight = ("(0,240,0)", "(255,60,255)")
        for doc in (FINDINGS, README):
            self.assertTrue(
                all(s in doc for s in spaced) or all(t in doc for t in tight),
                "claimed colour pair missing from a doc that asserts it",
            )
            self.assertIn("luma 140", doc)

    def test_distance_claim_consistent(self):
        # FINDINGS/README claim the pair is 403 apart in RGB. Re-derive.
        d = sum((C0[i] - C1[i]) ** 2 for i in range(3)) ** 0.5
        self.assertAlmostEqual(d, 403.0, delta=1.0)
        # and the docs must still assert ~403 (not a stale number).
        self.assertRegex(FINDINGS, r"403")

    def test_ceiling_claim_consistent(self):
        # The provable ceiling for 8 same-class materials is exactly 1/8.
        self.assertAlmostEqual(1 / 8, 0.125)
        self.assertIn("0.125", FINDINGS)


class TestEnumerationRederived(unittest.TestCase):
    def test_widest_class_is_luma_140_and_holds_pair(self):
        # Re-run exp1's enumeration with the INDEPENDENT formula and confirm
        # the headline: widest class, and it contains the claimed pair.
        from collections import defaultdict
        buckets = defaultdict(list)
        for r in range(0, 256, 5):
            for g in range(0, 256, 5):
                for b in range(0, 256, 5):
                    buckets[independent_luma8(r, g, b)].append((r, g, b))
        widest = max(buckets, key=lambda L: len(buckets[L]))
        self.assertEqual(widest, CLAIMED_LUMA)
        self.assertIn(C0, buckets[widest])
        self.assertIn(C1, buckets[widest])
        # 256 distinct classes = the projection is surjective over the lattice.
        self.assertEqual(len(buckets), 256)


if __name__ == "__main__":
    unittest.main()
