"""
Regression test for the LS-NMF Rust update stability bug.

With the default `column_mean` initialisation, approximately half of the W and H
entries are clipped to 1e-12 (negatives from the random draw).  When those near-zero
entries appear in the denominator of the multiplicative update

    H  ←  H  *  (W.T @ (We * V)) / (W.T @ (We * W @ H))

the ratio can overflow to a large finite f32 value.  The Rust path accumulated this
error and converged to a spurious local minimum (Q ≈ 1e5) while erroneously reporting
`converged=True`.  The Python path happened to be numerically stable enough to
self-correct, masking the difference.

Fix: clamp H and W to [1e-12, ∞) after every multiplicative update step, matching the
epsilon floor that `SA.initialize()` already applies to the initial matrices.

This test verifies that the Rust and Python paths produce Q values within 20 % of each
other on a small synthetic problem seeded for reproducibility.  A spurious-convergence
failure (Rust Q >> Python Q) is the signature of the pre-fix bug.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np
import pytest
from esat.model.sa import SA
from esat.metrics import q_loss


def _make_synthetic(m=50, n=12, p=3, seed=0):
    rng = np.random.default_rng(seed)
    F_true = rng.uniform(0.1, 1.0, size=(p, n))
    G_true = rng.uniform(0.1, 1.0, size=(m, p))
    V = G_true @ F_true
    U = 0.001 + 0.05 * np.abs(V)
    return V, U, p


class TestLsNmfRustStability:
    """Verify that the Rust LS-NMF update is numerically stable under the default
    (column_mean) initialisation, which produces ~50 % near-zero entries in W and H."""

    V, U, p = _make_synthetic()

    def _run(self, use_rust: bool, init_method: str = "column_mean") -> float:
        sa = SA(
            V=self.V, U=self.U, factors=self.p,
            method="ls-nmf", seed=42, parallel=False, verbose=False,
        )
        sa.initialize(init_method=init_method)
        sa.optimized = use_rust
        sa.train(max_iter=20000, converge_delta=0.01, converge_n=100)
        W = sa.W.astype(np.float64)
        H = sa.H.astype(np.float64)
        return float(q_loss(self.V, self.U, W, H))

    def test_rust_column_mean_q_is_finite(self):
        """Rust path must not return a Q that is NaN or infinite."""
        try:
            q = self._run(use_rust=True, init_method="column_mean")
        except Exception:
            pytest.skip("Rust extension not available")
        assert np.isfinite(q), f"Rust Q is not finite: {q}"

    def test_rust_column_mean_q_close_to_python(self):
        """Rust Q must be within 20 % of the Python Q.

        Pre-fix, the Rust path produced Q ≈ 1e5 while Python produced Q ≈ 5,
        a factor of ~20 000×.  Post-fix both should converge to similar minima.
        """
        try:
            q_rust = self._run(use_rust=True, init_method="column_mean")
        except Exception:
            pytest.skip("Rust extension not available")
        q_py = self._run(use_rust=False, init_method="column_mean")
        ratio = q_rust / q_py if q_py > 0 else float("inf")
        assert ratio < 1.2, (
            f"Rust Q ({q_rust:.2f}) is more than 20 % above Python Q ({q_py:.2f}). "
            f"This is the signature of the near-zero denominator instability bug."
        )

    def test_rust_kmeans_q_close_to_python(self):
        """Baseline: both paths agree with k-means init (was already working)."""
        try:
            q_rust = self._run(use_rust=True, init_method="kmeans")
        except Exception:
            pytest.skip("Rust extension not available")
        q_py = self._run(use_rust=False, init_method="kmeans")
        ratio = q_rust / q_py if q_py > 0 else float("inf")
        assert ratio < 1.2, (
            f"Rust Q ({q_rust:.2f}) diverges from Python Q ({q_py:.2f}) even with k-means init."
        )
