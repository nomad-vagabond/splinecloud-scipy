import unittest

import numpy as np
from scipy.interpolate import UnivariateSpline, NdBSpline

from splinecloud_scipy import NdBsplExplicit, ExplicitUnivariateSpline, ParametricUnivariateSpline


def fitted_curve(log_x=False, log_y=False):
    """A smoothing fit stored the way SplineCloud stores function fits: the legacy parametric
    payload (Greville embedding) and the equivalent explicit schema v1 payload."""
    rng = np.random.default_rng(0)
    x = np.sort(rng.uniform(1, 20, 120))
    y = np.sin(x / 3) + 2.5 + rng.normal(0, 0.03, len(x))
    xs = np.log10(x) if log_x else x
    ys = np.log10(y) if log_y else y
    f = UnivariateSpline(xs, ys, k=3, s=0.1)

    k = 3
    inner = f.get_knots()
    tx = np.r_[[inner[0]] * k, inner, [inner[-1]] * k]
    cx = np.array([tx[i + 1:i + k + 1].mean() for i in range(len(tx) - k - 1)])
    cy = f.get_coeffs()
    t_norm = (tx - tx[0]) / (tx[-1] - tx[0])

    scales = ["Logarithmic" if log_x else "Linear", "Logarithmic" if log_y else "Linear"]
    native = {"v": 1, "nx": 1, "ny": 1, "k": [k], "t": [tx.tolist()], "shape": [len(cy)],
              "c": cy.tolist(), "domain": [[tx[0], tx[-1]]], "scales": scales,
              "extrapolation": "none"}
    parametric = ParametricUnivariateSpline((t_norm, cx, cy, k), log_x=log_x, log_y=log_y)
    return x, native, parametric


class TestExplicitUnivariateSpline(unittest.TestCase):

    def test_matches_parametric_inside_domain(self):
        x, native, parametric = fitted_curve()
        spline = ExplicitUnivariateSpline.from_json(native)
        xe = np.linspace(x.min(), x.max(), 500)
        np.testing.assert_allclose(spline.eval(xe), parametric.eval(xe), atol=1e-12)
        self.assertAlmostEqual(spline.eval(7.3), parametric.eval(7.3), places=12)
        self.assertIsInstance(spline.eval(7.3), float)

    def test_matches_parametric_extrapolation(self):
        x, native, parametric = fitted_curve()
        spline = ExplicitUnivariateSpline.from_json(native)
        xe = np.array([x.min() - 0.5, x.max() + 0.5])
        np.testing.assert_allclose(spline.eval(xe, extrapolate=True),
                                   parametric.eval(xe, extrapolate=True), atol=1e-9)

    def test_outside_domain_is_nan_without_extrapolation(self):
        x, native, _ = fitted_curve()
        spline = ExplicitUnivariateSpline.from_json(native)
        self.assertTrue(np.isnan(spline.eval(x.max() + 1.0)))

    def test_log_scales(self):
        x, native, parametric = fitted_curve(log_x=True, log_y=True)
        spline = ExplicitUnivariateSpline.from_json(native)
        xe = np.linspace(x.min(), x.max(), 200)
        np.testing.assert_allclose(spline.eval(xe), parametric.eval(xe), rtol=1e-12)
        np.testing.assert_allclose(spline.domain[0], [x.min(), x.max()], rtol=1e-6)

    def test_domain_ends_are_inside(self):
        ## the first/last data points must evaluate, even after a log10 / 10** round trip
        for log in (False, True):
            x, native, _ = fitted_curve(log_x=log, log_y=log)
            spline = ExplicitUnivariateSpline.from_json(native)
            ends = spline.domain[0]
            self.assertTrue(np.all(np.isfinite(spline.eval(ends))))
            self.assertTrue(np.all(np.isfinite(spline.eval([x.min(), x.max()]))))

    def test_fit_accuracy_matches_parametric(self):
        x, native, parametric = fitted_curve()
        spline = ExplicitUnivariateSpline.from_json(native)
        points = np.column_stack([x, np.sin(x / 3) + 2.5])
        for method in ("RMSE", "MSE", "MAE"):
            self.assertAlmostEqual(spline.fit_accuracy(points, method=method),
                                   parametric.fit_accuracy(points, method=method), places=10)

    def test_to_parametric_roundtrip(self):
        x, native, _ = fitted_curve()
        spline = ExplicitUnivariateSpline.from_json(native)
        xe = np.linspace(x.min(), x.max(), 300)
        np.testing.assert_allclose(spline.to_parametric().eval(xe), spline.eval(xe), atol=1e-12)

    def test_extrapolation_policies(self):
        ## y = 2x on [0, 1] (linear B-spline), so every policy is easy to check by hand
        spline = ExplicitUnivariateSpline([[0, 0, 1, 1]], [1], [0, 2])
        self.assertTrue(np.isnan(spline.eval(2.0, extrapolate="none")))
        self.assertAlmostEqual(spline.eval(2.0, extrapolate="clamp"), 2.0)
        self.assertAlmostEqual(spline.eval(2.0, extrapolate="linear"), 4.0)
        self.assertAlmostEqual(spline.eval(2.0, extrapolate="polynomial"), 4.0)
        with self.assertRaises(ValueError):
            spline.eval(0.5, extrapolate="sideways")

    def test_rejects_unsupported_schema(self):
        _, native, _ = fitted_curve()
        with self.assertRaises(ValueError):
            ExplicitUnivariateSpline.from_json({**native, "v": 2})


class TestNdBsplExplicit(unittest.TestCase):

    def setUp(self):
        k = 3
        self.t = [np.r_[[0.0] * 4, 0.3, 0.6, [1.0] * 4], np.r_[[-1.0] * 4, 0.5, [2.0] * 4]]
        rng = np.random.default_rng(1)
        self.c = rng.normal(size=(6, 5))
        self.reference = NdBSpline(tuple(self.t), self.c, (k, k))
        self.spline = NdBsplExplicit(self.t, [k, k], self.c.ravel())

    def test_eval_matches_scipy(self):
        pts = np.random.default_rng(2).uniform([0, -1], [1, 2], (200, 2))
        np.testing.assert_allclose(self.spline.eval(pts), self.reference(pts), atol=1e-12)
        np.testing.assert_allclose(self.spline(pts[:, 0], pts[:, 1]), self.reference(pts),
                                   atol=1e-12)

    def test_eval_grid_shape(self):
        grid = self.spline.eval_grid(np.linspace(0, 1, 4), np.linspace(-1, 2, 7))
        self.assertEqual(grid.shape, (4, 7))

    def test_grad_matches_finite_differences(self):
        p, h = np.array([[0.4, 0.7]]), 1e-6
        g = self.spline.grad(p)[0]
        for d in range(2):
            e = np.zeros(2); e[d] = h
            fd = (self.spline.eval(p + e) - self.spline.eval(p - e))[0] / (2 * h)
            self.assertAlmostEqual(g[d], fd, places=5)

    def test_outside_domain_is_nan_by_default(self):
        self.assertTrue(np.isnan(self.spline.eval([[1.5, 0.0]])[0]))


if __name__ == "__main__":
    unittest.main()
