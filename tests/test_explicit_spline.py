import unittest

import numpy as np
from scipy.interpolate import UnivariateSpline, NdBSpline

from splinecloud_scipy import (NdBsplExplicit, ExplicitUnivariateSpline, ParametricUnivariateSpline,
                               ParametricBivariateSpline)


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


def explicit_surface(log_x1=False, log_x2=False, log_y=False):
    """A cubic explicit surface on x1 in [1, 10], x2 in [0.5, 5] (physical), stored in scaled
    space like SplineCloud's fitted surfaces."""
    k = 3
    lo1, hi1 = (0.0, 1.0) if log_x1 else (1.0, 10.0)
    lo2, hi2 = (np.log10(0.5), np.log10(5.0)) if log_x2 else (0.5, 5.0)
    t1 = np.r_[[lo1] * 4, np.linspace(lo1, hi1, 5)[1:-1], [hi1] * 4]
    t2 = np.r_[[lo2] * 4, np.linspace(lo2, hi2, 4)[1:-1], [hi2] * 4]
    rng = np.random.default_rng(3)
    c = rng.uniform(0.5, 1.5, (len(t1) - k - 1, len(t2) - k - 1))
    scales = ["Logarithmic" if log else "Linear" for log in (log_x1, log_x2, log_y)]
    return NdBsplExplicit([t1, t2], [k, k], c.ravel(), scales=scales)


class TestExplicitSurfaceEmbeddingAndSlices(unittest.TestCase):

    def _inner_points(self, spline, n=40):
        lo, hi = spline.domain[:, 0], spline.domain[:, 1]
        return np.random.default_rng(4).uniform(lo + 0.02 * (hi - lo), hi - 0.02 * (hi - lo),
                                                (n, 2))

    def test_to_parametric_matches_explicit(self):
        for logs in ((False, False, False), (True, False, True), (True, True, True)):
            spline = explicit_surface(*logs)
            parametric = spline.to_parametric()
            self.assertIsInstance(parametric, ParametricBivariateSpline)
            pts = self._inner_points(spline)
            expected = spline.eval(pts)
            got = np.array([parametric.eval_point(x1, x2) for x1, x2 in pts])
            np.testing.assert_allclose(got, expected, rtol=1e-8, err_msg=str(logs))

    def test_slice_fixed_x2(self):
        for logs in ((False, False, False), (True, True, True)):
            spline = explicit_surface(*logs)
            x2 = 2.0
            curve = spline.slice({1: x2})
            self.assertIsInstance(curve, ExplicitUnivariateSpline)
            x1 = np.linspace(*spline.domain[0], 50)
            np.testing.assert_allclose(curve.eval(x1), spline(x1, np.full_like(x1, x2)),
                                       rtol=1e-12, err_msg=str(logs))

    def test_slice_fixed_x1(self):
        spline = explicit_surface(log_x1=True)
        curve = spline.slice({0: 3.0})
        x2 = np.linspace(*spline.domain[1], 30)
        np.testing.assert_allclose(curve.eval(x2), spline(np.full_like(x2, 3.0), x2),
                                   rtol=1e-12)

    def test_slice_rejects_bad_requests(self):
        spline = explicit_surface()
        self.assertIs(spline.slice({}), spline)
        with self.assertRaises(ValueError):
            spline.slice({1: 50.0})             ## outside the domain
        with self.assertRaises(ValueError):
            spline.slice({0: 2.0, 1: 2.0})      ## nothing left free
        with self.assertRaises(ValueError):
            spline.slice({2: 1.0})              ## no such axis

    def test_eval_points_and_call_agree(self):
        spline = explicit_surface()
        pts = self._inner_points(spline)
        np.testing.assert_allclose(spline.eval_points(pts), spline(pts[:, 0], pts[:, 1]))


if __name__ == "__main__":
    unittest.main()
