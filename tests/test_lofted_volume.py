import unittest

import numpy as np

from splinecloud_scipy import ParametricBivariateSpline, LoftedSplineVolume


def make_surface(x3_offset):
    """A simple bicubic surface: y = x1^2 + x2^2 + x3_offset."""
    tu = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    tv = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    ku, kv = 3, 3

    gu = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    gv = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    GU, GV = np.meshgrid(gu, gv, indexing='ij')

    cp = np.zeros((5, 5, 3))
    cp[:, :, 0] = GU                          # x1
    cp[:, :, 1] = GU**2 + GV**2 + x3_offset   # y (dependent)
    cp[:, :, 2] = GV                          # x2

    return ParametricBivariateSpline(tu, tv, cp, ku, kv)


class TestLoftedSplineVolumeExtrapolation(unittest.TestCase):
    """Tests for ParametricTrivariateSpline.eval(extrapolate=True)."""

    def setUp(self):
        self.x3_values = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        surfaces = [make_surface(x3) for x3 in self.x3_values]
        self.surfaces = surfaces
        self.vol = LoftedSplineVolume(surfaces, self.x3_values, w_degree=3)

    def test_in_domain_regression(self):
        """extrapolate=True must reproduce the clipped in-domain result."""
        y_clipped = self.vol.eval(x1=0.6, x2=0.4, x3=25.0, extrapolate=False)
        y_extrap = self.vol.eval(x1=0.6, x2=0.4, x3=25.0, extrapolate=True)
        self.assertAlmostEqual(y_clipped, y_extrap, places=6)

    def test_cross_section_matches_source_surface_at_boundary(self):
        """
        At an exact station (w = 0 or 1), the extrapolation path's synthetic
        cross-section must equal the original input surface exactly.
        """
        y_via_volume = self.vol.eval(x1=0.6, x2=0.4, x3=40.0, extrapolate=True)
        y_via_surface = self.surfaces[-1].eval_point(0.6, 0.4)
        self.assertAlmostEqual(y_via_volume, y_via_surface, places=10)

    def test_x3_extrapolation_only(self):
        """x3 beyond the station domain, x1/x2 in-domain."""
        y = self.vol.eval(x1=0.6, x2=0.4, x3=45.0, extrapolate=True)
        self.assertIsNotNone(y)
        # analytic surface is y = x1^2 + x2^2 + x3; the B-spline fit carries
        # a small systematic offset already present in-domain (~0.1 here).
        expected = 0.6**2 + 0.4**2 + 45.0
        self.assertAlmostEqual(y, expected, delta=0.2)

    def test_x1_extrapolation_only(self):
        """x1 beyond the surface's own domain, x3 in-domain."""
        y = self.vol.eval(x1=1.3, x2=0.4, x3=20.0, extrapolate=True)
        self.assertIsNotNone(y)
        expected = 1.3**2 + 0.4**2 + 20.0
        self.assertAlmostEqual(y, expected, delta=0.05)

    def test_simultaneous_three_axis_extrapolation(self):
        """x1 and x3 both out of range at once."""
        y = self.vol.eval(x1=1.3, x2=0.4, x3=45.0, extrapolate=True)
        self.assertIsNotNone(y)
        expected = 1.3**2 + 0.4**2 + 45.0
        self.assertAlmostEqual(y, expected, delta=0.2)

    def test_default_extrapolate_false_clips_instead_of_extrapolating(self):
        """Legacy behaviour is unchanged: out-of-range values are clipped, not extrapolated."""
        y_clipped = self.vol.eval(x1=1.3, x2=0.4, x3=20.0, extrapolate=False)
        y_extrapolated = self.vol.eval(x1=1.3, x2=0.4, x3=20.0, extrapolate=True)
        self.assertIsNotNone(y_clipped)
        self.assertNotAlmostEqual(y_clipped, y_extrapolated, places=2)

    def test_w_distance_guard_refuses_far_extrapolation(self):
        """Far outside the station domain, the w-distance guard refuses (returns None)."""
        y = self.vol.eval(x1=0.6, x2=0.4, x3=1000.0, extrapolate=True)
        self.assertIsNone(y)

    def test_w_distance_guard_threshold_is_configurable(self):
        """A tighter w_distance_threshold refuses extrapolation sooner."""
        y_default = self.vol.eval(x1=0.6, x2=0.4, x3=41.0, extrapolate=True)
        y_tight = self.vol.eval(x1=0.6, x2=0.4, x3=41.0, extrapolate=True,
                                 w_distance_threshold=0.001)
        self.assertIsNotNone(y_default)
        self.assertIsNone(y_tight)

    def test_diagnostics_report_extrapolation_state(self):
        y, diagnostics = self.vol.eval(x1=0.6, x2=0.4, x3=45.0, extrapolate=True,
                                        return_diagnostics=True)
        self.assertIsNotNone(y)
        self.assertTrue(diagnostics['w_extrapolated'])
        self.assertFalse(diagnostics['refused'])

    def test_diagnostics_report_refusal(self):
        y, diagnostics = self.vol.eval(x1=0.6, x2=0.4, x3=1000.0, extrapolate=True,
                                        return_diagnostics=True)
        self.assertIsNone(y)
        self.assertTrue(diagnostics['refused'])

    def test_cross_section_cache_reused(self):
        """Repeated queries at the same station reuse the cached cross-section surface."""
        trivariate = self.vol._volume
        trivariate.eval(x1=0.6, x2=0.4, x3=45.0, extrapolate=True)
        trivariate.eval(x1=0.6, x2=0.5, x3=45.0, extrapolate=True)
        self.assertEqual(len(trivariate._cross_section_cache), 1)

    def test_array_input_builds_outer_product_grid(self):
        """
        x1, x2, x3 arrays combine as a rectangular grid (outer product),
        mirroring ParametricBivariateSpline.eval's (x1, x2) convention --
        works even when the arrays have different lengths.
        """
        x1 = np.array([0.2, 0.4, 0.6])
        x2 = np.array([0.1, 0.3, 0.5, 0.7])
        x3 = np.array([5.0, 15.0])

        y_grid = self.vol.eval(x1, x2, x3)
        self.assertEqual(y_grid.shape, (3, 4, 2))

        for i, a in enumerate(x1):
            for j, b in enumerate(x2):
                for k, c in enumerate(x3):
                    expected = self.vol.eval(x1=a, x2=b, x3=c)
                    self.assertAlmostEqual(y_grid[i, j, k], expected, places=10)

    def test_array_input_broadcasts_scalar_axis(self):
        x1 = np.array([0.2, 0.4, 0.6, 0.8])
        x2 = np.array([0.1, 0.3, 0.5, 0.7])

        y = self.vol.eval(x1, x2, 20.0)
        self.assertEqual(y.shape, (4, 4, 1))

    def test_array_input_with_extrapolation_and_refusal(self):
        """Mixed in-range/extrapolated/refused points in one grid call."""
        x1 = np.array([0.6])
        x2 = np.array([0.4])
        x3 = np.array([25.0, 20.0, 1000.0])  # last one exceeds the w-distance guard

        y = self.vol.eval(x1, x2, x3, extrapolate=True)
        self.assertEqual(y.shape, (1, 1, 3))
        self.assertFalse(np.isnan(y[0, 0, 0]))
        self.assertFalse(np.isnan(y[0, 0, 1]))
        self.assertTrue(np.isnan(y[0, 0, 2]))

    def test_array_input_diagnostics(self):
        x1 = np.array([0.6])
        x2 = np.array([0.4])
        x3 = np.array([25.0, 1000.0])

        y, diagnostics = self.vol.eval(x1, x2, x3, extrapolate=True, return_diagnostics=True)
        self.assertEqual(diagnostics.shape, (1, 1, 2))
        self.assertFalse(diagnostics[0, 0, 0]['refused'])
        self.assertTrue(diagnostics[0, 0, 1]['refused'])

    def test_list_input_accepted(self):
        y = self.vol.eval([0.2, 0.4], [0.1, 0.3], [5.0, 15.0])
        self.assertEqual(y.shape, (2, 2, 2))


if __name__ == "__main__":
    unittest.main()
