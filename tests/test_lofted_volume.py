import unittest

import numpy as np
from splipy import Surface

from splinecloud_scipy import ParametricBivariateSpline, LoftedSplineVolume
from splinecloud_scipy.lofted_spline_volume import pbs_to_splipy


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


def make_surface_variant(x3_offset, ku=3, kv=3, u_interior=(0.5,), v_interior=(0.5,)):
    """
    Same y = x1^2 + x2^2 + x3_offset surface as make_surface, but with a
    configurable degree and interior-knot placement per direction, so
    surfaces with genuinely different structure (knot vectors, basis
    functions, order) can be built for lofting.
    """
    tu = np.concatenate(([0.0] * (ku + 1), sorted(u_interior), [1.0] * (ku + 1)))
    tv = np.concatenate(([0.0] * (kv + 1), sorted(v_interior), [1.0] * (kv + 1)))
    nu = len(tu) - ku - 1
    nv = len(tv) - kv - 1

    gu = np.linspace(0.0, 1.0, nu)
    gv = np.linspace(0.0, 1.0, nv)
    GU, GV = np.meshgrid(gu, gv, indexing='ij')

    cp = np.zeros((nu, nv, 3))
    cp[:, :, 0] = GU
    cp[:, :, 1] = GU**2 + GV**2 + x3_offset
    cp[:, :, 2] = GV

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


class TestLoftedSplineVolumeHeterogeneousSurfaces(unittest.TestCase):
    """
    make_surface() always produces bicubic surfaces sharing the same knot
    vector, so it never exercises Surface.make_splines_identical's actual
    job: reconciling surfaces with different knot vectors and/or orders.
    These tests build surfaces that genuinely differ in structure, check
    the reconciliation step directly (white-box), and check that
    LoftedSplineVolume built from such surfaces still evaluates correctly
    (black-box) -- including the case of different orders per surface.
    """

    def test_make_splines_identical_reconciles_different_knot_vectors(self):
        """
        Two same-degree surfaces with different interior knots: after
        reconciliation both must share one knot vector (a superset of
        each original's interior knots -- refinement only inserts, never
        removes) and the same control-net shape.
        """
        p1 = make_surface_variant(0.0, ku=3, kv=3, u_interior=(0.3,), v_interior=(0.2,))
        p2 = make_surface_variant(10.0, ku=3, kv=3, u_interior=(0.7,), v_interior=(0.8,))

        s1, _ = pbs_to_splipy(p1)
        s2, _ = pbs_to_splipy(p2)
        s1, s2 = s1.clone(), s2.clone()
        Surface.make_splines_identical(s1, s2)

        np.testing.assert_allclose(s1.bases[0].knots, s2.bases[0].knots)
        np.testing.assert_allclose(s1.bases[1].knots, s2.bases[1].knots)
        self.assertEqual(s1.controlpoints.shape, s2.controlpoints.shape)

        for original_knot in (0.3, 0.7):
            self.assertTrue(np.any(np.isclose(s1.bases[0].knots, original_knot)))
        for original_knot in (0.2, 0.8):
            self.assertTrue(np.any(np.isclose(s1.bases[1].knots, original_knot)))

    def test_make_splines_identical_reconciles_different_orders(self):
        """Two surfaces with different degree are raised to their common max order."""
        p1 = make_surface_variant(0.0, ku=2, kv=2)
        p2 = make_surface_variant(10.0, ku=4, kv=4)

        s1, _ = pbs_to_splipy(p1)
        s2, _ = pbs_to_splipy(p2)
        s1, s2 = s1.clone(), s2.clone()

        self.assertEqual((s1.order(0), s1.order(1)), (3, 3))
        self.assertEqual((s2.order(0), s2.order(1)), (5, 5))

        Surface.make_splines_identical(s1, s2)

        self.assertEqual(s1.order(0), 5)
        self.assertEqual(s1.order(1), 5)
        self.assertEqual((s1.order(0), s1.order(1)), (s2.order(0), s2.order(1)))
        self.assertEqual(s1.controlpoints.shape, s2.controlpoints.shape)

    def test_lofted_volume_from_surfaces_with_different_knot_vectors(self):
        """Same degree, different interior knots -- must build and evaluate correctly."""
        x3_values = [0.0, 10.0]
        surfaces = [
            make_surface_variant(0.0, ku=3, kv=3, u_interior=(0.3,), v_interior=(0.3,)),
            make_surface_variant(10.0, ku=3, kv=3, u_interior=(0.7,), v_interior=(0.7,)),
        ]
        vol = LoftedSplineVolume(surfaces, x3_values, w_degree=1)

        y = vol.eval(x1=0.5, x2=0.5, x3=5.0)
        expected = 0.5**2 + 0.5**2 + 5.0
        self.assertAlmostEqual(y, expected, delta=0.2)

        # exact match against the original surfaces at each station
        for surface, x3 in zip(surfaces, x3_values):
            y_vol = vol.eval(x1=0.5, x2=0.5, x3=x3, extrapolate=True)
            y_src = surface.eval_point(0.5, 0.5)
            self.assertAlmostEqual(y_vol, y_src, places=8)

    def test_lofted_volume_from_surfaces_with_different_orders(self):
        """
        Surfaces of different order ARE valid LoftedSplineVolume inputs:
        make_splines_identical raises the lower-degree surface's order to
        match, so a quadratic and a quartic surface loft together fine.
        """
        x3_values = [0.0, 10.0]
        surfaces = [
            make_surface_variant(0.0, ku=2, kv=2),
            make_surface_variant(10.0, ku=4, kv=4),
        ]
        vol = LoftedSplineVolume(surfaces, x3_values, w_degree=1)

        y = vol.eval(x1=0.5, x2=0.5, x3=5.0)
        expected = 0.5**2 + 0.5**2 + 5.0
        self.assertAlmostEqual(y, expected, delta=0.2)

        for surface, x3 in zip(surfaces, x3_values):
            y_vol = vol.eval(x1=0.5, x2=0.5, x3=x3, extrapolate=True)
            y_src = surface.eval_point(0.5, 0.5)
            self.assertAlmostEqual(y_vol, y_src, places=8)

    def test_lofted_volume_from_many_structurally_different_surfaces(self):
        """
        Three surfaces, each different in a different way (asymmetric
        u/v degree, distinct knot placement, distinct grid resolution):
        exercises _build_volume's all-pairs reconciliation loop, not just
        a single pair, and checks it converges to one common basis for
        every surface, not just adjacent ones.
        """
        x3_values = [0.0, 10.0, 20.0]
        surfaces = [
            make_surface_variant(0.0, ku=2, kv=4, u_interior=(0.2,), v_interior=(0.4, 0.6)),
            make_surface_variant(10.0, ku=3, kv=3, u_interior=(0.5,), v_interior=(0.5,)),
            make_surface_variant(20.0, ku=4, kv=2, u_interior=(0.3, 0.7), v_interior=(0.8,)),
        ]
        vol = LoftedSplineVolume(surfaces, x3_values, w_degree=2)

        for surface, x3 in zip(surfaces, x3_values):
            y_vol = vol.eval(x1=0.5, x2=0.5, x3=x3, extrapolate=True)
            y_src = surface.eval_point(0.5, 0.5)
            self.assertAlmostEqual(y_vol, y_src, places=8)

    def test_degree_one_surfaces_fail_under_cross_section_extrapolation(self):
        """
        Known limitation, surfaced by structurally minimal (degree 1)
        inputs: when every lofted surface is degree 1 in some direction,
        make_splines_identical has nothing higher to raise it to, so the
        cross-section built for extrapolate=True is itself degree 1.
        ParametricBivariateSpline.eval_point needs first derivatives
        (bisplev requires dx < kx) to run its Newton solve, which is
        impossible at degree 1 -- so extrapolate=True raises, even though
        the default extrapolate=False path (splipy's own Volume Newton
        solve, unrelated to bisplev) still works fine. Not introduced by
        lofting -- ParametricBivariateSpline.eval_point already fails the
        same way standalone -- but only visible in this class once the
        common degree across all lofted surfaces is 1.
        """
        x3_values = [0.0, 10.0]
        surfaces = [
            make_surface_variant(0.0, ku=1, kv=1, u_interior=(0.3,), v_interior=(0.3,)),
            make_surface_variant(10.0, ku=1, kv=1, u_interior=(0.7,), v_interior=(0.7,)),
        ]
        vol = LoftedSplineVolume(surfaces, x3_values, w_degree=1)

        y_clipped = vol.eval(x1=0.13, x2=0.87, x3=5.0, extrapolate=False)
        self.assertIsNotNone(y_clipped)

        with self.assertRaises(ValueError):
            vol.eval(x1=0.13, x2=0.87, x3=5.0, extrapolate=True)


if __name__ == "__main__":
    unittest.main()
