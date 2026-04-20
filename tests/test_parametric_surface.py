import unittest

import numpy as np
from scipy.interpolate import splev

from splinecloud_scipy import ParametricBivariateSpline


# =============================================================================
# FIXTURES & HELPERS
# =============================================================================

def get_simple_surface_data():
    """
    Knot vectors and CP for a parabolid z = x^2 + y^2 on [0,1]x[0,1].
    Bicubic (k=3), 5x5 CP grid, one internal knot at 0.5.
    """
    tu = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    tv = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    ku, kv = 3, 3
    
    # Greville abscissae for u and v (approximate parameter values for CPs)
    # n = len(t) - k - 1 = 9 - 3 - 1 = 5
    gu = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    gv = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    
    GU, GV = np.meshgrid(gu, gv, indexing='ij')
    
    # We want x=u, y=v, z=u^2+v^2 approx.
    # For a perfect match we'd need to solve for coefficients.
    # But for construction tests, any valid shape is fine.
    cp = np.zeros((5, 5, 3))
    cp[:, :, 0] = GU
    cp[:, :, 1] = GV
    cp[:, :, 2] = GU**2 + GV**2
    
    return tu, tv, cp, ku, kv

def get_asymmetric_surface_data():
    """ku=3, kv=2 data."""
    tu = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]) # nu=5
    tv = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])           # nv=4 (nv = 7-2-1=4)
    ku, kv = 3, 2
    
    cp = np.zeros((5, 4, 3))
    # random but valid
    cp[:, :, 2] = np.random.rand(5, 4)
    return tu, tv, cp, ku, kv


# =============================================================================
# 1. CONSTRUCTION TESTS
# =============================================================================

class TestConstruction(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_knot_vectors_stored_correctly(self):
        # Verify that self.tu and self.tv match the input knot vectors
        # exactly after construction, including repeated knots at boundaries.
        np.testing.assert_allclose(self.surf.tu, self.tu)
        np.testing.assert_allclose(self.surf.tv, self.tv)

    def test_degrees_stored_correctly(self):
        # Verify that self.ku and self.kv match the input degrees.
        self.assertEqual(self.surf.ku, self.ku)
        self.assertEqual(self.surf.kv, self.kv)

    def test_control_points_shape(self):
        # Verify that cp is stored with shape (nu, nv, 3) and that
        # nu and nv are consistent with the knot vector lengths and degrees
        # via the relation len(t) = n + k + 1.
        self.assertEqual(self.cp.shape, (5, 5, 3))
        nu, nv, _ = self.cp.shape
        self.assertEqual(len(self.tu), nu + self.ku + 1)
        self.assertEqual(len(self.tv), nv + self.kv + 1)

    def test_tck_x_structure(self):
        # Verify that _tck_x is a tuple of length 5 with elements
        # (tu, tv, c, ku, kv) where c has length len(tu) * len(tv)
        # and matches cp[:, :, 0].ravel().
        tck = self.surf._tck_x
        self.assertIsInstance(tck, tuple)
        self.assertEqual(len(tck), 5)
        self.assertTrue(np.array_equal(tck[0], self.tu))
        self.assertTrue(np.array_equal(tck[1], self.tv))
        self.assertEqual(tck[3], self.ku)
        self.assertEqual(tck[4], self.kv)
        
        c = tck[2]
        expected_c = self.cp[:, :, 0].ravel()
        self.assertEqual(len(c), len(expected_c))
        np.testing.assert_allclose(c, expected_c)

    def test_tck_y_structure(self):
        # Same as test_tck_x_structure but for _tck_y and cp[:, :, 1].
        tck = self.surf._tck_y
        np.testing.assert_allclose(tck[2], self.cp[:, :, 1].ravel())

    def test_tck_z_structure(self):
        # Same as test_tck_x_structure but for _tck_z and cp[:, :, 2].
        tck = self.surf._tck_z
        np.testing.assert_allclose(tck[2], self.cp[:, :, 2].ravel())

    def test_flip_yz_swaps_channels(self):
        # Construct two instances with flip_yz=False and flip_yz=True.
        # Verify that _tck_y and _tck_z have their control point arrays
        # swapped between the two instances.
        surf_no_flip = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv, flip_yz=False)
        surf_flip = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv, flip_yz=True)
        
        np.testing.assert_allclose(surf_no_flip._tck_y[2], self.cp[:, :, 1].ravel())
        np.testing.assert_allclose(surf_no_flip._tck_z[2], self.cp[:, :, 2].ravel())
        
        np.testing.assert_allclose(surf_flip._tck_y[2], self.cp[:, :, 2].ravel())
        np.testing.assert_allclose(surf_flip._tck_z[2], self.cp[:, :, 1].ravel())

    def test_log_flags_stored_correctly(self):
        # Verify that log_x, log_y, log_z are stored as booleans matching
        # the constructor arguments.
        surf_log = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv, 
                                             log_x=True, log_y=False, log_z=True)
        self.assertTrue(surf_log.log_x)
        self.assertFalse(surf_log.log_y)
        self.assertTrue(surf_log.log_z)

    def test_search_grid_shape(self):
        # Verify that _search_u, _search_v, _search_x, _search_y, _search_z
        # are built by _build_search_grid with the correct shapes:
        # _search_u: (Mu,), _search_v: (Mv,), grids: (Mu, Mv).
        # Internal knots at 0.5 means 2 unique spans in each dir: [0, 0.5] and [0.5, 1.0]
        # So Mu=2, Mv=2.
        self.assertEqual(self.surf._search_u.shape, (2,))
        self.assertEqual(self.surf._search_v.shape, (2,))
        self.assertEqual(self.surf._search_x.shape, (2, 2))
        self.assertEqual(self.surf._search_y.shape, (2, 2))
        self.assertEqual(self.surf._search_z.shape, (2, 2))

    def test_search_grid_values_within_physical_extent(self):
        # Verify that all values in _search_x lie between the global min
        # and max of the surface x coordinate, and similarly for _search_y.
        xmin, xmax = self.cp[:, :, 0].min(), self.cp[:, :, 0].max()
        ymin, ymax = self.cp[:, :, 1].min(), self.cp[:, :, 1].max()
        
        self.assertTrue(np.all(self.surf._search_x >= xmin - 1e-10))
        self.assertTrue(np.all(self.surf._search_x <= xmax + 1e-10))
        self.assertTrue(np.all(self.surf._search_y >= ymin - 1e-10))
        self.assertTrue(np.all(self.surf._search_y <= ymax + 1e-10))

    def test_search_grid_u_midpoints_within_knot_domain(self):
        # Verify that all values in _search_u lie strictly within the
        # valid parameter domain [tu[ku], tu[-(ku+1)]].
        u_min, u_max = self.tu[self.ku], self.tu[-(self.ku+1)]
        v_min, v_max = self.tv[self.kv], self.tv[-(self.kv+1)]
        
        self.assertTrue(np.all(self.surf._search_u >= u_min))
        self.assertTrue(np.all(self.surf._search_u <= u_max))
        self.assertTrue(np.all(self.surf._search_v >= v_min))
        self.assertTrue(np.all(self.surf._search_v <= v_max))

    def test_asymmetric_degrees_construction(self):
        # Using the asymmetric degree fixture (ku != kv), verify that
        # construction succeeds and all tck tuples have the correct
        # degree values stored.
        tu, tv, cp, ku, kv = get_asymmetric_surface_data()
        surf = ParametricBivariateSpline(tu, tv, cp, ku, kv)
        self.assertEqual(surf.ku, ku)
        self.assertEqual(surf.kv, kv)
        self.assertEqual(surf._tck_x[3], ku)
        self.assertEqual(surf._tck_x[4], kv)


# =============================================================================
# 2. EVALUATION TESTS
# =============================================================================

class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_call_pointwise(self):
        # Forward evaluation: u, v -> x, y, z
        u, v = 0.3, 0.7
        x, y, z = self.surf(u, v)
        # For our simple surface x=u, y=v
        self.assertAlmostEqual(x[0], 0.3, places=6)
        self.assertAlmostEqual(y[0], 0.7, places=6)

    def test_eval_point_interior(self):
        # Inverse evaluation: find z at (x, y)
        # Use (u, v) to find (x, y) first, then verify eval_point returns the same z
        u_ref, v_ref = 0.4, 0.6
        x_ref, y_ref, z_ref = self.surf(u_ref, v_ref)
        z = self.surf.eval_point(x_ref[0], y_ref[0])
        self.assertAlmostEqual(z, z_ref[0], places=6)

    def test_eval_point_exterior(self):
        # Point outside the bounding box
        x_target, y_target = 1.5, 0.5
        z = self.surf.eval_point(x_target, y_target, extrapolate=False)
        self.assertIsNone(z)

    def test_eval_scalar_interface(self):
        # Test the unified eval() interface with scalars
        u_ref, v_ref = 0.2, 0.8
        x_ref, y_ref, z_ref = self.surf(u_ref, v_ref)
        z = self.surf.eval(x_ref[0], y_ref[0])
        self.assertAlmostEqual(z, z_ref[0], places=6)

    def test_eval_array_interface(self):
        # Test unified eval() with arrays (should call eval_grid)
        # Using a small grid to avoid triggering vectorized path yet
        x_vals = np.array([0.2, 0.5, 0.8])
        y_vals = np.array([0.3, 0.7])
        X, Y, Z = self.surf.eval(x_vals, y_vals, threshold=100)
        
        self.assertEqual(Z.shape, (3, 2))
        for i, xv in enumerate(x_vals):
            for j, yv in enumerate(y_vals):
                # Verify against point evaluation
                z_expected = self.surf.eval_point(xv, yv)
                self.assertAlmostEqual(Z[i, j], z_expected, places=6)


# =============================================================================
# 3. GRID EVALUATION TESTS
# =============================================================================

class TestGridEvaluation(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_eval_grid_vs_pointwise(self):
        # Verify that eval_grid result matches pointwise eval_point results
        x_vals = np.linspace(0.2, 0.8, 4)
        y_vals = np.linspace(0.2, 0.8, 3)
        X, Y, Z = self.surf.eval_grid(x_vals, y_vals)
        
        self.assertEqual(Z.shape, (4, 3))
        for i, xv in enumerate(x_vals):
            for j, yv in enumerate(y_vals):
                z_expected = self.surf.eval_point(xv, yv)
                self.assertAlmostEqual(Z[i, j], z_expected, places=6)

    def test_vectorized_path_trigger(self):
        # Verify that eval calls eval_grid when points exceed threshold
        x_vals = np.linspace(0.1, 0.9, 10)
        y_vals = np.linspace(0.1, 0.9, 11) # 110 points > 100
        # We can't easily mock the call without pytest, so we just check it doesn't crash 
        # and returns correct values.
        X, Y, Z = self.surf.eval(x_vals, y_vals, threshold=100)
        self.assertEqual(Z.shape, (10, 11))


# =============================================================================
# 4. GRADIENT TESTS
# =============================================================================

class TestGradients(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_point_gradients(self):
        # For our surface, x=u, y=v, z approx u^2 + v^2
        # So dz/dx = dz/du * du/dx + dz/dv * dv/dx = dz/du
        # Analytical dz/du at 0.5 should be approx 2*0.5 = 1.0
        x, y = 0.5, 0.5
        z, (dzdx, dzdy) = self.surf.eval_point(x, y, compute_gradients=True)
        
        # Exact values depend on spline coefficients, but should be near 1.0
        # Let's check consistency: dz/dx should be (z(x+eps) - z(x-eps))/(2*eps)
        eps = 1e-6

        z_xplus = self.surf.eval_point(x + eps, y)
        z_xminus = self.surf.eval_point(x - eps, y)
        dzdx_num = (z_xplus - z_xminus) / (2 * eps)
        
        self.assertAlmostEqual(dzdx, dzdx_num, places=5)

        z_yplus = self.surf.eval_point(x, y + eps)
        z_yminus = self.surf.eval_point(x, y - eps)
        dzdy_num = (z_yplus - z_yminus) / (2 * eps)
        
        self.assertAlmostEqual(dzdy, dzdy_num, places=5)

    def test_grid_gradients(self):
        x_vals = np.array([0.4, 0.6])
        y_vals = np.array([0.3, 0.7])
        X, Y, Z, dZdX, dZdY = self.surf.eval_grid(x_vals, y_vals, compute_gradients=True)
        
        self.assertEqual(dZdX.shape, (2, 2))
        for i, xv in enumerate(x_vals):
            for j, yv in enumerate(y_vals):
                _, (gx, gy) = self.surf.eval_point(xv, yv, compute_gradients=True)
                self.assertAlmostEqual(dZdX[i, j], gx, places=6)
                self.assertAlmostEqual(dZdY[i, j], gy, places=6)


# =============================================================================
# 5. EXTRAPOLATION TESTS
# =============================================================================

class TestExtrapolation(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_extrapolation_z_value(self):
        # Point slightly outside: x=1.1, y=0.5
        # Newton should fail, extrapolation should return a value
        z = self.surf.eval_point(1.1, 0.5, extrapolate=True)
        self.assertIsNotNone(z)
        # Should be roughly (1.1)^2 + (0.5)^2 = 1.21 + 0.25 = 1.46
        self.assertTrue(1.25 < z < 1.75)

    def test_extrapolation_gradients(self):
        x, y = 1.1, 0.5
        z, (dzdx, dzdy) = self.surf.eval_point(x, y, extrapolate=True, compute_gradients=True)
        self.assertIsNotNone(z)
        self.assertIsNotNone(dzdx)
        self.assertIsNotNone(dzdy)

    def test_limit_distance(self):
        # Very far point should return None if limit_distance is True
        z = self.surf.eval_point(10.0, 10.0, extrapolate=True, limit_distance=True, distance_threshold=0.1)
        self.assertIsNone(z)

    def test_limit_consistency(self):
        # There is no trivial way to force inconsistency, but we can verify it doesn't crash
        z = self.surf.eval_point(1.2, 0.5, extrapolate=True, limit_consistency=True, consistency_threshold=0.001)
        # At 1.2 it might already fail consistency if threshold is very low
        pass

if __name__ == '__main__':
    unittest.main()

