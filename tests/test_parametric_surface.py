import unittest, time

import numpy as np
from scipy.interpolate import splev, bisplev

from splinecloud_scipy import ParametricBivariateSpline


# =============================================================================
# FIXTURES & HELPERS
# =============================================================================

def get_simple_surface_data():
    """
    Knot vectors and CP for a surface y = x1^2 + x2^2 on [0,1]x[0,1].
    Bicubic (k=3), 5x5 CP grid, one internal knot at 0.5.
    CP layout: [x1, y, x2] where y is dependent.
    """
    tu = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    tv = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    ku, kv = 3, 3
    
    # Greville abscissae for u and v (approximate parameter values for CPs)
    # n = len(t) - k - 1 = 9 - 3 - 1 = 5
    gu = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    gv = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    
    GU, GV = np.meshgrid(gu, gv, indexing='ij')
    
    # x1=u, x2=v, y=u^2+v^2 (dependent)
    cp = np.zeros((5, 5, 3))
    cp[:, :, 0] = GU              # x1
    cp[:, :, 1] = GU**2 + GV**2   # y (dependent)
    cp[:, :, 2] = GV              # x2
    
    return tu, tv, cp, ku, kv

def get_asymmetric_surface_data():
    """ku=3, kv=2 data. CP layout: [x1, y, x2]."""
    tu = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0]) # nu=5
    tv = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])           # nv=4
    ku, kv = 3, 2
    
    # Greville abscissae (non-degenerate mapping)
    gu = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0]) 
    gv = np.array([0.0, 0.25, 0.75, 1.0])                 
    GU, GV = np.meshgrid(gu, gv, indexing='ij')
    
    cp = np.zeros((5, 4, 3))
    cp[:, :, 0] = GU                    # x1
    cp[:, :, 1] = np.random.rand(5, 4)  # y (dependent)
    cp[:, :, 2] = GV                    # x2
    return tu, tv, cp, ku, kv


# =============================================================================
# 1. CONSTRUCTION TESTS
# =============================================================================

class TestConstruction(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_knot_vectors_stored_correctly(self):
        np.testing.assert_allclose(self.surf.tu, self.tu)
        np.testing.assert_allclose(self.surf.tv, self.tv)

    def test_degrees_stored_correctly(self):
        self.assertEqual(self.surf.ku, self.ku)
        self.assertEqual(self.surf.kv, self.kv)

    def test_control_points_shape(self):
        self.assertEqual(self.cp.shape, (5, 5, 3))
        nu, nv, _ = self.cp.shape
        self.assertEqual(len(self.tu), nu + self.ku + 1)
        self.assertEqual(len(self.tv), nv + self.kv + 1)

    def test_tck_x1_structure(self):
        # Verify _tck_x1 matches cp[:, :, 0].ravel().
        tck = self.surf._tck_x1
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
        # _tck_y matches cp[:, :, 1] (dependent axis).
        tck = self.surf._tck_y
        np.testing.assert_allclose(tck[2], self.cp[:, :, 1].ravel())

    def test_tck_x2_structure(self):
        # _tck_x2 matches cp[:, :, 2].
        tck = self.surf._tck_x2
        np.testing.assert_allclose(tck[2], self.cp[:, :, 2].ravel())

    def test_log_flags_stored_correctly(self):
        surf_log = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv, 
                                             log_x1=True, log_x2=False, log_y=True)
        self.assertTrue(surf_log.log_x1)
        self.assertFalse(surf_log.log_x2)
        self.assertTrue(surf_log.log_y)

    def test_search_grid_shape(self):
        self.assertEqual(self.surf._search_u.shape, (2,))
        self.assertEqual(self.surf._search_v.shape, (2,))
        self.assertEqual(self.surf._search_x1.shape, (2, 2))
        self.assertEqual(self.surf._search_x2.shape, (2, 2))
        self.assertEqual(self.surf._search_y.shape, (2, 2))

    def test_search_grid_values_within_physical_extent(self):
        x1min, x1max = self.cp[:, :, 0].min(), self.cp[:, :, 0].max()
        x2min, x2max = self.cp[:, :, 2].min(), self.cp[:, :, 2].max()
        
        self.assertTrue(np.all(self.surf._search_x1 >= x1min - 1e-10))
        self.assertTrue(np.all(self.surf._search_x1 <= x1max + 1e-10))
        self.assertTrue(np.all(self.surf._search_x2 >= x2min - 1e-10))
        self.assertTrue(np.all(self.surf._search_x2 <= x2max + 1e-10))

    def test_search_grid_u_midpoints_within_knot_domain(self):
        u_min, u_max = self.tu[self.ku], self.tu[-(self.ku+1)]
        v_min, v_max = self.tv[self.kv], self.tv[-(self.kv+1)]
        
        self.assertTrue(np.all(self.surf._search_u >= u_min))
        self.assertTrue(np.all(self.surf._search_u <= u_max))
        self.assertTrue(np.all(self.surf._search_v >= v_min))
        self.assertTrue(np.all(self.surf._search_v <= v_max))

    def test_asymmetric_degrees_construction(self):
        tu, tv, cp, ku, kv = get_asymmetric_surface_data()
        surf = ParametricBivariateSpline(tu, tv, cp, ku, kv)
        self.assertEqual(surf.ku, ku)
        self.assertEqual(surf.kv, kv)
        self.assertEqual(surf._tck_x1[3], ku)
        self.assertEqual(surf._tck_x1[4], kv)


# =============================================================================
# 2. FORWARD EVALUATION TESTS (__call__)
# =============================================================================

class TestCall(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_call_pointwise(self):
        # Forward evaluation: u, v -> x1, x2, y_val
        u, v = 0.3, 0.7
        x1, x2, y_val = self.surf(u, v)
        # For our simple surface x=u, y=v
        self.assertAlmostEqual(x1, 0.3, places=6)
        self.assertAlmostEqual(x2, 0.7, places=6)
    
    def test_scalar_input_returns_scalar(self):
        # Call __call__ with scalar u and v values.
        # Verify that returned x1, x2, y_val are Python floats, not arrays.
        u, v = 0.3, 0.7
        x1, x2, y_val = self.surf(u, v)
        self.assertIsInstance(x1, float)
        self.assertIsInstance(x2, float)
        self.assertIsInstance(y_val, float)

    def test_scalar_input_matches_bisplev(self):
        # Call __call__ with scalar u and v.
        # Verify that returned x1, x2, y_val match bisplev called directly on
        # _tck_x, _tck_y, _tck_z with the same scalar inputs, to
        # floating-point precision.
        
        u, v = 0.3, 0.7
        x1, x2, y_val = self.surf(u, v)
        
        x1b = float(bisplev(u, v, self.surf._tck_x1))
        x2b = float(bisplev(u, v, self.surf._tck_x2))
        yb = float(bisplev(u, v, self.surf._tck_y))
        
        self.assertAlmostEqual(x1, x1b)
        self.assertAlmostEqual(x2, x2b)
        self.assertAlmostEqual(y_val, yb)

    def test_array_input_returns_2d_grid(self):
        # Call __call__ with u of length Nu and v of length Nv where Nu != Nv.
        # Verify that x1, x2, y_val each have shape (Nu, Nv).
        u = np.array([0.1, 0.2, 0.3])
        v = np.array([0.4, 0.5])
        x1, x2, y_val = self.surf(u, v)
        self.assertEqual(x1.shape, (3, 2))
        self.assertEqual(x2.shape, (3, 2))
        self.assertEqual(y_val.shape, (3, 2))

    def test_equal_length_arrays_return_2d_grid(self):
        # Call __call__ with u and v arrays of equal length N.
        # Verify that x1, x2, y_val each have shape (N, N), confirming that
        # equal-length arrays still produce a full grid, not a diagonal.
        N = 4
        u = np.linspace(0.1, 0.4, N)
        v = np.linspace(0.5, 0.8, N)
        x1, x2, y_val = self.surf(u, v)
        self.assertEqual(x1.shape, (N, N))
        self.assertEqual(x2.shape, (N, N))
        self.assertEqual(y_val.shape, (N, N))

    def test_array_output_matches_bisplev(self):
        # Call __call__ with u of length Nu and v of length Nv.
        # Verify that x1, x2, y_val match bisplev called directly on _tck_x1,
        # _tck_y, _tck_x2 with the same arrays, to floating-point precision.
        
        u = np.linspace(0.1, 0.9, 5)
        v = np.linspace(0.1, 0.9, 7)
        x1, x2, y_val = self.surf(u, v)
        
        x1b = bisplev(u, v, self.surf._tck_x1)
        x2b = bisplev(u, v, self.surf._tck_x2)
        yb = bisplev(u, v, self.surf._tck_y)
        
        np.testing.assert_allclose(x1, x1b)
        np.testing.assert_allclose(x2, x2b)
        np.testing.assert_allclose(y_val, yb)

    def test_boundary_parameter_values_no_nan(self):
        # Call __call__ as scalar at the four boundary parameter corners:
        # (tu[ku], tv[kv]), (tu[ku], tv[-(kv+1)]),
        # (tu[-(ku+1)], tv[kv]), (tu[-(ku+1)], tv[-(kv+1)]).
        # Verify that no NaN or Inf values are returned.
        u_min, u_max = self.surf.tu[self.surf.ku], self.surf.tu[-(self.surf.ku + 1)]
        v_min, v_max = self.surf.tv[self.surf.kv], self.surf.tv[-(self.surf.kv + 1)]
        
        for u in [u_min, u_max]:
            for v in [v_min, v_max]:
                x1, x2, y_val = self.surf(u, v)
                self.assertTrue(np.isfinite(x1))
                self.assertTrue(np.isfinite(x2))
                self.assertTrue(np.isfinite(y_val))

    def test_asymmetric_input_lengths(self):
        # Call __call__ with u of length 7 and v of length 13 (deliberately
        # asymmetric and not related to knot counts). Verify output shape
        # is (7, 13) and values match bisplev directly.
        
        u = np.linspace(0.1, 0.9, 7)
        v = np.linspace(0.1, 0.9, 13)
        x1, x2, y_val = self.surf(u, v)
        self.assertEqual(x1.shape, (7, 13))
        
        x1b = bisplev(u, v, self.surf._tck_x1)
        np.testing.assert_allclose(x1, x1b)


# =============================================================================
# 3. INVERSE EVALUATION TESTS (eval_point)
# =============================================================================

class TestEvaluation(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_eval_point_interior(self):
        # Inverse evaluation: find z at (x, y)
        # Use (u, v) to find (x, y) first, then verify eval_point returns the same z
        u_ref, v_ref = 0.4, 0.6
        x1_ref, x2_ref, y_ref = self.surf(u_ref, v_ref)
        y_val = self.surf.eval_point(x1_ref, x2_ref)
        self.assertAlmostEqual(y_val, y_ref, places=6)

    def test_eval_point_exterior(self):
        # Point outside the bounding box
        x1_target, x2_target = 1.5, 0.5
        y_val = self.surf.eval_point(x1_target, x2_target, extrapolate=False)
        self.assertIsNone(y_val)

    def test_eval_scalar_interface(self):
        # Test the unified eval() interface with scalars
        u_ref, v_ref = 0.2, 0.8
        x1_ref, x2_ref, y_ref = self.surf(u_ref, v_ref)
        y_val = self.surf.eval(x1_ref, x2_ref)
        self.assertAlmostEqual(y_val, y_ref, places=6)

    def test_eval_array_interface(self):
        # Test unified eval() with arrays (should call eval_grid)
        # Using a small grid to avoid triggering vectorized path yet
        x1_vals = np.array([0.2, 0.5, 0.8])
        x2_vals = np.array([0.3, 0.7])
        X1, X2, Y = self.surf.eval(x1_vals, x2_vals, threshold=100)
        
        self.assertEqual(X2.shape, (3, 2))
        for i, xv in enumerate(x1_vals):
            for j, yv in enumerate(x2_vals):
                # Verify against point evaluation
                y_expected = self.surf.eval_point(xv, yv)
                self.assertAlmostEqual(Y[i, j], y_expected, places=6)

    def test_consistent_with_eval_grid_scalar(self):
        # For a single (x, y) point, verify that eval_point and eval_grid
        # called with 1-element arrays return the same z value.
        x1, x2 = 0.5, 0.5
        y_point = self.surf.eval_point(x1, x2)
        X1, X2, Y_grid = self.surf.eval_grid([x1], [x2])
        self.assertAlmostEqual(y_point, Y_grid[0, 0])

    def test_newton_convergence_near_boundary(self):
        # Sample points near but strictly inside the parameter boundary.
        # Verify that eval_point converges and returns values consistent
        # with __call__ at those boundary-adjacent parameters.
        u_min, u_max = self.surf.tu[self.surf.ku], self.surf.tu[-(self.surf.ku + 1)]
        v_min, v_max = self.surf.tv[self.surf.kv], self.surf.tv[-(self.surf.kv + 1)]
        
        eps = 1e-8
        test_points = [
            (u_min + eps, v_min + eps),
            (u_max - eps, v_min + eps),
            (u_min + eps, v_max - eps),
            (u_max - eps, v_max - eps),
        ]
        
        for u, v in test_points:
            x1_ref, x2_ref, y_ref = self.surf(u, v)
            y_val = self.surf.eval_point(x1_ref, x2_ref)
            self.assertIsNotNone(y_val)
            self.assertAlmostEqual(y_val, y_ref, places=6)

    def test_asymmetric_degrees_inversion(self):
        # Using the asymmetric degree fixture, verify that eval_point
        # correctly inverts __call__ for a set of interior (x, y) points.
        tu, tv, cp, ku, kv = get_asymmetric_surface_data()
        surf = ParametricBivariateSpline(tu, tv, cp, ku, kv)
        
        # Sample an interior point
        u_ref, v_ref = 0.5, 0.5
        x1_ref, x2_ref, y_ref = surf(u_ref, v_ref)
        y_val = surf.eval_point(x1_ref, x2_ref)
        self.assertIsNotNone(y_val)
        self.assertAlmostEqual(y_val, y_ref, places=6)

# =============================================================================
# 4. GRID EVALUATION TESTS (eval_grid)
# =============================================================================

class TestGridEvaluation(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_eval_grid_vs_pointwise(self):
        # Verify that eval_grid result matches pointwise eval_point results
        x1_vals = np.linspace(0.2, 0.8, 4)
        x2_vals = np.linspace(0.2, 0.8, 3)
        X1, X2, Y = self.surf.eval_grid(x1_vals, x2_vals)
        
        self.assertEqual(X2.shape, (4, 3))
        for i, xv in enumerate(x1_vals):
            for j, yv in enumerate(x2_vals):
                y_expected = self.surf.eval_point(xv, yv)
                self.assertAlmostEqual(Y[i, j], y_expected, places=6)

    def test_output_shape(self):
        # Call eval_grid with x1_vals of length Nx and x2_vals of length Ny.
        # Verify that X, Y, Z each have shape (Nx, Ny).
        Nx, Ny = 5, 7
        x1_vals = np.linspace(0.1, 0.9, Nx)
        x2_vals = np.linspace(0.1, 0.9, Ny)
        X1, X2, Y = self.surf.eval_grid(x1_vals, x2_vals)
        self.assertEqual(X1.shape, (Nx, Ny))
        self.assertEqual(X2.shape, (Nx, Ny))
        self.assertEqual(X2.shape, (Nx, Ny))

    def test_no_nan_for_interior_points(self):
        # Call eval_grid on a grid of (x, y) points known to be inside the
        # surface domain. Verify that no NaN or Inf values appear in Z.
        x1_vals = np.linspace(0.1, 0.9, 20)
        x2_vals = np.linspace(0.1, 0.9, 20)
        X1, X2, Y = self.surf.eval_grid(x1_vals, x2_vals)
        self.assertTrue(np.all(np.isfinite(Y)))

    def test_large_grid_performance(self):
        # Call eval_grid on a 200x200 grid and verify it completes within
        # a reasonable time bound (e.g. 30 seconds).This is a regression guard 
        # against performance degradation.
        
        N = 200
        x1_vals = np.linspace(0.1, 0.9, N)
        x2_vals = np.linspace(0.1, 0.9, N)
        
        start = time.time()
        X1, X2, Y = self.surf.eval_grid(x1_vals, x2_vals)
        duration = time.time() - start
        
        self.assertLess(duration, 30.0)
        self.assertEqual(X2.shape, (N, N))

# =============================================================================
# 5. GRADIENT TESTS
# =============================================================================

class TestGradients(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_eval_point_gradient_shape(self):
        # Call eval_point with compute_gradients=True. Verify the return
        # value is (z, (dydx1, dydx2)) where z, dydx1, dydx2 are all floats.
        x1, x2 = 0.5, 0.5
        y_val, (dydx1, dydx2) = self.surf.eval_point(x1, x2, compute_gradients=True)
        self.assertIsInstance(y_val, float)
        self.assertIsInstance(dydx1, float)
        self.assertIsInstance(dydx2, float)

    def test_eval_grid_gradient_shape(self):
        # Call eval_grid with compute_gradients=True. Verify the return
        # value is (X, Y, Z, dZdX, dZdY) where each is shape (Nx, Ny).
        Nx, Ny = 3, 4
        x1_vals = np.linspace(0.4, 0.6, Nx)
        x2_vals = np.linspace(0.4, 0.6, Ny)
        X1, X2, Y, dYdX1, dYdX2 = self.surf.eval_grid(x1_vals, x2_vals, compute_gradients=True)
        self.assertEqual(X1.shape, (Nx, Ny))
        self.assertEqual(X2.shape, (Nx, Ny))
        self.assertEqual(X2.shape, (Nx, Ny))
        self.assertEqual(dYdX1.shape, (Nx, Ny))
        self.assertEqual(dYdX2.shape, (Nx, Ny))

    def test_gradient_returns_none_on_failed_inversion(self):
        # Call eval_point with compute_gradients=True on a point outside
        # the domain with extrapolate=False. Verify return is (None, (None, None)).
        y_val, (dydx1, dydx2) = self.surf.eval_point(2.0, 2.0, compute_gradients=True, extrapolate=False)
        self.assertIsNone(y_val)
        self.assertIsNone(dydx1)
        self.assertIsNone(dydx2)

    def test_eval_grid_gradients_are_none_on_failed_inversion(self):
        # Call eval_grid with compute_gradients=True on a grid of points outside
        # the domain with extrapolate=False. Verify return is (X1, X2, Y, dYdX1, dYdX2)
        # where dYdX1 and dYdX2 are all None.
        Nx, Ny = 3, 4
        x1_vals = np.linspace(2.0, 3.0, Nx)
        x2_vals = np.linspace(2.0, 3.0, Ny)
        X1, X2, Y, dYdX1, dYdX2 = self.surf.eval_grid(
            x1_vals, x2_vals, compute_gradients=True, extrapolate=False)

        self.assertTrue(np.all(np.isnan(dYdX1)))
        self.assertTrue(np.all(np.isnan(dYdX2)))

    def test_eval_point_gradients(self):
        x1, x2 = 0.5, 0.5
        y_val, (dydx1, dydx2) = self.surf.eval_point(x1, x2, compute_gradients=True)
        
        eps = 1e-6
        y_x1plus = self.surf.eval_point(x1 + eps, x2)
        y_x1minus = self.surf.eval_point(x1 - eps, x2)
        dydx1_calc = (y_x1plus - y_x1minus) / (2 * eps)

        y_x2plus = self.surf.eval_point(x1, x2 + eps)
        y_x2minus = self.surf.eval_point(x1, x2 - eps)
        dydx2_calc = (y_x2plus - y_x2minus) / (2 * eps)
        
        self.assertAlmostEqual(dydx1, dydx1_calc, places=5)
        self.assertAlmostEqual(dydx2, dydx2_calc, places=5)

    def test_eval_point_gradients_on_grid(self):
        # Perform similar check but for several points inside domain
        x1_vals = [0.3, 0.5, 0.7]
        x2_vals = [0.2, 0.4, 0.8]
        eps = 1e-6
        
        for x1 in x1_vals:
            for x2 in x2_vals:
                y_val, (dydx1, dydx2) = self.surf.eval_point(x1, x2, compute_gradients=True)
                
                y_x1plus = self.surf.eval_point(x1 + eps, x2)
                y_x1minus = self.surf.eval_point(x1 - eps, x2)
                dydx1_calc = (y_x1plus - y_x1minus) / (2 * eps)

                y_x2plus = self.surf.eval_point(x1, x2 + eps)
                y_x2minus = self.surf.eval_point(x1, x2 - eps)
                dydx2_calc = (y_x2plus - y_x2minus) / (2 * eps)
                
                self.assertAlmostEqual(dydx1, dydx1_calc, places=5)
                self.assertAlmostEqual(dydx2, dydx2_calc, places=5)

    def test_asymmetric_degrees_gradient(self):
        # Using the asymmetric degree fixture, verify that gradients match
        # finite differences for ku != kv surfaces.
        tu, tv, cp, ku, kv = get_asymmetric_surface_data()
        surf = ParametricBivariateSpline(tu, tv, cp, ku, kv)
        
        x1, x2 = 0.5, 0.5
        y_val, (dydx1, dydx2) = surf.eval_point(x1, x2, compute_gradients=True)
        
        eps = 1e-6
        y_x1plus = surf.eval_point(x1 + eps, x2)
        y_x1minus = surf.eval_point(x1 - eps, x2)
        dydx1_calc = (y_x1plus - y_x1minus) / (2 * eps)
        
        y_x2plus = surf.eval_point(x1, x2 + eps)
        y_x2minus = surf.eval_point(x1, x2 - eps)
        dydx2_calc = (y_x2plus - y_x2minus) / (2 * eps)

        self.assertAlmostEqual(dydx1, dydx1_calc, places=5)
        self.assertAlmostEqual(dydx2, dydx2_calc, places=5)

    def test_eval_grid_gradients(self):
        ## Perform check with calculated gradients using epsilon for each point on grid
        x1_vals = np.linspace(0.3, 0.7, 3)
        x2_vals = np.linspace(0.3, 0.7, 4)
        X1, X2, Y, dYdX1, dYdX2 = self.surf.eval_grid(x1_vals, x2_vals, compute_gradients=True)
        
        eps = 1e-6
        for i in range(len(x1_vals)):
            for j in range(len(x2_vals)):
                x1, x2 = X1[i, j], X2[i, j]
                
                y_x1plus = self.surf.eval_point(x1 + eps, x2)
                y_x1minus = self.surf.eval_point(x1 - eps, x2)
                dydx1_calc = (y_x1plus - y_x1minus) / (2 * eps)

                y_x2plus = self.surf.eval_point(x1, x2 + eps)
                y_x2minus = self.surf.eval_point(x1, x2 - eps)
                dydx2_calc = (y_x2plus - y_x2minus) / (2 * eps)
                
                self.assertAlmostEqual(dYdX1[i, j], dydx1_calc, places=5)
                self.assertAlmostEqual(dYdX2[i, j], dydx2_calc, places=5)

    def test_eval_grid_gradients_same_as_eval_point_gradients(self):
        x1_vals = np.array([0.4, 0.6])
        x2_vals = np.array([0.3, 0.7])
        X1, X2, Y, dYdX1, dYdX2 = self.surf.eval_grid(x1_vals, x2_vals, compute_gradients=True)
        
        self.assertEqual(dYdX1.shape, (2, 2))
        for i, xv in enumerate(x1_vals):
            for j, yv in enumerate(x2_vals):
                _, (gx1, gx2) = self.surf.eval_point(xv, yv, compute_gradients=True)
                self.assertAlmostEqual(dYdX1[i, j], gx1, places=6)
                self.assertAlmostEqual(dYdX2[i, j], gx2, places=6)

# =============================================================================
# 6. EXTRAPOLATION TESTS
# =============================================================================

class TestExtrapolation(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_extrapolation_y_value(self):
        # Point slightly outside: x=1.1, y=0.5
        # Newton method should fail, extrapolation should return a value
        y_val = self.surf.eval_point(1.1, 0.5, extrapolate=True)
        self.assertIsNotNone(y_val)
        # Should be roughly (1.1)^2 + (0.5)^2 = 1.21 + 0.25 = 1.46
        self.assertTrue(1.4 < y_val < 1.6)

    def test_extrapolation_gradients_are_not_none(self):
        x1, x2 = 1.1, 0.5
        y_val, (dydx1, dydx2) = self.surf.eval_point(x1, x2, extrapolate=True, compute_gradients=True)
        self.assertIsNotNone(y_val)
        self.assertIsNotNone(dydx1)
        self.assertIsNotNone(dydx2)

    def test_point_extrapolate_false_returns_none_outside(self):
        # Call eval_point with extrapolate=False on a point known to be
        # outside the surface domain. Verify None is returned.
        y_val = self.surf.eval_point(1.5, 0.5, extrapolate=False)
        self.assertIsNone(y_val)

    def test_grid_extrapolate_false_returns_none_outside(self):
        # eval_grid with extrapolate=False should return NaN for outside points
        x1_vals = np.linspace(1.2, 1.5, 3)
        x2_vals = np.linspace(0.4, 0.6, 2)
        X1, X2, Y = self.surf.eval_grid(x1_vals, x2_vals, extrapolate=False)
        self.assertTrue(np.all(np.isnan(Y)))

    def test_eval_extrapolate_false_returns_none_outside(self):
        # eval() with extrapolate=False should return None (scalar) or NaN (array)
        z_scalar = self.surf.eval(1.5, 0.5, extrapolate=False)
        self.assertIsNone(z_scalar)
        
        X1, X2, Y = self.surf.eval(np.array([1.5]), np.array([0.5]), extrapolate=False)
        self.assertTrue(np.all(np.isnan(Y)))

    def test_point_extrapolate_true_returns_finite_outside(self):
        # Call eval_point with extrapolate=True on a point just outside
        # the surface domain. Verify a finite float is returned.
        y_val = self.surf.eval_point(1.1, 0.5, extrapolate=True)
        self.assertIsInstance(y_val, float)
        self.assertTrue(np.isfinite(y_val))

    def test_grid_extrapolate_true_returns_finite_outside(self):
        # eval_grid with extrapolate=True should return finite values
        x1_vals = np.linspace(1.1, 1.2, 2)
        x2_vals = np.linspace(0.4, 0.6, 2)
        X1, X2, Y = self.surf.eval_grid(x1_vals, x2_vals, extrapolate=True)
        self.assertTrue(np.all(np.isfinite(Y)))

    def test_eval_extrapolate_true_returns_finite_outside(self):
        # eval() with extrapolate=True should return finite values
        z_scalar = self.surf.eval(1.1, 0.5, extrapolate=True)
        self.assertIsInstance(z_scalar, float)
        self.assertTrue(np.isfinite(z_scalar))
        
        X1, X2, Y = self.surf.eval(np.array([1.1]), np.array([0.5]), extrapolate=True)
        self.assertTrue(np.all(np.isfinite(Y)))

    def test_point_limit_distance(self):
        # Very far point should return None if limit_distance is True
        y_val = self.surf.eval_point(10.0, 10.0, extrapolate=True, limit_distance=True, distance_threshold=0.1)
        self.assertIsNone(y_val)

    def test_grid_limit_distance(self):
        # eval_grid with limit_distance=True should return NaN for very far points
        x1_vals = np.array([10.0])
        x2_vals = np.array([10.0])
        X1, X2, Y = self.surf.eval_grid(x1_vals, x2_vals, extrapolate=True, limit_distance=True, distance_threshold=0.1)
        self.assertTrue(np.all(np.isnan(Y)))

    def test_eval_limit_distance(self):
        # eval() with limit_distance=True should return None/NaN for very far points
        z_scalar = self.surf.eval(10.0, 10.0, extrapolate=True, limit_distance=True, distance_threshold=0.1)
        self.assertIsNone(z_scalar)
        
        X1, X2, Y = self.surf.eval(np.array([10.0]), np.array([10.0]), extrapolate=True, limit_distance=True, distance_threshold=0.1)
        self.assertTrue(np.all(np.isnan(Y)))

    def test_steepness_check_returns_none_near_asymptote(self):
        # Construct a surface that rises steeply at one edge (simulating
        # near-asymptotic behaviour).
        tu, tv, cp, ku, kv = get_simple_surface_data()
        # Make y very large at u=1 edge
        cp[-1, :, 1] *= 1000.0  
        surf_steep = ParametricBivariateSpline(tu, tv, cp, ku, kv)
        
        # Evaluate outside the steep edge
        x1_b, x2_b, _ = surf_steep(1.0, 0.5)
        y_val = surf_steep.eval_point(x1_b + 0.01, x2_b, extrapolate=True, limit_steepness=True)
        self.assertIsNone(y_val)

    def test_g0_continuity_at_boundary(self):
        # Sample a sequence of (x, y) points crossing from inside to outside
        # the domain along a straight line.
        x1_b, x2_b, _ = self.surf(1.0, 0.5)
        steps = np.linspace(-0.01, 0.01, 20)
        y_vals = []
        for s in steps:
            y_val = self.surf.eval_point(x1_b + s, x2_b, extrapolate=True)
            y_vals.append(y_val)
        
        # Check for large jumps
        y_diffs = np.abs(np.diff(y_vals))
        self.assertTrue(np.all(y_diffs < 0.1))

    def test_g1_continuity_at_boundary(self):
        # Sample a point on the boundary edge. Evaluate __call__ to get
        # z_boundary and calculate first derivatives. Call eval_point with
        # extrapolate=True at a point epsilon outside the boundary. Calculate first derivatives.
        # Verify that calculated first derivatives match within O(epsilon).
        u_bound, v_bound = 1.0, 0.5
        x1_b, x2_b, y_b = self.surf(u_bound, v_bound)

        eps = 1e-6
        
        # Interior gradient via eval_point (at boundary)
        _, (gx1_in, gx2_in) = self.surf.eval_point(x1_b - eps, x2_b, compute_gradients=True)
        
        # Extrapolated gradient slightly outside
        _, (gx1_out, gx2_out) = self.surf.eval_point(x1_b + eps, x2_b, extrapolate=True, compute_gradients=True)
        
        self.assertAlmostEqual(gx1_in, gx1_out, places=3)
        self.assertAlmostEqual(gx2_in, gx2_out, places=3)

    def test_g2_continuity_at_boundary(self):
        # Extend the G1 test to verify that the second derivative of the
        # extrapolated z matches the surface second derivative at the
        # boundary to within O(epsilon).
        u_bound, v_bound = 1.0, 0.5
        x1_b, x2_b, y_b = self.surf(u_bound, v_bound)
        
        # Since extrapolation is second-order Taylor, the second derivative 
        # is constant in the extrapolation region and matches the boundary second order.
        # We can test this by checking if the gradient changes linearly.
        eps = 1e-4
        _, (g1_x1, _) = self.surf.eval_point(x1_b + eps, x2_b, extrapolate=True, compute_gradients=True)
        _, (g2_x1, _) = self.surf.eval_point(x1_b + 2*eps, x2_b, extrapolate=True, compute_gradients=True)
        
        # Finite difference of gradients (2nd derivative)
        d2ydx12_ext = (g2_x1 - g1_x1) / eps
        
        # Interior 2nd derivative at boundary
        _, (g0_x1, _) = self.surf.eval_point(x1_b - eps, x2_b, compute_gradients=True)
        # Central difference across the boundary using extrapolated gradient
        d2ydx12_int = (g1_x1 - g0_x1) / (2 * eps)
        self.assertAlmostEqual(d2ydx12_ext, d2ydx12_int, places=2)

        # --- x2 direction (crossing v_max boundary) ---
        x1_b, x2_b, _ = self.surf(0.5, 1.0)
        
        _, (_, g1_x2) = self.surf.eval_point(x1_b, x2_b + eps, extrapolate=True, compute_gradients=True)
        _, (_, g2_x2) = self.surf.eval_point(x1_b, x2_b + 2*eps, extrapolate=True, compute_gradients=True)
        
        # Finite difference of gradients (2nd derivative)
        d2ydx22_ext = (g2_x2 - g1_x2) / eps
        
        # Interior 2nd derivative at boundary
        _, (_, g0_x2) = self.surf.eval_point(x1_b, x2_b - eps, compute_gradients=True)
        # Central difference across the boundary using extrapolated gradient
        d2ydx22_int = (g1_x2 - g0_x2) / (2 * eps)
        
        self.assertAlmostEqual(d2ydx22_ext, d2ydx22_int, places=2)

    def test_corner_extrapolation(self):
        # Call eval_point with extrapolate=True on a point outside both
        # the u and v extents simultaneously (diagonal corner case).
        x1_b, x2_b, _ = self.surf(1.0, 1.0)
        y_val = self.surf.eval_point(x1_b + 0.1, x2_b + 0.1, extrapolate=True)
        self.assertIsNotNone(y_val)
        self.assertTrue(np.isfinite(y_val))

# =============================================================================
# 7. LOG-SCALE TESTS
# =============================================================================

def get_log_y_surface_data():
    """
    Surface with log10-y control points (y is the dependent axis).
    cp_y = log10(10 + GU + GV) to ensure all y > 0 in physical space.
    CP layout: [x1, y, x2].
    """
    tu = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    tv = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    ku, kv = 3, 3

    gu = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    gv = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    GU, GV = np.meshgrid(gu, gv, indexing='ij')

    cp = np.zeros((5, 5, 3))
    cp[:, :, 0] = GU                        # x1 = u (linear)
    cp[:, :, 1] = np.log10(10 + GU + GV)    # y in log10 space (dependent)
    cp[:, :, 2] = GV                        # x2 = v (linear)

    return tu, tv, cp, ku, kv


def get_log_x1x2y_surface_data():
    """
    Surface where all three axes are in log10 space.
    x1_phys in [10, 1000], x2_phys in [1, 100], x2_phys > 0.
    CP layout: [x1, y, x2].
    """
    tu = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    tv = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    ku, kv = 3, 3

    gu = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    gv = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    GU, GV = np.meshgrid(gu, gv, indexing='ij')

    # Map u -> log10(x1_phys): x1_phys in [10, 1000] => log10 in [1, 3]
    log_x1 = 1.0 + 2.0 * GU
    # Map v -> log10(x2_phys): x2_phys in [1, 100] => log10 in [0, 2]
    log_x2 = 2.0 * GV
    # y in log10-space (dependent)
    log_y = np.log10(10 + 10**log_x1 * 0.01 + 10**log_x2 * 0.1)

    cp = np.zeros((5, 5, 3))
    cp[:, :, 0] = log_x1  # x1
    cp[:, :, 1] = log_y   # y (dependent)
    cp[:, :, 2] = log_x2  # x2

    return tu, tv, cp, ku, kv


class TestLogYScale(unittest.TestCase):

    def setUp(self):
        tu, tv, cp, ku, kv = get_log_y_surface_data()
        self.surf = ParametricBivariateSpline(tu, tv, cp, ku, kv, log_y=True)
        # Reference surface without log for comparison of control points
        self.tu, self.tv, self.cp, self.ku, self.kv = tu, tv, cp, ku, kv

    def test_direct_call_returns_physical_scale(self):
        """__call__ with log_y should return y in physical scale (10^w), not log."""
        u_ref, v_ref = 0.5, 0.5
        x1, x2, y_val = self.surf(u_ref, v_ref)
        # The CPs encode y_val = log10(10 + u + v).  At the Greville abscissae
        # the spline interpolates approximately, so y_phys ≈ 10 + 0.5 + 0.5 = 11.
        # The raw spline value is ~log10(11) ≈ 1.04; physical must be >> 1.
        self.assertGreater(y_val, 5.0, "y should be in physical scale, not log")
        # More precisely, the physical value should be near 10^log10(11) = 11
        self.assertAlmostEqual(y_val, 11.0, delta=1.0)

    def test_eval_point_log_y_roundtrip(self):
        """eval_point with log_y should return 10^(y) for interior points."""
        u_ref, v_ref = 0.4, 0.6
        x1_ref, x2_ref, y_ref = self.surf(u_ref, v_ref)
        # y_ref is already in physical space (10^w) (confirmed by the previous test)
        y_val = self.surf.eval_point(x1_ref, x2_ref)
        self.assertIsNotNone(y_val)
        self.assertAlmostEqual(y_val, y_ref, places=6)

    def test_eval_grid_log_y_roundtrip(self):
        """eval_grid with log_y should match eval_point for interior points."""
        x1_vals = np.linspace(0.2, 0.8, 4)
        x2_vals = np.linspace(0.2, 0.8, 3)
        X1, X2, Y = self.surf.eval_grid(x1_vals, x2_vals)

        for i, xv in enumerate(x1_vals):
            for j, yv in enumerate(x2_vals):
                y_point = self.surf.eval_point(xv, yv)
                self.assertAlmostEqual(Y[i, j], y_point, places=6,
                    msg=f"Mismatch at ({xv}, {yv})")

    def test_eval_point_extrapolation_log_y(self):
        """Extrapolation with log_y should return finite positive values."""
        y_val = self.surf.eval_point(1.1, 0.5, extrapolate=True)
        self.assertIsNotNone(y_val)
        self.assertTrue(np.isfinite(y_val))
        self.assertGreater(y_val, 0)

    def test_eval_grid_extrapolation_log_y(self):
        """eval_grid extrapolation with log_y should match eval_point."""
        x1_vals = np.linspace(0.8, 1.2, 5)
        x2_vals = np.linspace(0.4, 0.6, 3)
        X1, X2, Y = self.surf.eval_grid(x1_vals, x2_vals, extrapolate=True)
        self.assertTrue(np.all(np.isfinite(Y)))

        # Verify every grid value matches eval_point
        for i, xv in enumerate(x1_vals):
            for j, yv in enumerate(x2_vals):
                y_point = self.surf.eval_point(xv, yv, extrapolate=True)
                self.assertAlmostEqual(Y[i, j], y_point, places=6,
                    msg=f"Mismatch at ({xv:.2f}, {yv:.2f})")

    def test_eval_point_gradient_log_y_vs_finite_diff(self):
        """Gradient with log_y should match finite differences on physical y."""
        x1, x2 = 0.5, 0.5
        y_val, (dydx1, dydx2) = self.surf.eval_point(x1, x2, compute_gradients=True)

        eps = 1e-6
        y_x1p = self.surf.eval_point(x1 + eps, x2)
        y_x1m = self.surf.eval_point(x1 - eps, x2)
        dydx1_fd = (y_x1p - y_x1m) / (2 * eps)

        y_x2p = self.surf.eval_point(x1, x2 + eps)
        y_x2m = self.surf.eval_point(x1, x2 - eps)
        dydx2_fd = (y_x2p - y_x2m) / (2 * eps)

        self.assertAlmostEqual(dydx1, dydx1_fd, places=4,
            msg=f"dydx1: analytic={dydx1}, fd={dydx1_fd}")
        self.assertAlmostEqual(dydx2, dydx2_fd, places=4,
            msg=f"dydx2: analytic={dydx2}, fd={dydx2_fd}")

    def test_eval_grid_gradient_log_y_vs_eval_point(self):
        """eval_grid gradients with log_y should match eval_point gradients."""
        x1_vals = np.array([0.3, 0.5, 0.7])
        x2_vals = np.array([0.3, 0.7])
        X1, X2, Y, dYdX1, dYdX2 = self.surf.eval_grid(
            x1_vals, x2_vals, compute_gradients=True)

        for i, xv in enumerate(x1_vals):
            for j, yv in enumerate(x2_vals):
                _, (gx1, gx2) = self.surf.eval_point(xv, yv, compute_gradients=True)
                self.assertAlmostEqual(dYdX1[i, j], gx1, places=5)
                self.assertAlmostEqual(dYdX2[i, j], gx2, places=5)

    def test_extrapolation_gradient_log_y_vs_finite_diff(self):
        """Extrapolated gradient with log_y should match finite differences."""
        x1, x2 = 1.1, 0.5
        y_val, (dydx1, dydx2) = self.surf.eval_point(
            x1, x2, extrapolate=True, compute_gradients=True)
        self.assertIsNotNone(y_val)
        self.assertIsNotNone(dydx1)
        self.assertIsNotNone(dydx2)

        eps = 1e-5
        y_x1p = self.surf.eval_point(x1 + eps, x2, extrapolate=True)
        y_x1m = self.surf.eval_point(x1 - eps, x2, extrapolate=True)
        dydx1_fd = (y_x1p - y_x1m) / (2 * eps)

        y_x2p = self.surf.eval_point(x1, x2 + eps, extrapolate=True)
        y_x2m = self.surf.eval_point(x1, x2 - eps, extrapolate=True)
        dydx2_fd = (y_x2p - y_x2m) / (2 * eps)

        self.assertAlmostEqual(dydx1, dydx1_fd, places=3)
        self.assertAlmostEqual(dydx2, dydx2_fd, places=3)

    def test_extrapolation_c0_continuity(self):
        """Physical y should be continuous across the boundary with log_y."""
        # --- x1 direction (crossing u_max boundary) ---
        x1_b, x2_b, _ = self.surf(1.0, 0.5)
        steps = np.linspace(-0.01, 0.01, 20)
        y_vals_x = []
        for s in steps:
            y_val = self.surf.eval_point(x1_b + s, x2_b, extrapolate=True)
            y_vals_x.append(y_val)
        y_diffs_x = np.abs(np.diff(y_vals_x))
        self.assertTrue(np.all(y_diffs_x < 0.05),
            f"C0 x-discontinuity: max jump = {y_diffs_x.max():.6f}")

        # --- x2 direction (crossing v_max boundary) ---
        x1_b, x2_b, _ = self.surf(0.5, 1.0)
        y_vals_y = []
        for s in steps:
            y_val = self.surf.eval_point(x1_b, x2_b + s, extrapolate=True)
            y_vals_y.append(y_val)
        y_diffs_y = np.abs(np.diff(y_vals_y))
        self.assertTrue(np.all(y_diffs_y < 0.05),
            f"C0 y-discontinuity: max jump = {y_diffs_y.max():.6f}")

    def test_extrapolation_c1_continuity(self):
        """Gradients should be continuous across the boundary with log_y."""
        eps = 1e-6

        # --- x1 direction (crossing u_max boundary) ---
        x1_b, x2_b, _ = self.surf(1.0, 0.5)
        _, (gx1_in, gx2_in) = self.surf.eval_point(
            x1_b - eps, x2_b, compute_gradients=True)
        _, (gx1_out, gx2_out) = self.surf.eval_point(
            x1_b + eps, x2_b, extrapolate=True, compute_gradients=True)
        self.assertAlmostEqual(gx1_in, gx1_out, places=2,
            msg=f"x-boundary dydx1: interior={gx1_in}, exterior={gx1_out}")
        self.assertAlmostEqual(gx2_in, gx2_out, places=2,
            msg=f"x-boundary dydx2: interior={gx2_in}, exterior={gx2_out}")

        # --- x2 direction (crossing v_max boundary) ---
        x1_b, x2_b, _ = self.surf(0.5, 1.0)
        _, (gx1_in, gx2_in) = self.surf.eval_point(
            x1_b, x2_b - eps, compute_gradients=True)
        _, (gx1_out, gx2_out) = self.surf.eval_point(
            x1_b, x2_b + eps, extrapolate=True, compute_gradients=True)
        self.assertAlmostEqual(gx1_in, gx1_out, places=2,
            msg=f"y-boundary dydx1: interior={gx1_in}, exterior={gx1_out}")
        self.assertAlmostEqual(gx2_in, gx2_out, places=2,
            msg=f"y-boundary dydx2: interior={gx2_in}, exterior={gx2_out}")

    def test_extrapolation_c2_continuity(self):
        """Second derivative should be continuous across the boundary with log_y."""
        eps = 1e-4

        # --- x1 direction (crossing u_max boundary) ---
        x1_b, x2_b, _ = self.surf(1.0, 0.5)
        _, (g1_x1, _) = self.surf.eval_point(
            x1_b + eps, x2_b, extrapolate=True, compute_gradients=True)
        _, (g2_x1, _) = self.surf.eval_point(
            x1_b + 2*eps, x2_b, extrapolate=True, compute_gradients=True)
        d2ydx12_ext = (g2_x1 - g1_x1) / eps
        _, (g0_x1, _) = self.surf.eval_point(
            x1_b - eps, x2_b, compute_gradients=True)
        d2ydx12_int = (g1_x1 - g0_x1) / (2 * eps)
        self.assertAlmostEqual(d2ydx12_ext, d2ydx12_int, places=1,
            msg=f"x-boundary d2ydx12: interior={d2ydx12_int}, exterior={d2ydx12_ext}")

        # --- x2 direction (crossing v_max boundary) ---
        x1_b, x2_b, _ = self.surf(0.5, 1.0)
        _, (_, g1_x2) = self.surf.eval_point(
            x1_b, x2_b + eps, extrapolate=True, compute_gradients=True)
        _, (_, g2_x2) = self.surf.eval_point(
            x1_b, x2_b + 2*eps, extrapolate=True, compute_gradients=True)
        d2ydx22_ext = (g2_x2 - g1_x2) / eps
        _, (_, g0_x2) = self.surf.eval_point(
            x1_b, x2_b - eps, compute_gradients=True)
        d2ydx22_int = (g1_x2 - g0_x2) / (2 * eps)
        self.assertAlmostEqual(d2ydx22_ext, d2ydx22_int, places=1,
            msg=f"y-boundary d2ydx22: interior={d2ydx22_int}, exterior={d2ydx22_ext}")


class TestLogX1X2YScale(unittest.TestCase):

    def setUp(self):
        tu, tv, cp, ku, kv = get_log_x1x2y_surface_data()
        self.surf = ParametricBivariateSpline(
            tu, tv, cp, ku, kv, log_x1=True, log_x2=True, log_y=True)

    def test_direct_call_returns_physical_scale(self):
        """__call__ with log_x/y/z should return all coords in physical scale."""
        u_ref, v_ref = 0.5, 0.5
        x1, x2, y_val = self.surf(u_ref, v_ref)
        # At u=0.5: log_x = 1 + 2*0.5 = 2, so x1_phys = 10^2 = 100
        self.assertAlmostEqual(x1, 100.0, delta=10.0)
        # At v=0.5: log_y = 2*0.5 = 1, so x2_phys = 10^1 = 10
        self.assertAlmostEqual(x2, 10.0, delta=2.0)
        # y_phys = 10^log10(10 + 100*0.01 + 10*0.1) = 10 + 1 + 1 = 12
        self.assertAlmostEqual(y_val, 12.0, delta=2.0)

    def test_eval_point_returns_physical_coordinates(self):
        """eval_point should accept physical x1,x2 and return physical y."""
        # Physical coordinates: x=100, y=10 (corresponding to u≈0.5, v≈0.5)
        u_ref, v_ref = 0.5, 0.5
        x1_phys, x2_phys, y_expected = self.surf(u_ref, v_ref)
        y_val = self.surf.eval_point(x1_phys, x2_phys)
        self.assertIsNotNone(y_val)
        # z must be in physical scale (>> 1), not log scale (~1)
        self.assertGreater(y_val, 5.0, "y should be in physical scale")
        self.assertAlmostEqual(y_val, y_expected, places=5)

    def test_eval_point_log_xyz_roundtrip(self):
        """eval_point with should recover __call__ values."""
        u_ref, v_ref = 0.4, 0.6
        x1_ref, x2_ref, y_ref = self.surf(u_ref, v_ref)
        y_val = self.surf.eval_point(x1_ref, x2_ref)
        self.assertIsNotNone(y_val)
        self.assertAlmostEqual(y_val, y_ref, places=5)

    def test_eval_grid_log_x1x2z_roundtrip(self):
        """eval_grid with should match eval_point."""
        # Physical x in [10, 1000], y in [1, 100]
        u_refs = [0.3, 0.5, 0.7]
        v_refs = [0.3, 0.7]
        x1_vals = np.array([float(self.surf(u, 0.5)[0]) for u in u_refs])
        x2_vals = np.array([float(self.surf(0.5, v)[1]) for v in v_refs])

        X1, X2, Y = self.surf.eval_grid(x1_vals, x2_vals)

        for i, xv in enumerate(x1_vals):
            for j, yv in enumerate(x2_vals):
                y_point = self.surf.eval_point(xv, yv)
                if y_point is not None:
                    self.assertAlmostEqual(Y[i, j], y_point, places=5,
                        msg=f"Mismatch at ({xv:.2f}, {yv:.2f})")

    def test_eval_point_gradient_log_x1x2y_vs_finite_diff(self):
        """Gradient with all log scales vs finite differences."""
        u_ref, v_ref = 0.5, 0.5
        x1_ref, x2_ref, y_ref = self.surf(u_ref, v_ref)
        y_val, (dydx1, dydx2) = self.surf.eval_point(x1_ref, x2_ref, compute_gradients=True)

        eps_x1 = x1_ref * 1e-5  # relative epsilon for log-space
        eps_x2 = x2_ref * 1e-5

        y_x1p = self.surf.eval_point(x1_ref + eps_x1, x2_ref)
        y_x1m = self.surf.eval_point(x1_ref - eps_x1, x2_ref)
        dydx1_fd = (y_x1p - y_x1m) / (2 * eps_x1)

        y_x2p = self.surf.eval_point(x1_ref, x2_ref + eps_x2)
        y_x2m = self.surf.eval_point(x1_ref, x2_ref - eps_x2)
        dydx2_fd = (y_x2p - y_x2m) / (2 * eps_x2)

        self.assertAlmostEqual(dydx1, dydx1_fd, places=3,
            msg=f"dydx1: analytic={dydx1}, fd={dydx1_fd}")
        self.assertAlmostEqual(dydx2, dydx2_fd, places=3,
            msg=f"dydx2: analytic={dydx2}, fd={dydx2_fd}")

    def test_eval_grid_returns_physical_coordinates(self):
        """eval_grid X1, X2 outputs should be in physical (not log) space."""
        x1_vals = np.array([100.0, 500.0])
        x2_vals = np.array([10.0, 50.0])
        X1, X2, Y = self.surf.eval_grid(x1_vals, x2_vals)
        np.testing.assert_allclose(X1[:, 0], x1_vals)
        np.testing.assert_allclose(X2[0, :], x2_vals)

    def test_extrapolation_log_x1x2y(self):
        """Extrapolation with all log scales should produce finite results."""
        # Get a point just outside the domain
        x1_b, x2_b, _ = self.surf(1.0, 0.5)
        y_val = self.surf.eval_point(x1_b * 1.5, x2_b, extrapolate=True)
        self.assertIsNotNone(y_val)
        self.assertTrue(np.isfinite(y_val))
        self.assertGreater(y_val, 0)

    def test_extrapolation_c0_continuity(self):
        """Physical y should be continuous across boundary with all log scales."""
        # Use multiplicative steps since we're in log-space
        factors = np.linspace(0.99, 1.01, 20)

        # --- x direction (crossing u_max boundary) ---
        x1_b, x2_b, _ = self.surf(1.0, 0.5)
        y_vals_x = []
        for f in factors:
            y_val = self.surf.eval_point(x1_b * f, x2_b, extrapolate=True)
            y_vals_x.append(y_val)
        y_diffs_x = np.abs(np.diff(y_vals_x))
        self.assertTrue(np.all(y_diffs_x < 0.05),
            f"C0 x-discontinuity: max jump = {y_diffs_x.max():.6f}")

        # --- y direction (crossing v_max boundary) ---
        x1_b, x2_b, _ = self.surf(0.5, 1.0)
        y_vals_y = []
        for f in factors:
            y_val = self.surf.eval_point(x1_b, x2_b * f, extrapolate=True)
            y_vals_y.append(y_val)
        y_diffs_y = np.abs(np.diff(y_vals_y))
        self.assertTrue(np.all(y_diffs_y < 0.05),
            f"C0 y-discontinuity: max jump = {y_diffs_y.max():.6f}")

    def test_extrapolation_c1_continuity(self):
        """Gradients should be continuous across boundary with all log scales."""
        # --- x1 direction (crossing u_max boundary) ---
        x1_b, x2_b, _ = self.surf(1.0, 0.5)
        eps_x1 = x1_b * 1e-5  # relative epsilon for log-scale x
        _, (gx1_in, gx2_in) = self.surf.eval_point(
            x1_b - eps_x1, x2_b, compute_gradients=True)
        _, (gx1_out, gx2_out) = self.surf.eval_point(
            x1_b + eps_x1, x2_b, extrapolate=True, compute_gradients=True)
        self.assertAlmostEqual(gx1_in, gx1_out, places=2,
            msg=f"x-boundary dydx1: interior={gx1_in}, exterior={gx1_out}")
        self.assertAlmostEqual(gx2_in, gx2_out, places=2,
            msg=f"x-boundary dydx2: interior={gx2_in}, exterior={gx2_out}")

        # --- x2 direction (crossing v_max boundary) ---
        x1_b, x2_b, _ = self.surf(0.5, 1.0)
        eps_x2 = x2_b * 1e-5  # relative epsilon for log-scale y
        _, (gx1_in, gx2_in) = self.surf.eval_point(
            x1_b, x2_b - eps_x2, compute_gradients=True)
        _, (gx1_out, gx2_out) = self.surf.eval_point(
            x1_b, x2_b + eps_x2, extrapolate=True, compute_gradients=True)
        self.assertAlmostEqual(gx1_in, gx1_out, places=2,
            msg=f"y-boundary dydx1: interior={gx1_in}, exterior={gx1_out}")
        self.assertAlmostEqual(gx2_in, gx2_out, places=2,
            msg=f"y-boundary dydx2: interior={gx2_in}, exterior={gx2_out}")

    def test_extrapolation_c2_continuity(self):
        """Second derivative should be continuous across boundary with all log scales."""
        # --- x1 direction (crossing u_max boundary) ---
        x1_b, x2_b, _ = self.surf(1.0, 0.5)
        eps_x1 = x1_b * 1e-4  # relative epsilon
        _, (g1_x1, _) = self.surf.eval_point(
            x1_b + eps_x1, x2_b, extrapolate=True, compute_gradients=True)
        _, (g2_x1, _) = self.surf.eval_point(
            x1_b + 2*eps_x1, x2_b, extrapolate=True, compute_gradients=True)
        d2ydx12_ext = (g2_x1 - g1_x1) / eps_x1
        _, (g0_x1, _) = self.surf.eval_point(
            x1_b - eps_x1, x2_b, compute_gradients=True)
        d2ydx12_int = (g1_x1 - g0_x1) / (2 * eps_x1)
        self.assertAlmostEqual(d2ydx12_ext, d2ydx12_int, places=1,
            msg=f"x-boundary d2ydx12: interior={d2ydx12_int}, exterior={d2ydx12_ext}")

        # --- x2 direction (crossing v_max boundary) ---
        x1_b, x2_b, _ = self.surf(0.5, 1.0)
        eps_x2 = x2_b * 1e-4  # relative epsilon
        _, (_, g1_x2) = self.surf.eval_point(
            x1_b, x2_b + eps_x2, extrapolate=True, compute_gradients=True)
        _, (_, g2_x2) = self.surf.eval_point(
            x1_b, x2_b + 2*eps_x2, extrapolate=True, compute_gradients=True)
        d2ydx22_ext = (g2_x2 - g1_x2) / eps_x2
        _, (_, g0_x2) = self.surf.eval_point(
            x1_b, x2_b - eps_x2, compute_gradients=True)
        d2ydx22_int = (g1_x2 - g0_x2) / (2 * eps_x2)
        self.assertAlmostEqual(d2ydx22_ext, d2ydx22_int, places=1,
            msg=f"y-boundary d2ydx22: interior={d2ydx22_int}, exterior={d2ydx22_ext}")


if __name__ == '__main__':
    unittest.main()
