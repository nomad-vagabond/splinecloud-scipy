import unittest, time

import numpy as np
from scipy.interpolate import splev, bisplev

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
    tv = np.array([0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0])           # nv=4
    ku, kv = 3, 2
    
    # Greville abscissae (non-degenerate mapping)
    gu = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0]) 
    gv = np.array([0.0, 0.25, 0.75, 1.0])                 
    GU, GV = np.meshgrid(gu, gv, indexing='ij')
    
    cp = np.zeros((5, 4, 3))
    cp[:, :, 0] = GU
    cp[:, :, 1] = GV
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
# 2. FORWARD EVALUATION TESTS (__call__)
# =============================================================================

class TestCall(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_call_pointwise(self):
        # Forward evaluation: u, v -> x, y, z
        u, v = 0.3, 0.7
        x, y, z = self.surf(u, v)
        # For our simple surface x=u, y=v
        self.assertAlmostEqual(x, 0.3, places=6)
        self.assertAlmostEqual(y, 0.7, places=6)
    
    def test_scalar_input_returns_scalar(self):
        # Call __call__ with scalar u and v values.
        # Verify that returned x, y, z are Python floats, not arrays.
        u, v = 0.3, 0.7
        x, y, z = self.surf(u, v)
        self.assertIsInstance(x, float)
        self.assertIsInstance(y, float)
        self.assertIsInstance(z, float)

    def test_scalar_input_matches_bisplev(self):
        # Call __call__ with scalar u and v.
        # Verify that returned x, y, z match bisplev called directly on
        # _tck_x, _tck_y, _tck_z with the same scalar inputs, to
        # floating-point precision.
        
        u, v = 0.3, 0.7
        x, y, z = self.surf(u, v)
        
        xb = float(bisplev(u, v, self.surf._tck_x))
        yb = float(bisplev(u, v, self.surf._tck_y))
        zb = float(bisplev(u, v, self.surf._tck_z))
        
        self.assertAlmostEqual(x, xb)
        self.assertAlmostEqual(y, yb)
        self.assertAlmostEqual(z, zb)

    def test_array_input_returns_2d_grid(self):
        # Call __call__ with u of length Nu and v of length Nv where Nu != Nv.
        # Verify that x, y, z each have shape (Nu, Nv).
        u = np.array([0.1, 0.2, 0.3])
        v = np.array([0.4, 0.5])
        x, y, z = self.surf(u, v)
        self.assertEqual(x.shape, (3, 2))
        self.assertEqual(y.shape, (3, 2))
        self.assertEqual(z.shape, (3, 2))

    def test_equal_length_arrays_return_2d_grid(self):
        # Call __call__ with u and v arrays of equal length N.
        # Verify that x, y, z each have shape (N, N), confirming that
        # equal-length arrays still produce a full grid, not a diagonal.
        N = 4
        u = np.linspace(0.1, 0.4, N)
        v = np.linspace(0.5, 0.8, N)
        x, y, z = self.surf(u, v)
        self.assertEqual(x.shape, (N, N))
        self.assertEqual(y.shape, (N, N))
        self.assertEqual(z.shape, (N, N))

    def test_array_output_matches_bisplev(self):
        # Call __call__ with u of length Nu and v of length Nv.
        # Verify that x, y, z match bisplev called directly on _tck_x,
        # _tck_y, _tck_z with the same arrays, to floating-point precision.
        
        u = np.linspace(0.1, 0.9, 5)
        v = np.linspace(0.1, 0.9, 7)
        x, y, z = self.surf(u, v)
        
        xb = bisplev(u, v, self.surf._tck_x)
        yb = bisplev(u, v, self.surf._tck_y)
        zb = bisplev(u, v, self.surf._tck_z)
        
        np.testing.assert_allclose(x, xb)
        np.testing.assert_allclose(y, yb)
        np.testing.assert_allclose(z, zb)

    def test_boundary_parameter_values_no_nan(self):
        # Call __call__ as scalar at the four boundary parameter corners:
        # (tu[ku], tv[kv]), (tu[ku], tv[-(kv+1)]),
        # (tu[-(ku+1)], tv[kv]), (tu[-(ku+1)], tv[-(kv+1)]).
        # Verify that no NaN or Inf values are returned.
        u_min, u_max = self.surf.tu[self.surf.ku], self.surf.tu[-(self.surf.ku + 1)]
        v_min, v_max = self.surf.tv[self.surf.kv], self.surf.tv[-(self.surf.kv + 1)]
        
        for u in [u_min, u_max]:
            for v in [v_min, v_max]:
                x, y, z = self.surf(u, v)
                self.assertTrue(np.isfinite(x))
                self.assertTrue(np.isfinite(y))
                self.assertTrue(np.isfinite(z))

    def test_asymmetric_input_lengths(self):
        # Call __call__ with u of length 7 and v of length 13 (deliberately
        # asymmetric and not related to knot counts). Verify output shape
        # is (7, 13) and values match bisplev directly.
        
        u = np.linspace(0.1, 0.9, 7)
        v = np.linspace(0.1, 0.9, 13)
        x, y, z = self.surf(u, v)
        self.assertEqual(x.shape, (7, 13))
        
        xb = bisplev(u, v, self.surf._tck_x)
        np.testing.assert_allclose(x, xb)


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
        x_ref, y_ref, z_ref = self.surf(u_ref, v_ref)
        z = self.surf.eval_point(x_ref, y_ref)
        self.assertAlmostEqual(z, z_ref, places=6)

    def test_eval_point_exterior(self):
        # Point outside the bounding box
        x_target, y_target = 1.5, 0.5
        z = self.surf.eval_point(x_target, y_target, extrapolate=False)
        self.assertIsNone(z)

    def test_eval_scalar_interface(self):
        # Test the unified eval() interface with scalars
        u_ref, v_ref = 0.2, 0.8
        x_ref, y_ref, z_ref = self.surf(u_ref, v_ref)
        z = self.surf.eval(x_ref, y_ref)
        self.assertAlmostEqual(z, z_ref, places=6)

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

    def test_consistent_with_eval_grid_scalar(self):
        # For a single (x, y) point, verify that eval_point and eval_grid
        # called with 1-element arrays return the same z value.
        x, y = 0.5, 0.5
        z_point = self.surf.eval_point(x, y)
        X, Y, Z_grid = self.surf.eval_grid([x], [y])
        self.assertAlmostEqual(z_point, Z_grid[0, 0])

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
            x, y, z_ref = self.surf(u, v)
            z = self.surf.eval_point(x, y)
            self.assertIsNotNone(z)
            self.assertAlmostEqual(z, z_ref, places=6)

    def test_asymmetric_degrees_inversion(self):
        # Using the asymmetric degree fixture, verify that eval_point
        # correctly inverts __call__ for a set of interior (x, y) points.
        tu, tv, cp, ku, kv = get_asymmetric_surface_data()
        surf = ParametricBivariateSpline(tu, tv, cp, ku, kv)
        
        # Sample an interior point
        u_ref, v_ref = 0.5, 0.5
        x, y, z_ref = surf(u_ref, v_ref)
        z = surf.eval_point(x, y)
        self.assertIsNotNone(z)
        self.assertAlmostEqual(z, z_ref, places=6)

# =============================================================================
# 4. GRID EVALUATION TESTS (eval_grid)
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

    def test_output_shape(self):
        # Call eval_grid with x_vals of length Nx and y_vals of length Ny.
        # Verify that X, Y, Z each have shape (Nx, Ny).
        Nx, Ny = 5, 7
        x_vals = np.linspace(0.1, 0.9, Nx)
        y_vals = np.linspace(0.1, 0.9, Ny)
        X, Y, Z = self.surf.eval_grid(x_vals, y_vals)
        self.assertEqual(X.shape, (Nx, Ny))
        self.assertEqual(Y.shape, (Nx, Ny))
        self.assertEqual(Z.shape, (Nx, Ny))

    def test_matches_eval_point_on_same_grid(self):
        # Call eval_grid on a grid of (x_vals, y_vals). Then call eval_point
        # individually on each (X[i,j], Y[i,j]) pair. Verify that Z values
        # match to floating-point precision across all grid points.
        x_vals = np.linspace(0.3, 0.7, 3)
        y_vals = np.linspace(0.3, 0.7, 3)
        X, Y, Z = self.surf.eval_grid(x_vals, y_vals)
        
        for i in range(len(x_vals)):
            for j in range(len(y_vals)):
                z_point = self.surf.eval_point(X[i, j], Y[i, j])
                self.assertAlmostEqual(Z[i, j], z_point, places=6)

    def test_no_nan_for_interior_points(self):
        # Call eval_grid on a grid of (x, y) points known to be inside the
        # surface domain. Verify that no NaN or Inf values appear in Z.
        x_vals = np.linspace(0.1, 0.9, 20)
        y_vals = np.linspace(0.1, 0.9, 20)
        X, Y, Z = self.surf.eval_grid(x_vals, y_vals)
        self.assertTrue(np.all(np.isfinite(Z)))

    def test_large_grid_performance(self):
        # Call eval_grid on a 200x200 grid and verify it completes within
        # a reasonable time bound (e.g. 30 seconds).This is a regression guard 
        # against performance degradation.
        
        N = 200
        x_vals = np.linspace(0.1, 0.9, N)
        y_vals = np.linspace(0.1, 0.9, N)
        
        start = time.time()
        X, Y, Z = self.surf.eval_grid(x_vals, y_vals)
        duration = time.time() - start
        
        self.assertLess(duration, 30.0)
        self.assertEqual(Z.shape, (N, N))

# =============================================================================
# 5. GRADIENT TESTS
# =============================================================================

class TestGradients(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_eval_point_gradient_shape(self):
        # Call eval_point with compute_gradients=True. Verify the return
        # value is (z, (dzdx, dzdy)) where z, dzdx, dzdy are all floats.
        x, y = 0.5, 0.5
        z, (dzdx, dzdy) = self.surf.eval_point(x, y, compute_gradients=True)
        self.assertIsInstance(z, float)
        self.assertIsInstance(dzdx, float)
        self.assertIsInstance(dzdy, float)

    def test_eval_grid_gradient_shape(self):
        # Call eval_grid with compute_gradients=True. Verify the return
        # value is (X, Y, Z, dZdX, dZdY) where each is shape (Nx, Ny).
        Nx, Ny = 3, 4
        x_vals = np.linspace(0.4, 0.6, Nx)
        y_vals = np.linspace(0.4, 0.6, Ny)
        X, Y, Z, dZdX, dZdY = self.surf.eval_grid(x_vals, y_vals, compute_gradients=True)
        self.assertEqual(X.shape, (Nx, Ny))
        self.assertEqual(Y.shape, (Nx, Ny))
        self.assertEqual(Z.shape, (Nx, Ny))
        self.assertEqual(dZdX.shape, (Nx, Ny))
        self.assertEqual(dZdY.shape, (Nx, Ny))

    def test_gradient_returns_none_on_failed_inversion(self):
        # Call eval_point with compute_gradients=True on a point outside
        # the domain with extrapolate=False. Verify return is (None, (None, None)).
        z, (dzdx, dzdy) = self.surf.eval_point(2.0, 2.0, compute_gradients=True, extrapolate=False)
        self.assertIsNone(z)
        self.assertIsNone(dzdx)
        self.assertIsNone(dzdy)

    def test_eval_grid_gradients_are_none_on_failed_inversion(self):
        # Call eval_grid with compute_gradients=True on a grid of points outside
        # the domain with extrapolate=False. Verify return is (X, Y, Z, dZdX, dZdY)
        # where dZdX and dZdY are all None.
        Nx, Ny = 3, 4
        x_vals = np.linspace(2.0, 3.0, Nx)
        y_vals = np.linspace(2.0, 3.0, Ny)
        X, Y, Z, dZdX, dZdY = self.surf.eval_grid(
            x_vals, y_vals, compute_gradients=True, extrapolate=False)

        self.assertTrue(np.all(np.isnan(dZdX)))
        self.assertTrue(np.all(np.isnan(dZdY)))

    def test_eval_point_gradients(self):
        x, y = 0.5, 0.5
        z, (dzdx, dzdy) = self.surf.eval_point(x, y, compute_gradients=True)
        
        eps = 1e-6
        z_xplus = self.surf.eval_point(x + eps, y)
        z_xminus = self.surf.eval_point(x - eps, y)
        dzdx_calc = (z_xplus - z_xminus) / (2 * eps)

        z_yplus = self.surf.eval_point(x, y + eps)
        z_yminus = self.surf.eval_point(x, y - eps)
        dzdy_calc = (z_yplus - z_yminus) / (2 * eps)
        
        self.assertAlmostEqual(dzdx, dzdx_calc, places=5)
        self.assertAlmostEqual(dzdy, dzdy_calc, places=5)

    def test_eval_point_gradients_on_grid(self):
        # Perform similar check but for several points inside domain
        x_vals = [0.3, 0.5, 0.7]
        y_vals = [0.2, 0.4, 0.8]
        eps = 1e-6
        
        for x in x_vals:
            for y in y_vals:
                z, (dzdx, dzdy) = self.surf.eval_point(x, y, compute_gradients=True)
                
                z_xplus = self.surf.eval_point(x + eps, y)
                z_xminus = self.surf.eval_point(x - eps, y)
                dzdx_calc = (z_xplus - z_xminus) / (2 * eps)

                z_yplus = self.surf.eval_point(x, y + eps)
                z_yminus = self.surf.eval_point(x, y - eps)
                dzdy_calc = (z_yplus - z_yminus) / (2 * eps)
                
                self.assertAlmostEqual(dzdx, dzdx_calc, places=5)
                self.assertAlmostEqual(dzdy, dzdy_calc, places=5)

    def test_asymmetric_degrees_gradient(self):
        # Using the asymmetric degree fixture, verify that gradients match
        # finite differences for ku != kv surfaces.
        tu, tv, cp, ku, kv = get_asymmetric_surface_data()
        surf = ParametricBivariateSpline(tu, tv, cp, ku, kv)
        
        x, y = 0.5, 0.5
        z, (dzdx, dzdy) = surf.eval_point(x, y, compute_gradients=True)
        
        eps = 1e-6
        z_xplus = surf.eval_point(x + eps, y)
        z_xminus = surf.eval_point(x - eps, y)
        dzdx_calc = (z_xplus - z_xminus) / (2 * eps)
        
        z_yplus = surf.eval_point(x, y + eps)
        z_yminus = surf.eval_point(x, y - eps)
        dzdy_calc = (z_yplus - z_yminus) / (2 * eps)

        self.assertAlmostEqual(dzdx, dzdx_calc, places=5)
        self.assertAlmostEqual(dzdy, dzdy_calc, places=5)

    def test_eval_grid_gradients(self):
        ## Perform check with calculated gradients using epsilon for each point on grid
        x_vals = np.linspace(0.3, 0.7, 3)
        y_vals = np.linspace(0.3, 0.7, 4)
        X, Y, Z, dZdX, dZdY = self.surf.eval_grid(x_vals, y_vals, compute_gradients=True)
        
        eps = 1e-6
        for i in range(len(x_vals)):
            for j in range(len(y_vals)):
                x, y = X[i, j], Y[i, j]
                
                z_xplus = self.surf.eval_point(x + eps, y)
                z_xminus = self.surf.eval_point(x - eps, y)
                dzdx_calc = (z_xplus - z_xminus) / (2 * eps)

                z_yplus = self.surf.eval_point(x, y + eps)
                z_yminus = self.surf.eval_point(x, y - eps)
                dzdy_calc = (z_yplus - z_yminus) / (2 * eps)
                
                self.assertAlmostEqual(dZdX[i, j], dzdx_calc, places=5)
                self.assertAlmostEqual(dZdY[i, j], dzdy_calc, places=5)

    def test_eval_grid_gradients_same_as_eval_point_gradients(self):
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
# 6. EXTRAPOLATION TESTS
# =============================================================================

class TestExtrapolation(unittest.TestCase):

    def setUp(self):
        self.tu, self.tv, self.cp, self.ku, self.kv = get_simple_surface_data()
        self.surf = ParametricBivariateSpline(self.tu, self.tv, self.cp, self.ku, self.kv)

    def test_extrapolation_z_value(self):
        # Point slightly outside: x=1.1, y=0.5
        # Newton method should fail, extrapolation should return a value
        z = self.surf.eval_point(1.1, 0.5, extrapolate=True)
        self.assertIsNotNone(z)
        # Should be roughly (1.1)^2 + (0.5)^2 = 1.21 + 0.25 = 1.46
        self.assertTrue(1.4 < z < 1.6)

    def test_extrapolation_gradients_are_not_none(self):
        x, y = 1.1, 0.5
        z, (dzdx, dzdy) = self.surf.eval_point(x, y, extrapolate=True, compute_gradients=True)
        self.assertIsNotNone(z)
        self.assertIsNotNone(dzdx)
        self.assertIsNotNone(dzdy)

    def test_point_extrapolate_false_returns_none_outside(self):
        # Call eval_point with extrapolate=False on a point known to be
        # outside the surface domain. Verify None is returned.
        z = self.surf.eval_point(1.5, 0.5, extrapolate=False)
        self.assertIsNone(z)

    def test_grid_extrapolate_false_returns_none_outside(self):
        # eval_grid with extrapolate=False should return NaN for outside points
        x_vals = np.linspace(1.2, 1.5, 3)
        y_vals = np.linspace(0.4, 0.6, 2)
        X, Y, Z = self.surf.eval_grid(x_vals, y_vals, extrapolate=False)
        self.assertTrue(np.all(np.isnan(Z)))

    def test_eval_extrapolate_false_returns_none_outside(self):
        # eval() with extrapolate=False should return None (scalar) or NaN (array)
        z_scalar = self.surf.eval(1.5, 0.5, extrapolate=False)
        self.assertIsNone(z_scalar)
        
        X, Y, Z = self.surf.eval(np.array([1.5]), np.array([0.5]), extrapolate=False)
        self.assertTrue(np.all(np.isnan(Z)))

    def test_point_extrapolate_true_returns_finite_outside(self):
        # Call eval_point with extrapolate=True on a point just outside
        # the surface domain. Verify a finite float is returned.
        z = self.surf.eval_point(1.1, 0.5, extrapolate=True)
        self.assertIsInstance(z, float)
        self.assertTrue(np.isfinite(z))

    def test_grid_extrapolate_true_returns_finite_outside(self):
        # eval_grid with extrapolate=True should return finite values
        x_vals = np.linspace(1.1, 1.2, 2)
        y_vals = np.linspace(0.4, 0.6, 2)
        X, Y, Z = self.surf.eval_grid(x_vals, y_vals, extrapolate=True)
        self.assertTrue(np.all(np.isfinite(Z)))

    def test_eval_extrapolate_true_returns_finite_outside(self):
        # eval() with extrapolate=True should return finite values
        z_scalar = self.surf.eval(1.1, 0.5, extrapolate=True)
        self.assertIsInstance(z_scalar, float)
        self.assertTrue(np.isfinite(z_scalar))
        
        X, Y, Z = self.surf.eval(np.array([1.1]), np.array([0.5]), extrapolate=True)
        self.assertTrue(np.all(np.isfinite(Z)))

    def test_point_limit_distance(self):
        # Very far point should return None if limit_distance is True
        z = self.surf.eval_point(10.0, 10.0, extrapolate=True, limit_distance=True, distance_threshold=0.1)
        self.assertIsNone(z)

    def test_grid_limit_distance(self):
        # eval_grid with limit_distance=True should return NaN for very far points
        x_vals = np.array([10.0])
        y_vals = np.array([10.0])
        X, Y, Z = self.surf.eval_grid(x_vals, y_vals, extrapolate=True, limit_distance=True, distance_threshold=0.1)
        self.assertTrue(np.all(np.isnan(Z)))

    def test_eval_limit_distance(self):
        # eval() with limit_distance=True should return None/NaN for very far points
        z_scalar = self.surf.eval(10.0, 10.0, extrapolate=True, limit_distance=True, distance_threshold=0.1)
        self.assertIsNone(z_scalar)
        
        X, Y, Z = self.surf.eval(np.array([10.0]), np.array([10.0]), extrapolate=True, limit_distance=True, distance_threshold=0.1)
        self.assertTrue(np.all(np.isnan(Z)))

    def test_steepness_check_returns_none_near_asymptote(self):
        # Construct a surface that rises steeply at one edge (simulating
        # near-asymptotic behaviour).
        tu, tv, cp, ku, kv = get_simple_surface_data()
        # Make z very large at u=1 edge
        cp[-1, :, 2] *= 1000.0  
        surf_steep = ParametricBivariateSpline(tu, tv, cp, ku, kv)
        
        # Evaluate outside the steep edge
        x_b, y_b, _ = surf_steep(1.0, 0.5)
        z = surf_steep.eval_point(x_b + 0.01, y_b, extrapolate=True, limit_steepness=True)
        self.assertIsNone(z)

    def test_g0_continuity_at_boundary(self):
        # Sample a sequence of (x, y) points crossing from inside to outside
        # the domain along a straight line.
        x_b, y_b, _ = self.surf(1.0, 0.5)
        steps = np.linspace(-0.01, 0.01, 20)
        z_vals = []
        for s in steps:
            z = self.surf.eval_point(x_b + s, y_b, extrapolate=True)
            z_vals.append(z)
        
        # Check for large jumps
        z_diffs = np.abs(np.diff(z_vals))
        self.assertTrue(np.all(z_diffs < 0.1))

    def test_g1_continuity_at_boundary(self):
        # Sample a point on the boundary edge. Evaluate __call__ to get
        # z_boundary and calculate first derivatives. Call eval_point with
        # extrapolate=True at a point epsilon outside the boundary. Calculate first derivatives.
        # Verify that calculated first derivatives match within O(epsilon).
        u_bound, v_bound = 1.0, 0.5
        x_b, y_b, z_b = self.surf(u_bound, v_bound)

        eps = 1e-6
        
        # Interior gradient via eval_point (at boundary)
        _, (gx_in, gy_in) = self.surf.eval_point(x_b - eps, y_b, compute_gradients=True)
        
        # Extrapolated gradient slightly outside
        _, (gx_out, gy_out) = self.surf.eval_point(x_b + eps, y_b, extrapolate=True, compute_gradients=True)
        
        self.assertAlmostEqual(gx_in, gx_out, places=3)
        self.assertAlmostEqual(gy_in, gy_out, places=3)

    def test_g2_continuity_at_boundary(self):
        # Extend the G1 test to verify that the second derivative of the
        # extrapolated z matches the surface second derivative at the
        # boundary to within O(epsilon).
        u_bound, v_bound = 1.0, 0.5
        x_b, y_b, z_b = self.surf(u_bound, v_bound)
        
        # Since extrapolation is second-order Taylor, the second derivative 
        # is constant in the extrapolation region and matches the boundary second order.
        # We can test this by checking if the gradient changes linearly.
        eps = 1e-4
        _, (g1_x, _) = self.surf.eval_point(x_b + eps, y_b, extrapolate=True, compute_gradients=True)
        _, (g2_x, _) = self.surf.eval_point(x_b + 2*eps, y_b, extrapolate=True, compute_gradients=True)
        
        # Finite difference of gradients (2nd derivative)
        d2zdx2_ext = (g2_x - g1_x) / eps
        
        # Interior 2nd derivative at boundary
        _, (g0_x, _) = self.surf.eval_point(x_b - eps, y_b, compute_gradients=True)
        # Central difference across the boundary using extrapolated gradient
        d2zdx2_int = (g1_x - g0_x) / (2 * eps)
        
        self.assertAlmostEqual(d2zdx2_ext, d2zdx2_int, places=2)

    def test_eval_grid_extrapolation_works(self):
        # Call eval_grid with extrapolate=True on a grid that contains both
        # inside and outside points.
        x_vals = np.linspace(0.8, 1.2, 10)
        y_vals = np.linspace(0.4, 0.6, 5)
        X, Y, Z = self.surf.eval_grid(x_vals, y_vals, extrapolate=True)
        self.assertTrue(np.all(np.isfinite(Z)))
        # Verify a point outside is actually extrapolated
        z_out = self.surf.eval_point(1.2, 0.5, extrapolate=True)
        self.assertAlmostEqual(Z[-1, 2], z_out)

    def test_corner_extrapolation(self):
        # Call eval_point with extrapolate=True on a point outside both
        # the u and v extents simultaneously (diagonal corner case).
        x_b, y_b, _ = self.surf(1.0, 1.0)
        z = self.surf.eval_point(x_b + 0.1, y_b + 0.1, extrapolate=True)
        self.assertIsNotNone(z)
        self.assertTrue(np.isfinite(z))

# =============================================================================
# 7. LOG-SCALE TESTS
# =============================================================================

def get_log_z_surface_data():
    """
    Surface with log10-z control points.
    cp_z = log10(10 + GU + GV) to ensure all z > 0 in physical space.
    """
    tu = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    tv = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    ku, kv = 3, 3

    gu = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    gv = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    GU, GV = np.meshgrid(gu, gv, indexing='ij')

    cp = np.zeros((5, 5, 3))
    cp[:, :, 0] = GU                        # x = u (linear)
    cp[:, :, 1] = GV                        # y = v (linear)
    cp[:, :, 2] = np.log10(10 + GU + GV)    # z in log10 space

    return tu, tv, cp, ku, kv


def get_log_xyz_surface_data():
    """
    Surface where all three axes are in log10 space.
    x_phys in [10, 1000], y_phys in [1, 100], z_phys > 0.
    """
    tu = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    tv = np.array([0.0, 0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 1.0, 1.0])
    ku, kv = 3, 3

    gu = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    gv = np.array([0.0, 0.16666667, 0.5, 0.83333333, 1.0])
    GU, GV = np.meshgrid(gu, gv, indexing='ij')

    # Map u -> log10(x_phys): x_phys in [10, 1000] => log10 in [1, 3]
    log_x = 1.0 + 2.0 * GU
    # Map v -> log10(y_phys): y_phys in [1, 100] => log10 in [0, 2]
    log_y = 2.0 * GV
    # z in log10-space
    log_z = np.log10(10 + 10**log_x * 0.01 + 10**log_y * 0.1)

    cp = np.zeros((5, 5, 3))
    cp[:, :, 0] = log_x
    cp[:, :, 1] = log_y
    cp[:, :, 2] = log_z

    return tu, tv, cp, ku, kv


class TestLogZScale(unittest.TestCase):

    def setUp(self):
        tu, tv, cp, ku, kv = get_log_z_surface_data()
        self.surf = ParametricBivariateSpline(tu, tv, cp, ku, kv, log_z=True)
        # Reference surface without log for comparison of control points
        self.tu, self.tv, self.cp, self.ku, self.kv = tu, tv, cp, ku, kv

    def test_direct_call_returns_physical_scale(self):
        """__call__ with log_z should return z in physical scale (10^w), not log."""
        u_ref, v_ref = 0.5, 0.5
        x, y, z = self.surf(u_ref, v_ref)
        # The CPs encode z = log10(10 + u + v).  At the Greville abscissae
        # the spline interpolates approximately, so z_phys ≈ 10 + 0.5 + 0.5 = 11.
        # The raw spline value is ~log10(11) ≈ 1.04; physical must be >> 1.
        self.assertGreater(z, 5.0, "z should be in physical scale, not log")
        # More precisely, the physical value should be near 10^log10(11) = 11
        self.assertAlmostEqual(z, 11.0, delta=1.0)

    def test_eval_point_log_z_roundtrip(self):
        """eval_point with log_z should return 10^(spline_z) for interior points."""
        u_ref, v_ref = 0.4, 0.6
        x_ref, y_ref, z_ref = self.surf(u_ref, v_ref)
        # z_ref is already in physical space (10^w) (confirmed by the previous test)
        z = self.surf.eval_point(x_ref, y_ref)
        self.assertIsNotNone(z)
        self.assertAlmostEqual(z, z_ref, places=6)

    def test_eval_grid_log_z_roundtrip(self):
        """eval_grid with log_z should match eval_point for interior points."""
        x_vals = np.linspace(0.2, 0.8, 4)
        y_vals = np.linspace(0.2, 0.8, 3)
        X, Y, Z = self.surf.eval_grid(x_vals, y_vals)

        for i, xv in enumerate(x_vals):
            for j, yv in enumerate(y_vals):
                z_point = self.surf.eval_point(xv, yv)
                self.assertAlmostEqual(Z[i, j], z_point, places=6,
                    msg=f"Mismatch at ({xv}, {yv})")

    def test_eval_point_extrapolation_log_z(self):
        """Extrapolation with log_z should return finite positive values."""
        z = self.surf.eval_point(1.1, 0.5, extrapolate=True)
        self.assertIsNotNone(z)
        self.assertTrue(np.isfinite(z))
        self.assertGreater(z, 0)

    def test_eval_grid_extrapolation_log_z(self):
        """eval_grid extrapolation with log_z should match eval_point."""
        x_vals = np.linspace(0.8, 1.2, 5)
        y_vals = np.linspace(0.4, 0.6, 3)
        X, Y, Z = self.surf.eval_grid(x_vals, y_vals, extrapolate=True)
        self.assertTrue(np.all(np.isfinite(Z)))

        # Verify every grid value matches eval_point
        for i, xv in enumerate(x_vals):
            for j, yv in enumerate(y_vals):
                z_point = self.surf.eval_point(xv, yv, extrapolate=True)
                self.assertAlmostEqual(Z[i, j], z_point, places=6,
                    msg=f"Mismatch at ({xv:.2f}, {yv:.2f})")

    def test_eval_point_gradient_log_z_vs_finite_diff(self):
        """Gradient with log_z should match finite differences on physical z."""
        x, y = 0.5, 0.5
        z, (dzdx, dzdy) = self.surf.eval_point(x, y, compute_gradients=True)

        eps = 1e-6
        z_xp = self.surf.eval_point(x + eps, y)
        z_xm = self.surf.eval_point(x - eps, y)
        dzdx_fd = (z_xp - z_xm) / (2 * eps)

        z_yp = self.surf.eval_point(x, y + eps)
        z_ym = self.surf.eval_point(x, y - eps)
        dzdy_fd = (z_yp - z_ym) / (2 * eps)

        self.assertAlmostEqual(dzdx, dzdx_fd, places=4,
            msg=f"dzdx: analytic={dzdx}, fd={dzdx_fd}")
        self.assertAlmostEqual(dzdy, dzdy_fd, places=4,
            msg=f"dzdy: analytic={dzdy}, fd={dzdy_fd}")

    def test_eval_grid_gradient_log_z_vs_eval_point(self):
        """eval_grid gradients with log_z should match eval_point gradients."""
        x_vals = np.array([0.3, 0.5, 0.7])
        y_vals = np.array([0.3, 0.7])
        X, Y, Z, dZdX, dZdY = self.surf.eval_grid(
            x_vals, y_vals, compute_gradients=True)

        for i, xv in enumerate(x_vals):
            for j, yv in enumerate(y_vals):
                _, (gx, gy) = self.surf.eval_point(xv, yv, compute_gradients=True)
                self.assertAlmostEqual(dZdX[i, j], gx, places=5)
                self.assertAlmostEqual(dZdY[i, j], gy, places=5)

    def test_extrapolation_gradient_log_z_vs_finite_diff(self):
        """Extrapolated gradient with log_z should match finite differences."""
        x, y = 1.1, 0.5
        z, (dzdx, dzdy) = self.surf.eval_point(
            x, y, extrapolate=True, compute_gradients=True)
        self.assertIsNotNone(z)
        self.assertIsNotNone(dzdx)
        self.assertIsNotNone(dzdy)

        eps = 1e-5
        z_xp = self.surf.eval_point(x + eps, y, extrapolate=True)
        z_xm = self.surf.eval_point(x - eps, y, extrapolate=True)
        dzdx_fd = (z_xp - z_xm) / (2 * eps)

        z_yp = self.surf.eval_point(x, y + eps, extrapolate=True)
        z_ym = self.surf.eval_point(x, y - eps, extrapolate=True)
        dzdy_fd = (z_yp - z_ym) / (2 * eps)

        self.assertAlmostEqual(dzdx, dzdx_fd, places=3)
        self.assertAlmostEqual(dzdy, dzdy_fd, places=3)

    def test_extrapolation_c0_continuity(self):
        """Physical z should be continuous across the boundary with log_z."""
        # --- x direction (crossing u_max boundary) ---
        x_b, y_b, _ = self.surf(1.0, 0.5)
        steps = np.linspace(-0.01, 0.01, 20)
        z_vals = []
        for s in steps:
            z = self.surf.eval_point(x_b + s, y_b, extrapolate=True)
            z_vals.append(z)
        z_diffs_x = np.abs(np.diff(z_vals))
        self.assertTrue(np.all(z_diffs_x < 0.5),
            f"C0 x-discontinuity: max jump = {z_diffs_x.max():.6f}")

        # --- y direction (crossing v_max boundary) ---
        x_b, y_b, _ = self.surf(0.5, 1.0)
        z_vals_y = []
        for s in steps:
            z = self.surf.eval_point(x_b, y_b + s, extrapolate=True)
            z_vals_y.append(z)
        z_diffs_y = np.abs(np.diff(z_vals_y))
        self.assertTrue(np.all(z_diffs_y < 0.5),
            f"C0 y-discontinuity: max jump = {z_diffs_y.max():.6f}")

    def test_extrapolation_c1_continuity(self):
        """Gradients should be continuous across the boundary with log_z."""
        eps = 1e-6

        # --- x direction (crossing u_max boundary) ---
        x_b, y_b, _ = self.surf(1.0, 0.5)
        _, (gx_in, gy_in) = self.surf.eval_point(
            x_b - eps, y_b, compute_gradients=True)
        _, (gx_out, gy_out) = self.surf.eval_point(
            x_b + eps, y_b, extrapolate=True, compute_gradients=True)
        self.assertAlmostEqual(gx_in, gx_out, places=2,
            msg=f"x-boundary dzdx: interior={gx_in}, exterior={gx_out}")
        self.assertAlmostEqual(gy_in, gy_out, places=2,
            msg=f"x-boundary dzdy: interior={gy_in}, exterior={gy_out}")

        # --- y direction (crossing v_max boundary) ---
        x_b, y_b, _ = self.surf(0.5, 1.0)
        _, (gx_in, gy_in) = self.surf.eval_point(
            x_b, y_b - eps, compute_gradients=True)
        _, (gx_out, gy_out) = self.surf.eval_point(
            x_b, y_b + eps, extrapolate=True, compute_gradients=True)
        self.assertAlmostEqual(gx_in, gx_out, places=2,
            msg=f"y-boundary dzdx: interior={gx_in}, exterior={gx_out}")
        self.assertAlmostEqual(gy_in, gy_out, places=2,
            msg=f"y-boundary dzdy: interior={gy_in}, exterior={gy_out}")

    def test_extrapolation_c2_continuity(self):
        """Second derivative should be continuous across the boundary with log_z."""
        eps = 1e-4

        # --- x direction (crossing u_max boundary) ---
        x_b, y_b, _ = self.surf(1.0, 0.5)
        _, (g1_x, _) = self.surf.eval_point(
            x_b + eps, y_b, extrapolate=True, compute_gradients=True)
        _, (g2_x, _) = self.surf.eval_point(
            x_b + 2*eps, y_b, extrapolate=True, compute_gradients=True)
        d2zdx2_ext = (g2_x - g1_x) / eps
        _, (g0_x, _) = self.surf.eval_point(
            x_b - eps, y_b, compute_gradients=True)
        d2zdx2_int = (g1_x - g0_x) / (2 * eps)
        self.assertAlmostEqual(d2zdx2_ext, d2zdx2_int, places=1,
            msg=f"x-boundary d2zdx2: interior={d2zdx2_int}, exterior={d2zdx2_ext}")

        # --- y direction (crossing v_max boundary) ---
        x_b, y_b, _ = self.surf(0.5, 1.0)
        _, (_, g1_y) = self.surf.eval_point(
            x_b, y_b + eps, extrapolate=True, compute_gradients=True)
        _, (_, g2_y) = self.surf.eval_point(
            x_b, y_b + 2*eps, extrapolate=True, compute_gradients=True)
        d2zdy2_ext = (g2_y - g1_y) / eps
        _, (_, g0_y) = self.surf.eval_point(
            x_b, y_b - eps, compute_gradients=True)
        d2zdy2_int = (g1_y - g0_y) / (2 * eps)
        self.assertAlmostEqual(d2zdy2_ext, d2zdy2_int, places=1,
            msg=f"y-boundary d2zdy2: interior={d2zdy2_int}, exterior={d2zdy2_ext}")


class TestLogXYZScale(unittest.TestCase):

    def setUp(self):
        tu, tv, cp, ku, kv = get_log_xyz_surface_data()
        self.surf = ParametricBivariateSpline(
            tu, tv, cp, ku, kv, log_x=True, log_y=True, log_z=True)

    def test_direct_call_returns_physical_scale(self):
        """__call__ with log_x/y/z should return all coords in physical scale."""
        u_ref, v_ref = 0.5, 0.5
        x, y, z = self.surf(u_ref, v_ref)
        # At u=0.5: log_x = 1 + 2*0.5 = 2, so x_phys = 10^2 = 100
        self.assertAlmostEqual(x, 100.0, delta=10.0)
        # At v=0.5: log_y = 2*0.5 = 1, so y_phys = 10^1 = 10
        self.assertAlmostEqual(y, 10.0, delta=2.0)
        # z_phys = 10^log10(10 + 100*0.01 + 10*0.1) = 10 + 1 + 1 = 12
        self.assertAlmostEqual(z, 12.0, delta=2.0)

    def test_eval_point_returns_physical_coordinates(self):
        """eval_point should accept physical x,y and return physical z."""
        # Physical coordinates: x=100, y=10 (corresponding to u≈0.5, v≈0.5)
        u_ref, v_ref = 0.5, 0.5
        x_phys, y_phys, z_expected = self.surf(u_ref, v_ref)
        z = self.surf.eval_point(x_phys, y_phys)
        self.assertIsNotNone(z)
        # z must be in physical scale (>> 1), not log scale (~1)
        self.assertGreater(z, 5.0, "z should be in physical scale")
        self.assertAlmostEqual(z, z_expected, places=5)

    def test_eval_point_log_xyz_roundtrip(self):
        """eval_point with log_x/log_y/log_z should recover __call__ values."""
        u_ref, v_ref = 0.4, 0.6
        x_ref, y_ref, z_ref = self.surf(u_ref, v_ref)
        z = self.surf.eval_point(x_ref, y_ref)
        self.assertIsNotNone(z)
        self.assertAlmostEqual(z, z_ref, places=5)

    def test_eval_grid_log_xyz_roundtrip(self):
        """eval_grid with log_x/log_y/log_z should match eval_point."""
        # Physical x in [10, 1000], y in [1, 100]
        u_refs = [0.3, 0.5, 0.7]
        v_refs = [0.3, 0.7]
        x_vals = np.array([float(self.surf(u, 0.5)[0]) for u in u_refs])
        y_vals = np.array([float(self.surf(0.5, v)[1]) for v in v_refs])

        X, Y, Z = self.surf.eval_grid(x_vals, y_vals)

        # X, Y should be in physical space
        self.assertTrue(np.all(X > 1))   # x_phys > 1
        self.assertTrue(np.all(Y > 0.5)) # y_phys > 0.5

        for i, xv in enumerate(x_vals):
            for j, yv in enumerate(y_vals):
                z_point = self.surf.eval_point(xv, yv)
                if z_point is not None:
                    self.assertAlmostEqual(Z[i, j], z_point, places=5,
                        msg=f"Mismatch at ({xv:.2f}, {yv:.2f})")

    def test_eval_point_gradient_log_xyz_vs_finite_diff(self):
        """Gradient with all log scales vs finite differences."""
        u_ref, v_ref = 0.5, 0.5
        x, y, z_ref = self.surf(u_ref, v_ref)
        z, (dzdx, dzdy) = self.surf.eval_point(x, y, compute_gradients=True)

        eps_x = x * 1e-5  # relative epsilon for log-space
        eps_y = y * 1e-5

        z_xp = self.surf.eval_point(x + eps_x, y)
        z_xm = self.surf.eval_point(x - eps_x, y)
        dzdx_fd = (z_xp - z_xm) / (2 * eps_x)

        z_yp = self.surf.eval_point(x, y + eps_y)
        z_ym = self.surf.eval_point(x, y - eps_y)
        dzdy_fd = (z_yp - z_ym) / (2 * eps_y)

        self.assertAlmostEqual(dzdx, dzdx_fd, places=3,
            msg=f"dzdx: analytic={dzdx}, fd={dzdx_fd}")
        self.assertAlmostEqual(dzdy, dzdy_fd, places=3,
            msg=f"dzdy: analytic={dzdy}, fd={dzdy_fd}")

    def test_eval_grid_returns_physical_coordinates(self):
        """eval_grid X, Y outputs should be in physical (not log) space."""
        x_vals = np.array([100.0, 500.0])
        y_vals = np.array([10.0, 50.0])
        X, Y, Z = self.surf.eval_grid(x_vals, y_vals)
        np.testing.assert_allclose(X[:, 0], x_vals)
        np.testing.assert_allclose(Y[0, :], y_vals)

    def test_extrapolation_log_xyz(self):
        """Extrapolation with all log scales should produce finite results."""
        # Get a point just outside the domain
        x_b, y_b, _ = self.surf(1.0, 0.5)
        z = self.surf.eval_point(x_b * 1.5, y_b, extrapolate=True)
        self.assertIsNotNone(z)
        self.assertTrue(np.isfinite(z))
        self.assertGreater(z, 0)

    def test_extrapolation_c0_continuity(self):
        """Physical z should be continuous across boundary with all log scales."""
        # Use multiplicative steps since we're in log-space
        factors = np.linspace(0.99, 1.01, 20)

        # --- x direction (crossing u_max boundary) ---
        x_b, y_b, _ = self.surf(1.0, 0.5)
        z_vals_x = []
        for f in factors:
            z = self.surf.eval_point(x_b * f, y_b, extrapolate=True)
            z_vals_x.append(z)
        z_diffs_x = np.abs(np.diff(z_vals_x))
        self.assertTrue(np.all(z_diffs_x < 1.0),
            f"C0 x-discontinuity: max jump = {z_diffs_x.max():.6f}")

        # --- y direction (crossing v_max boundary) ---
        x_b, y_b, _ = self.surf(0.5, 1.0)
        z_vals_y = []
        for f in factors:
            z = self.surf.eval_point(x_b, y_b * f, extrapolate=True)
            z_vals_y.append(z)
        z_diffs_y = np.abs(np.diff(z_vals_y))
        self.assertTrue(np.all(z_diffs_y < 1.0),
            f"C0 y-discontinuity: max jump = {z_diffs_y.max():.6f}")

    def test_extrapolation_c1_continuity(self):
        """Gradients should be continuous across boundary with all log scales."""
        # --- x direction (crossing u_max boundary) ---
        x_b, y_b, _ = self.surf(1.0, 0.5)
        eps_x = x_b * 1e-5  # relative epsilon for log-scale x
        _, (gx_in, gy_in) = self.surf.eval_point(
            x_b - eps_x, y_b, compute_gradients=True)
        _, (gx_out, gy_out) = self.surf.eval_point(
            x_b + eps_x, y_b, extrapolate=True, compute_gradients=True)
        self.assertAlmostEqual(gx_in, gx_out, places=2,
            msg=f"x-boundary dzdx: interior={gx_in}, exterior={gx_out}")
        self.assertAlmostEqual(gy_in, gy_out, places=2,
            msg=f"x-boundary dzdy: interior={gy_in}, exterior={gy_out}")

        # --- y direction (crossing v_max boundary) ---
        x_b, y_b, _ = self.surf(0.5, 1.0)
        eps_y = y_b * 1e-5  # relative epsilon for log-scale y
        _, (gx_in, gy_in) = self.surf.eval_point(
            x_b, y_b - eps_y, compute_gradients=True)
        _, (gx_out, gy_out) = self.surf.eval_point(
            x_b, y_b + eps_y, extrapolate=True, compute_gradients=True)
        self.assertAlmostEqual(gx_in, gx_out, places=2,
            msg=f"y-boundary dzdx: interior={gx_in}, exterior={gx_out}")
        self.assertAlmostEqual(gy_in, gy_out, places=2,
            msg=f"y-boundary dzdy: interior={gy_in}, exterior={gy_out}")

    def test_extrapolation_c2_continuity(self):
        """Second derivative should be continuous across boundary with all log scales."""
        # --- x direction (crossing u_max boundary) ---
        x_b, y_b, _ = self.surf(1.0, 0.5)
        eps_x = x_b * 1e-4  # relative epsilon
        _, (g1_x, _) = self.surf.eval_point(
            x_b + eps_x, y_b, extrapolate=True, compute_gradients=True)
        _, (g2_x, _) = self.surf.eval_point(
            x_b + 2*eps_x, y_b, extrapolate=True, compute_gradients=True)
        d2zdx2_ext = (g2_x - g1_x) / eps_x
        _, (g0_x, _) = self.surf.eval_point(
            x_b - eps_x, y_b, compute_gradients=True)
        d2zdx2_int = (g1_x - g0_x) / (2 * eps_x)
        self.assertAlmostEqual(d2zdx2_ext, d2zdx2_int, places=1,
            msg=f"x-boundary d2zdx2: interior={d2zdx2_int}, exterior={d2zdx2_ext}")

        # --- y direction (crossing v_max boundary) ---
        x_b, y_b, _ = self.surf(0.5, 1.0)
        eps_y = y_b * 1e-4  # relative epsilon
        _, (_, g1_y) = self.surf.eval_point(
            x_b, y_b + eps_y, extrapolate=True, compute_gradients=True)
        _, (_, g2_y) = self.surf.eval_point(
            x_b, y_b + 2*eps_y, extrapolate=True, compute_gradients=True)
        d2zdy2_ext = (g2_y - g1_y) / eps_y
        _, (_, g0_y) = self.surf.eval_point(
            x_b, y_b - eps_y, compute_gradients=True)
        d2zdy2_int = (g1_y - g0_y) / (2 * eps_y)
        self.assertAlmostEqual(d2zdy2_ext, d2zdy2_int, places=1,
            msg=f"y-boundary d2zdy2: interior={d2zdy2_int}, exterior={d2zdy2_ext}")


if __name__ == '__main__':
    unittest.main()

