"""
Build a B-spline volume from a set of surfaces.

Each SplineCloud surface is a ParametricBivariateSpline storing three coordinate
fields (x1, y, x2) over parameters (u, v). Surfaces are stacked along a third
physical axis (x3) to build a parametric trivariate volume storing
(x1, x2, x3, y) over (u, v, w).

Evaluation mirrors the SplineCloud class: invert (x1, x2, x3) -> (u, v, w) by
Newton, then read y.
"""

import numpy as np
from splipy import Surface, Volume, BSplineBasis
from scipy.interpolate import BSpline, PchipInterpolator, make_interp_spline
from numpy.polynomial import polynomial as nppoly

from .parametric_spline_surface import ParametricBivariateSpline
from .api_client import load_spline_surface


def _resolve_surface(surface):
    """Resolve a surface spec to a ParametricBivariateSpline instance."""
    if isinstance(surface, ParametricBivariateSpline):
        return surface
    if isinstance(surface, str):
        return load_spline_surface(surface)
    raise TypeError(
        "Each surface must be a ParametricBivariateSpline instance or a "
        f"surface UID string, got {type(surface).__name__!r}"
    )


def pbs_to_splipy(pbs):
    """
    Rebuild the (nu, nv, 3) control net from the three tck tuples of a
    ParametricBivariateSpline and wrap it as a non-rational splipy.Surface.

    The class stores coefficients as:
        _tck_x1 = (tu, tv, cp[:,:,0].ravel(), ku, kv)
        _tck_y  = (tu, tv, cp[:,:,1].ravel(), ku, kv)
        _tck_x2 = (tu, tv, cp[:,:,2].ravel(), ku, kv)
    so the physical control-point ordering is cp[i,j] = [x1, y, x2].

    Keep all three coordinate fields, but reorder to [x1, x2, y] so the
    'y' value is the LAST coordinate. This makes the trivariate inversion
    below solve for (x1, x2, x3) and read y cleanly.
    """
    tu, tv = pbs.tu, pbs.tv
    ku, kv = pbs.ku, pbs.kv

    basis_u = BSplineBasis(order=ku + 1, knots=tu)
    basis_v = BSplineBasis(order=kv + 1, knots=tv)
    nu = basis_u.num_functions()
    nv = basis_v.num_functions()

    c_x1 = pbs._tck_x1[2].reshape(nu, nv)
    c_y  = pbs._tck_y[2].reshape(nu, nv)
    c_x2 = pbs._tck_x2[2].reshape(nu, nv)

    # store as [x1, x2, y]  (note the swap of y and x2 vs. the source layout)
    cp = np.stack([c_x1, c_x2, c_y], axis=-1)   # (nu, nv, 3)
    return Surface(basis_u, basis_v, cp, rational=False, raw=True), pbs


class ParametricTrivariateSpline:
    """
    Trivariate analogue of ParametricBivariateSpline.

    Holds four scipy-style coefficient tensors over (u, v, w):
        x1(u,v,w), x2(u,v,w), x3(u,v,w), y(u,v,w)
    and evaluates y at a physical (x1, x2, x3) by Newton inversion in (u,v,w).

    Built so x3 matches the station values supplied at construction, with w
    an affine image of those samples.
    """

    def __init__(self, volume, x3_bounds, log_flags=(False, False, False, False)):
        self.vol = volume                       # splipy.Volume, cp last axis = [x1,x2,x3,y]
        self.x3_min, self.x3_max = x3_bounds
        self.log_x1, self.log_x2, self.log_x3, self.log_y = log_flags
        # parameter domains (clamped knot interiors)
        bu, bv, bw = volume.bases
        self._u_dom = (bu.start(), bu.end())
        self._v_dom = (bv.start(), bv.end())
        self._w_dom = (bw.start(), bw.end())
        self._cross_section_cache = {}

    def _eval_phys(self, u, v, w):
        """Return (x1, x2, x3, y) at parameter (u,v,w)."""
        return self.vol(u, v, w)                # shape (4,)

    def eval(self, x1, x2, x3, tol=1e-10, max_iter=60, extrapolate=False,
             w_distance_threshold=0.5,
             limit_distance=False, limit_consistency=False, limit_steepness=False,
             consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10,
             return_diagnostics=False):
        """
        Evaluate y at physical (x1, x2, x3), scalar or array-like.

        If x1, x2, x3 are all scalars, delegates directly to eval_point.
        Otherwise, any array-like inputs are combined into a rectangular
        grid via an outer product -- mirroring
        ParametricBivariateSpline.eval's (x1, x2) grid convention, and
        working even when the arrays have different lengths -- and
        evaluated element-wise across that grid by looping eval_point,
        since w, and thus the cross-section surface used for
        extrapolation, generally differs per grid point along x3.

        Returns
        -------
        y : float, or ndarray of shape (len(x1), len(x2), len(x3))
            (scalar axes counted as length 1). Points where
            extrapolation/inversion is refused are NaN (mirroring
            ParametricBivariateSpline.eval_grid's convention).
        diagnostics : dict, or object-ndarray of dicts matching the output
            shape, only when return_diagnostics=True.
        """
        eval_params = dict(
            tol=tol, max_iter=max_iter, extrapolate=extrapolate,
            w_distance_threshold=w_distance_threshold,
            limit_distance=limit_distance, limit_consistency=limit_consistency,
            limit_steepness=limit_steepness, consistency_threshold=consistency_threshold,
            distance_threshold=distance_threshold, steepness_threshold=steepness_threshold,
        )

        scalar_input = (not hasattr(x1, '__iter__') and not hasattr(x2, '__iter__')
                         and not hasattr(x3, '__iter__'))
        if scalar_input:
            return self.eval_point(x1, x2, x3, return_diagnostics=return_diagnostics, **eval_params)

        x1_vals = np.atleast_1d(np.asarray(x1, dtype=float))
        x2_vals = np.atleast_1d(np.asarray(x2, dtype=float))
        x3_vals = np.atleast_1d(np.asarray(x3, dtype=float))

        X1, X2, X3 = np.meshgrid(x1_vals, x2_vals, x3_vals, indexing='ij')
        shape = X1.shape
        Y = np.empty(shape, dtype=float)
        diagnostics = np.empty(shape, dtype=object) if return_diagnostics else None

        for idx in np.ndindex(shape):
            result = self.eval_point(X1[idx], X2[idx], X3[idx],
                                      return_diagnostics=return_diagnostics, **eval_params)
            if return_diagnostics:
                y, diag = result
                diagnostics[idx] = diag
            else:
                y = result
            Y[idx] = np.nan if y is None else y

        if return_diagnostics:
            return Y, diagnostics
        return Y

    def eval_point(self, x1, x2, x3, tol=1e-10, max_iter=60, extrapolate=False,
             w_distance_threshold=0.5,
             limit_distance=False, limit_consistency=False, limit_steepness=False,
             consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10,
             return_diagnostics=False):
        """
        Evaluate y at a single physical (x1, x2, x3) point.

        If extrapolate is False (default), (u, v, w) are clamped to the
        volume's parametric domain -- unchanged legacy behaviour.

        If extrapolate is True, x3 may lie outside [x3_min, x3_max]: w is
        computed directly from x3 (an exactly-determined 1-unknown/1-target
        problem, no Newton needed), then a synthetic cross-section
        ParametricBivariateSpline is built at that w -- in-domain or,
        via an extrapolation-tolerant basis evaluator, beyond it -- and
        x1/x2 (in- or out-of-domain) are resolved entirely by that
        surface's own eval_point(extrapolate=True), an already-tested
        exactly-determined 2-unknown/2-target problem. Extrapolation
        refused by the w-distance guard or by the cross-section's own
        reliability checks returns None, mirroring
        ParametricBivariateSpline's convention.

        Parameters
        ----------
        w_distance_threshold : float
            Maximum allowed distance of w outside [0, 1], as a fraction
            of the station domain (mirrors distance_threshold below, but
            in station/w space instead of physical x1/x2 space).
        limit_distance, limit_consistency, limit_steepness,
        consistency_threshold, distance_threshold, steepness_threshold :
            Forwarded to the cross-section's ParametricBivariateSpline.eval_point
            for x1/x2 extrapolation reliability checks.
        return_diagnostics : bool
            If True, also return a dict describing whether/where
            extrapolation occurred.
        """
        if not extrapolate:
            y = self._eval_clipped(x1, x2, x3, tol=tol, max_iter=max_iter)
            if return_diagnostics:
                return y, {'w_extrapolated': False, 'refused': False}
            return y

        x3_int = np.log10(x3) if self.log_x3 else x3
        w = self.x3_to_w(x3_int)
        w_dist = self._w_extrapolation_distance(w)

        if w_dist > w_distance_threshold:
            if return_diagnostics:
                return None, {'w': w, 'w_extrapolated': w_dist > 0.0, 'refused': True}
            return None

        surface = self._cross_section_at_w(w)
        y = surface.eval_point(
            x1, x2, tol=tol, max_iter=max_iter, extrapolate=True,
            limit_distance=limit_distance, limit_consistency=limit_consistency,
            limit_steepness=limit_steepness, consistency_threshold=consistency_threshold,
            distance_threshold=distance_threshold, steepness_threshold=steepness_threshold,
        )

        if return_diagnostics:
            return y, {'w': w, 'w_extrapolated': w_dist > 0.0, 'refused': y is None}
        return y

    def _eval_clipped(self, x1, x2, x3, tol=1e-10, max_iter=60):
        """Invert (x1,x2,x3) -> (u,v,w) by Newton, return y. Mirrors PBS.eval_point."""
        if self.log_x1: x1 = np.log10(x1)
        if self.log_x2: x2 = np.log10(x2)
        if self.log_x3: x3 = np.log10(x3)
        target = np.array([x1, x2, x3], float)

        # initial guess: midpoint of each domain
        u = 0.5 * sum(self._u_dom)
        v = 0.5 * sum(self._v_dom)
        w = 0.5 * sum(self._w_dom)

        for _ in range(max_iter):
            p = self._eval_phys(u, v, w)
            F = p[:3] - target
            if np.all(np.abs(F) < tol * (1 + np.abs(target))):
                break
            # Jacobian d(x1,x2,x3)/d(u,v,w) via splipy derivatives
            du = self.vol.derivative(u, v, w, d=(1, 0, 0))[:3]
            dv = self.vol.derivative(u, v, w, d=(0, 1, 0))[:3]
            dw = self.vol.derivative(u, v, w, d=(0, 0, 1))[:3]
            J = np.column_stack([du, dv, dw])
            try:
                step = np.linalg.solve(J, -F)
            except np.linalg.LinAlgError:
                break
            u = np.clip(u + step[0], *self._u_dom)
            v = np.clip(v + step[1], *self._v_dom)
            w = np.clip(w + step[2], *self._w_dom)

        y = float(self._eval_phys(u, v, w)[3])
        return 10.0 ** y if self.log_y else y

    def x3_to_w(self, x3):
        return (x3 - self.x3_min) / (self.x3_max - self.x3_min)

    def _w_extrapolation_distance(self, w):
        """Distance of w outside [0, 1], as a fraction of the station domain."""
        if w < 0.0:
            return -w
        if w > 1.0:
            return w - 1.0
        return 0.0

    def _w_basis_vector(self, w):
        """
        Basis-function vector for the w (station) axis at w, tolerant of w
        outside [0, 1]. splipy.BSplineBasis clamps to all-zero outside its
        domain; scipy.interpolate.BSpline.design_matrix(extrapolate=True)
        matches it exactly in-domain and extends the boundary polynomial
        piece outside, which is what a Taylor-consistent extrapolation needs.
        """
        bw = self.vol.bases[2]
        k = bw.order - 1
        dm = BSpline.design_matrix([w], bw.knots, k, extrapolate=True)
        return np.asarray(dm.todense()).ravel()

    def _cross_section_at_w(self, w):
        """
        Build (or fetch from cache) the ParametricBivariateSpline that is
        the volume's cross-section at station w -- exact if w is in [0, 1],
        extrapolated along the station axis otherwise.
        """
        cached = self._cross_section_cache.get(w)
        if cached is not None:
            return cached

        Nw = self._w_basis_vector(w)                      # (nw,)
        cp = self.vol.controlpoints                       # (nu, nv, nw, 4) = [x1, x2, x3, y]
        cp_cs = np.einsum('k,ijkd->ijd', Nw, cp)           # (nu, nv, 4)

        bu, bv, _ = self.vol.bases
        tu, tv = bu.knots, bv.knots
        ku, kv = bu.order - 1, bv.order - 1

        # ParametricBivariateSpline expects cp[i,j] = [x1, y, x2]
        cp_pbs = np.empty(cp_cs.shape[:2] + (3,))
        cp_pbs[:, :, 0] = cp_cs[:, :, 0]     # x1
        cp_pbs[:, :, 1] = cp_cs[:, :, 3]     # y
        cp_pbs[:, :, 2] = cp_cs[:, :, 1]     # x2

        surface = ParametricBivariateSpline(
            tu, tv, cp_pbs, ku, kv,
            log_x1=self.log_x1, log_x2=self.log_x2, log_y=self.log_y,
        )
        self._cross_section_cache[w] = surface
        return surface


class LoftedSplineVolume:
    """
    Builds a trivariate spline volume by lofting a stack of parametric
    surfaces across a third station axis (x3), and evaluates y at any
    physical (x1, x2, x3).
    """

    def __init__(self, surfaces, x3_values, w_degree=3,
                 log_flags=(False, False, False, False)):
        """
        Parameters
        ----------
        surfaces : list of ParametricBivariateSpline or str
            One surface per station, each given either as a
            ParametricBivariateSpline instance or as a SplineCloud surface
            UID to be fetched via load_spline_surface. Order must match
            x3_values (ascending).
        x3_values : array-like
            Station value for each surface, strictly increasing.
        w_degree : int
            Spline degree along the station (x3/w) axis.
        log_flags : tuple of 4 bool
            Whether (x1, x2, x3, y) are stored/evaluated in log10 space.
        """
        if len(surfaces) != len(x3_values):
            raise ValueError("surfaces and x3_values must have the same length")

        self.surfaces = [self._resolve_surface(s) for s in surfaces]
        self.x3_values = np.asarray(x3_values, dtype=float)
        self._volume = self._build_volume(w_degree, log_flags)

    @staticmethod
    def _resolve_surface(surface):
        return _resolve_surface(surface)

    def _build_volume(self, w_degree, log_flags):
        """
        Reconcile self.surfaces onto one common (u,v) basis and loft them
        along self.x3_values into a ParametricTrivariateSpline.
        """
        x3_values = self.x3_values
        n = len(self.surfaces)
        assert n == len(x3_values)
        assert np.all(np.diff(x3_values) > 0), "x3_values must be strictly increasing"

        # (a) convert and losslessly reconcile all surfaces to one common (u,v) basis
        surfaces = [pbs_to_splipy(pbs)[0] for pbs in self.surfaces]
        surfaces = [s.clone() for s in surfaces]
        for i in range(n):
            for j in range(i + 1, n):
                Surface.make_splines_identical(surfaces[i], surfaces[j])

        basis_u = surfaces[0].bases[0]
        basis_v = surfaces[0].bases[1]
        nu = basis_u.num_functions()
        nv = basis_v.num_functions()

        # (b) w-basis built from the TRUE x3 samples (normalized to [0,1])
        x3_0, x3_1 = x3_values[0], x3_values[-1]
        w_sites = (x3_values - x3_0) / (x3_1 - x3_0)
        p = min(w_degree, n - 1)
        inner = [np.mean(w_sites[i:i + p]) for i in range(1, n - p)]
        w_knots = np.concatenate(([w_sites[0]] * (p + 1), inner, [w_sites[-1]] * (p + 1)))
        basis_w = BSplineBasis(order=p + 1, knots=w_knots)
        Nw_inv = np.linalg.inv(basis_w(w_sites))         # (n, n)

        # (c) stack control nets and append x3 as an explicit 3rd input coordinate.
        #     Each reconciled surface cp is [x1, x2, y]; we insert x3 -> [x1,x2,x3,y]
        stacked = np.empty((nu, nv, n, 4))
        for k, s in enumerate(surfaces):
            cp = s.controlpoints                          # (nu, nv, 3) = [x1, x2, y]
            stacked[:, :, k, 0] = cp[:, :, 0]             # x1
            stacked[:, :, k, 1] = cp[:, :, 1]             # x2
            stacked[:, :, k, 2] = x3_values[k]            # x3 -- the NEW axis
            stacked[:, :, k, 3] = cp[:, :, 2]             # y

        # solve the w-interpolation so the volume passes through each surface
        cp_vol = np.einsum("wk, ijkd -> ijwd", Nw_inv, stacked)   # (nu, nv, n, 4)

        volume = Volume(basis_u, basis_v, basis_w, cp_vol, rational=False, raw=True)
        return ParametricTrivariateSpline(volume, (x3_0, x3_1), log_flags=log_flags)

    def eval(self, x1, x2, x3, tol=1e-10, max_iter=60, extrapolate=False,
             w_distance_threshold=0.5,
             limit_distance=False, limit_consistency=False, limit_steepness=False,
             consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10,
             return_diagnostics=False):
        return self._volume.eval(
            x1, x2, x3, tol=tol, max_iter=max_iter, extrapolate=extrapolate,
            w_distance_threshold=w_distance_threshold,
            limit_distance=limit_distance, limit_consistency=limit_consistency,
            limit_steepness=limit_steepness, consistency_threshold=consistency_threshold,
            distance_threshold=distance_threshold, steepness_threshold=steepness_threshold,
            return_diagnostics=return_diagnostics,
        )


class InterpolatedSplineVolume:
    """
    Builds a pseudo-volume by evaluating a stack of independent parametric
    surfaces at physical (x1, x2), then interpolating those station values
    along a third axis (x3), instead of lofting the surfaces into one true
    trivariate B-spline volume like LoftedSplineVolume does.

    Because each station's surface is evaluated independently -- there is
    no Surface.make_splines_identical reconciliation step -- the surfaces
    are free to differ arbitrarily in structure (knot vectors, order,
    control-net resolution); x1/x2 extrapolation reliability is exactly
    ParametricBivariateSpline.eval's own (per station), while x3 behaviour
    (clipping / extrapolation / the distance guard) mirrors
    ParametricTrivariateSpline.eval's.
    """

    def __init__(self, surfaces, x3_station_values):
        """
        Parameters
        ----------
        surfaces : list of ParametricBivariateSpline or str
            One surface per station, each given either as a
            ParametricBivariateSpline instance or as a SplineCloud surface
            UID to be fetched via load_spline_surface. Order must match
            x3_station_values (ascending).
        x3_station_values : array-like
            Station value for each surface, strictly increasing.
        """
        if len(surfaces) != len(x3_station_values):
            raise ValueError("surfaces and x3_station_values must have the same length")

        self.surfaces = [_resolve_surface(s) for s in surfaces]
        self.stations = np.asarray(x3_station_values, dtype=float)
        if not np.all(np.diff(self.stations) > 0):
            raise ValueError("x3_station_values must be strictly increasing")

    def _y_slices(self, x1_vals, x2_vals, extrapolate,
                   limit_distance, limit_consistency, limit_steepness,
                   consistency_threshold, distance_threshold, steepness_threshold):
        """Evaluate every station surface once on the (x1, x2) grid: (n_stations, Nx1, Nx2)."""
        y_slices = np.empty((len(self.surfaces), len(x1_vals), len(x2_vals)))
        for k, surf in enumerate(self.surfaces):
            _, _, Y = surf.eval(
                x1_vals, x2_vals, extrapolate=extrapolate,
                limit_distance=limit_distance, limit_consistency=limit_consistency,
                limit_steepness=limit_steepness, consistency_threshold=consistency_threshold,
                distance_threshold=distance_threshold, steepness_threshold=steepness_threshold,
            )
            y_slices[k] = Y
        return y_slices

    def _interpolate_x3(self, y_slices, x3_query, method, interp_degree):
        """
        Interpolate y_slices (n_stations, Nx1, Nx2) along axis 0 at
        x3_query -> (Nx1, Nx2, Nx3).

        A given (x1, x2) grid point can be NaN at some stations (that
        station's own surface refused x1/x2, e.g. it's outside that
        station's own domain) -- those columns are excluded from the fit
        entirely (scipy's interpolators reject NaN input outright) and
        their whole x3 output is NaN, since interpolation can't recover a
        missing station.
        """
        n = len(self.surfaces)
        method = method.lower()
        grid_shape = y_slices.shape[1:]                        # (Nx1, Nx2)

        flat = y_slices.reshape(n, -1)                         # (n, Nx1*Nx2)
        good = np.all(np.isfinite(flat), axis=0)                # (Nx1*Nx2,)

        Y_flat = np.full((flat.shape[1],) + x3_query.shape, np.nan)  # (Nx1*Nx2, Nx3)

        if good.any():
            good_flat = flat[:, good]                          # (n, n_good)

            if method == 'pchip':
                # PCHIP prevents overshooting between stations (monotone Hermite).
                interp = PchipInterpolator(self.stations, good_flat, axis=0, extrapolate=True)
                result = interp(x3_query)                      # (Nx3, n_good)
                Y_flat[good] = np.moveaxis(result, 0, -1)
            elif method == 'spline':
                k = min(interp_degree, n - 1)
                interp = make_interp_spline(self.stations, good_flat, k=k, axis=0)
                result = interp(x3_query, extrapolate=True)    # (Nx3, n_good)
                Y_flat[good] = np.moveaxis(result, 0, -1)
            elif method == 'poly':
                deg = min(interp_degree, n - 1)
                coefs = nppoly.polyfit(self.stations, good_flat, deg=deg)
                Y_flat[good] = nppoly.polyval(x3_query, coefs, tensor=True)  # (n_good, Nx3)
            else:
                raise ValueError(f"Unknown method {method!r}; expected 'pchip', 'spline', or 'poly'")

        return Y_flat.reshape(grid_shape + x3_query.shape)     # (Nx1, Nx2, Nx3)

    def eval(self, x1, x2, x3, extrapolate=False, method='pchip', interp_degree=3,
             x3_distance_threshold=0.5,
             limit_distance=False, limit_consistency=False, limit_steepness=False,
             consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10,
             return_diagnostics=False):
        """
        Evaluate y at physical (x1, x2, x3), scalar or array-like.

        x1, x2 are resolved per station independently via
        ParametricBivariateSpline.eval (extrapolate and the limit_*/
        *_threshold reliability checks are forwarded to it unchanged --
        including its own convention of returning NaN for a refused
        point, regardless of the extrapolate flag). x3 is then resolved
        by 1D interpolation across stations:

        If extrapolate is False (default), x3 is clipped into
        [x3_min, x3_max] before interpolating -- mirrors
        ParametricTrivariateSpline.eval's clipped default.

        If extrapolate is True, x3 may lie outside [x3_min, x3_max]: the
        interpolator (pchip/spline) is evaluated past its fitted range,
        or, for 'poly', the fitted polynomial (defined everywhere) is
        evaluated directly. A distance guard refuses (NaN) x3 beyond
        x3_distance_threshold, mirroring
        ParametricTrivariateSpline.eval's w_distance_threshold.

        Any array-like inputs among x1, x2, x3 are combined into a
        rectangular grid via an outer product, exactly like
        LoftedSplineVolume.eval.

        Parameters
        ----------
        method : {'pchip', 'spline', 'poly'}
            1D interpolation scheme along the station (x3) axis.
            'pchip' is a monotone cubic Hermite interpolant (no
            overshoot). 'spline' is a degree-interp_degree B-spline
            interpolant. 'poly' is a degree-interp_degree polynomial fit
            (exact interpolation when interp_degree == n_stations - 1;
            ill-conditioned for many stations -- prefer 'spline' then).
        interp_degree : int
            Degree used by 'spline'/'poly' (capped at n_stations - 1);
            unused by 'pchip', which is always cubic Hermite.
        x3_distance_threshold : float
            Maximum allowed distance of x3 outside [x3_min, x3_max], as a
            fraction of the station domain.
        return_diagnostics : bool
            If True, also return a dict (or, for array input, a matching
            object-ndarray of dicts) describing whether/where x3 was
            extrapolated or refused.

        Returns
        -------
        y : float, or ndarray of shape (len(x1), len(x2), len(x3))
            (scalar axes counted as length 1).
        diagnostics : dict, or object-ndarray of dicts matching the output
            shape, only when return_diagnostics=True.
        """
        scalar_input = (not hasattr(x1, '__iter__') and not hasattr(x2, '__iter__')
                         and not hasattr(x3, '__iter__'))

        x1_vals = np.atleast_1d(np.asarray(x1, dtype=float))
        x2_vals = np.atleast_1d(np.asarray(x2, dtype=float))
        x3_vals = np.atleast_1d(np.asarray(x3, dtype=float))

        x3_min, x3_max = self.stations[0], self.stations[-1]
        w = (x3_vals - x3_min) / (x3_max - x3_min)
        w_dist = np.where(w < 0.0, -w, np.where(w > 1.0, w - 1.0, 0.0))

        if extrapolate:
            x3_query = x3_vals
            refused = w_dist > x3_distance_threshold
        else:
            x3_query = np.clip(x3_vals, x3_min, x3_max)
            refused = np.zeros_like(x3_vals, dtype=bool)

        y_slices = self._y_slices(
            x1_vals, x2_vals, extrapolate,
            limit_distance, limit_consistency, limit_steepness,
            consistency_threshold, distance_threshold, steepness_threshold,
        )
        Y = self._interpolate_x3(y_slices, x3_query, method, interp_degree)  # (Nx1, Nx2, Nx3)

        if refused.any():
            Y[..., refused] = np.nan

        if return_diagnostics:
            diagnostics = np.empty(Y.shape, dtype=object)
            for k in range(len(x3_vals)):
                diagnostics[..., k] = {'x3_extrapolated': bool(w_dist[k] > 0.0),
                                        'refused': bool(refused[k])}
            if scalar_input:
                y = Y.reshape(())[()]
                return (None if refused[0] else float(y)), diagnostics.reshape(())[()]
            return Y, diagnostics

        if scalar_input:
            y = Y.reshape(())[()]
            return None if refused[0] else float(y)
        return Y
        
        return float(interp_1d(x3))