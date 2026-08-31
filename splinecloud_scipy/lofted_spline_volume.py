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
from scipy.interpolate import BSpline

from .parametric_spline_surface import ParametricBivariateSpline
from .api_client import load_spline_surface


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
        Evaluate y at physical (x1, x2, x3).

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
        if isinstance(surface, ParametricBivariateSpline):
            return surface
        if isinstance(surface, str):
            return load_spline_surface(surface)
        raise TypeError(
            "Each surface must be a ParametricBivariateSpline instance or a "
            f"surface UID string, got {type(surface).__name__!r}"
        )

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
