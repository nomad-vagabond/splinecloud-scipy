"""
Lofted spline volumes served by SplineCloud: y = f(x1, x2, x3) as a parametric B-spline
volume, the parametric volume schema v1 (contract C.7):

    {"v": 1, "nx": 3, "ny": 1, "form": "parametric",
     "tu", "tv", "tw", "ku", "kv", "kw",
     "cp": [nu][nv][nw][4] = [x1, x2, x3, y],  "w": [nu][nv][nw] (all 1),
     "x3_range": [x3 of the first station, x3 of the last], "scales": [x1, x2, x3, y]}

Control points and x3_range are in scaled space (log10 on "Logarithmic" axes). The
volume is lofted through 2-D models (stations) so that x3 is exactly affine in w:

    x3(u, v, w) = x3_range[0] + w * (x3_range[1] - x3_range[0])

Evaluation therefore takes w straight from x3 and inverts only the slice at that w
(an exact ParametricBivariateSpline) for (x1, x2) -> (u, v). Values in and out are
physical; log axes are handled.
"""
import numpy as np
from scipy.interpolate import BSpline

from .parametric_spline_surface import ParametricBivariateSpline


VOLUME_SCHEMA_VERSION = 1
LOG = "Logarithmic"
## cross-sections kept per volume (slices at the most recent w values)
SLICE_CACHE_SIZE = 64


class ParametricSplineVolume:
    """
    A parametric spline volume (C.7) evaluated at physical (x1, x2, x3).

    - ``eval(x1, x2, x3)``: a float for scalars; for array-likes the outer-product grid,
      shape ``(len(x1), len(x2), len(x3))``, NaN where evaluation is refused
    - ``eval_point(x1, x2, x3)``: one point (None when refused)
    - ``slice_at(x3)``: the exact surface at x3, a ParametricBivariateSpline

    Without ``extrapolate``, x3 is clamped to the stations' range, and (x1, x2) outside the
    slice give None (NaN in grids), as for lofted surfaces. With it, x3 may leave the range by up to ``w_distance_threshold``
    (a share of the range; farther is refused) and (x1, x2) are extrapolated by the
    slice (ParametricBivariateSpline.eval_point, with its reliability options).
    """

    def __init__(self, payload):
        if payload.get("v") != VOLUME_SCHEMA_VERSION:
            raise ValueError(f"Unsupported volume schema version: {payload.get('v')}")
        if payload.get("form") != "parametric":
            raise ValueError("ParametricSplineVolume needs a parametric volume payload")
        if payload.get("nx") != 3 or payload.get("ny", 1) != 1:
            raise ValueError("A parametric volume has 3 inputs and 1 output")

        self.tu = np.asarray(payload["tu"], dtype=float)
        self.tv = np.asarray(payload["tv"], dtype=float)
        self.tw = np.asarray(payload["tw"], dtype=float)
        self.ku, self.kv, self.kw = int(payload["ku"]), int(payload["kv"]), int(payload["kw"])
        self.cp = np.asarray(payload["cp"], dtype=float)            ## (nu, nv, nw, 4)
        sizes = tuple(len(t) - k - 1 for t, k in ((self.tu, self.ku), (self.tv, self.kv),
                                                 (self.tw, self.kw)))
        if self.cp.shape != sizes + (4,):
            raise ValueError(f"Control net shape {self.cp.shape} doesn't match the knots "
                             f"{sizes + (4,)}")
        weights = payload.get("w")
        if weights is not None and not np.allclose(np.asarray(weights, dtype=float), 1.0):
            raise ValueError("Rational volumes (weights other than 1) aren't supported")

        self.x3_range = tuple(float(v) for v in payload["x3_range"])    ## scaled
        self.scales = list(payload.get("scales") or ["Linear"] * 4)
        self.log_x1, self.log_x2, self.log_x3, self.log_y = (s == LOG for s in self.scales)
        self._slices = {}

    @classmethod
    def from_json(cls, payload):
        return cls(payload)

    ## ---------------------------------------------------------------- x3 and w

    @property
    def x3_bounds(self):
        """The stations' x3 range, physical."""
        lo, hi = self.x3_range
        return (10.0 ** lo, 10.0 ** hi) if self.log_x3 else (lo, hi)

    def x3_to_w(self, x3):
        """w of a physical x3 (outside [0, 1] beyond the stations)."""
        if self.log_x3:
            if x3 <= 0:
                raise ValueError("x3 must be positive on a logarithmic axis")
            x3 = np.log10(x3)
        lo, hi = self.x3_range
        return (float(x3) - lo) / (hi - lo)

    @staticmethod
    def _w_distance(w):
        """How far w lies outside [0, 1], as a share of the stations' range."""
        return -w if w < 0.0 else (w - 1.0 if w > 1.0 else 0.0)

    ## ---------------------------------------------------------------- slices

    def _slice_at_w(self, w):
        cached = self._slices.get(w)
        if cached is not None:
            return cached
        ## the w basis at w (extended beyond [0, 1] for extrapolation)
        basis = BSpline.design_matrix([w], self.tw, self.kw, extrapolate=True).toarray()[0]
        cp = np.tensordot(self.cp, basis, axes=(2, 0))               ## (nu, nv, 4)
        surface = ParametricBivariateSpline(
            self.tu, self.tv, cp[:, :, [0, 1, 3]], self.ku, self.kv,  ## [x1, x2, y]
            log_x1=self.log_x1, log_x2=self.log_x2, log_y=self.log_y)
        if len(self._slices) >= SLICE_CACHE_SIZE:
            self._slices.pop(next(iter(self._slices)))
        self._slices[w] = surface
        return surface

    def slice_at(self, x3, extrapolate=False):
        """
        The volume's surface at a physical x3: exact (x3 is affine in w), a
        ParametricBivariateSpline over (x1, x2) with the volume's scales. At a station it
        is that station's surface. Outside the stations' range it raises ValueError unless
        ``extrapolate`` (the boundary w polynomials extended).
        """
        w = self.x3_to_w(x3)
        if not extrapolate and self._w_distance(w) > 1e-12:
            raise ValueError(f"x3 = {x3} lies outside the stations' range {self.x3_bounds}")
        return self._slice_at_w(w)

    ## ---------------------------------------------------------------- evaluation

    def eval_point(self, x1, x2, x3, tol=1e-10, max_iter=60, extrapolate=False,
                   w_distance_threshold=0.5,
                   limit_distance=False, limit_consistency=False, limit_steepness=False,
                   consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10,
                   return_diagnostics=False):
        """y at one physical (x1, x2, x3); None when extrapolation is refused."""
        w = self.x3_to_w(x3)
        w_dist = self._w_distance(w)
        if not extrapolate:
            w = min(max(w, 0.0), 1.0)
        elif w_dist > w_distance_threshold:
            diagnostics = {"w": w, "w_extrapolated": True, "refused": True}
            return (None, diagnostics) if return_diagnostics else None

        options = {}
        if extrapolate:
            options = dict(limit_distance=limit_distance, limit_consistency=limit_consistency,
                           limit_steepness=limit_steepness,
                           consistency_threshold=consistency_threshold,
                           distance_threshold=distance_threshold,
                           steepness_threshold=steepness_threshold)
        y = self._slice_at_w(w).eval_point(x1, x2, tol=tol, max_iter=max_iter,
                                           extrapolate=extrapolate, **options)
        if return_diagnostics:
            return y, {"w": w, "w_extrapolated": bool(extrapolate and w_dist > 0.0),
                       "refused": y is None}
        return y

    def eval(self, x1, x2, x3, tol=1e-10, max_iter=60, extrapolate=False,
             w_distance_threshold=0.5,
             limit_distance=False, limit_consistency=False, limit_steepness=False,
             consistency_threshold=0.5, distance_threshold=0.5, steepness_threshold=10,
             return_diagnostics=False):
        """
        y at physical (x1, x2, x3): a float for three scalars, else the outer-product grid
        of the inputs (scalars count as length 1), shape ``(len(x1), len(x2), len(x3))``,
        NaN where refused (as ParametricBivariateSpline.eval_grid). With
        ``return_diagnostics``, also a dict per point.
        """
        params = dict(tol=tol, max_iter=max_iter, extrapolate=extrapolate,
                      w_distance_threshold=w_distance_threshold,
                      limit_distance=limit_distance, limit_consistency=limit_consistency,
                      limit_steepness=limit_steepness,
                      consistency_threshold=consistency_threshold,
                      distance_threshold=distance_threshold,
                      steepness_threshold=steepness_threshold)
        if not any(hasattr(x, "__iter__") for x in (x1, x2, x3)):
            return self.eval_point(x1, x2, x3, return_diagnostics=return_diagnostics,
                                   **params)

        axes = [np.atleast_1d(np.asarray(x, dtype=float)) for x in (x1, x2, x3)]
        X1, X2, X3 = np.meshgrid(*axes, indexing="ij")
        Y = np.empty(X1.shape, dtype=float)
        diagnostics = np.empty(X1.shape, dtype=object) if return_diagnostics else None
        for idx in np.ndindex(X1.shape):
            result = self.eval_point(X1[idx], X2[idx], X3[idx],
                                     return_diagnostics=return_diagnostics, **params)
            if return_diagnostics:
                result, diagnostics[idx] = result
            Y[idx] = np.nan if result is None else result
        return (Y, diagnostics) if return_diagnostics else Y
