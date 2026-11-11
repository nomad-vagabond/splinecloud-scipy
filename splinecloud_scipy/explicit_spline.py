import math
from typing import Sequence, Union

import numpy as np
from scipy.interpolate import BSpline

from .parametric_spline import ParametricUnivariateSpline


LOG_SCALE = "Logarithmic"
EXTRAPOLATION_POLICIES = ("none", "linear", "clamp", "polynomial")
## Points within this fraction of the domain width outside a boundary count as inside, so
## round-off (e.g. a log10/10** round trip of the last data point) doesn't produce NaN
DOMAIN_RTOL = 1e-12


class NdBsplExplicit:
    """
    Explicit tensor-product B-spline function y = f(x1, ..., xN).

    Built from SplineCloud's explicit spline schema (v1):
    ``{v, nx, ny, k[], t[], shape, c, domain, scales, extrapolation}``.
    Knots are clamped and given in each axis's *scaled* space (log10 for logarithmic axes).
    Coefficients are flat, in C order. Evaluation is direct - no inversion of a parametric
    map is needed.

    Parameters
    ----------
    t : sequence of 1-D array-like
        Knot vector per input axis (scaled space).
    k : sequence of int
        Degree per input axis.
    c : array-like
        Coefficients, flat (C order) or already shaped as ``len(t[i]) - k[i] - 1`` per axis.
    scales : sequence of str, optional
        ``"Linear"`` / ``"Logarithmic"`` for each input axis followed by the output.
    extrapolation : str
        Default policy for points outside the domain: ``"none"`` (NaN), ``"linear"``
        (first-order Taylor from the nearest domain point), ``"clamp"`` (value at the
        nearest domain point) or ``"polynomial"`` (continue the edge polynomial pieces).
    """

    def __init__(self, t, k, c, scales=None, extrapolation="none"):
        self.t = tuple(np.asarray(ti, dtype=float) for ti in t)
        self.k = tuple(int(ki) for ki in k)
        if len(self.t) != len(self.k):
            raise ValueError("t and k must have one entry per input axis")

        self.ndim = len(self.t)
        self.shape = tuple(len(ti) - ki - 1 for ti, ki in zip(self.t, self.k))
        self.c = np.asarray(c, dtype=float).reshape(self.shape)

        scales = list(scales) if scales is not None else ["Linear"] * (self.ndim + 1)
        if len(scales) != self.ndim + 1:
            raise ValueError("scales must list every input axis followed by the output")
        self.log_x = np.array([s == LOG_SCALE for s in scales[:-1]])
        self.log_y = scales[-1] == LOG_SCALE

        self._check_policy(extrapolation)
        self.extrapolation = extrapolation

        ## domain in scaled space (clamped knots: [t[k], t[-k-1]])
        self._domain = np.array([[ti[ki], ti[-ki - 1]] for ti, ki in zip(self.t, self.k)])
        self._spline = self._build_spline()

    @classmethod
    def from_json(cls, payload: dict, extrapolation=None):
        """Build from an explicit schema v1 payload (the ``spline`` field of an API response)."""
        if payload.get("v") != 1:
            raise ValueError(f"Unsupported explicit spline schema version: {payload.get('v')}")
        if payload.get("ny", 1) != 1:
            raise ValueError("Only single-output explicit splines (ny = 1) are supported")
        return cls(payload["t"], payload["k"], payload["c"], scales=payload.get("scales"),
                   extrapolation=extrapolation or payload.get("extrapolation", "none"))

    def _build_spline(self):
        if self.ndim == 1:
            return BSpline(self.t[0], self.c, self.k[0], extrapolate=True)
        try:
            from scipy.interpolate import NdBSpline
        except ImportError as ex:
            raise ImportError("Explicit splines with more than one input require "
                              "scipy >= 1.12 (scipy.interpolate.NdBSpline)") from ex
        return NdBSpline(self.t, self.c, self.k, extrapolate=True)

    @staticmethod
    def _check_policy(policy):
        if policy not in EXTRAPOLATION_POLICIES:
            raise ValueError(f"Invalid extrapolation policy: {policy!r}. "
                             f"Use one of {EXTRAPOLATION_POLICIES}")

    @property
    def domain(self) -> np.ndarray:
        """(ndim, 2) array of [min, max] per input axis, in physical units."""
        return np.where(self.log_x[:, None], np.power(10.0, self._domain), self._domain)

    ## --------------------------------------------------------------- evaluation

    def _raw(self, xs, nu=None):
        ## xs: (n, ndim) in scaled space; returns scaled-space values (or derivatives)
        if self.ndim == 1:
            spline = self._spline if not nu else self._spline.derivative(nu[0])
            return spline(xs[:, 0])
        return self._spline(xs, nu=nu)

    def _outside(self, xs):
        ## (n,) bool: farther than DOMAIN_RTOL * width outside the domain (or NaN, e.g. log of <= 0)
        lo, hi = self._domain[:, 0], self._domain[:, 1]
        tol = DOMAIN_RTOL * (hi - lo)
        with np.errstate(invalid="ignore"):
            inside = (xs >= lo - tol) & (xs <= hi + tol)
        return ~np.all(inside, axis=1)

    def _to_scaled(self, points):
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.shape[1] != self.ndim:
            raise ValueError(f"Expected points of shape (n, {self.ndim}), got {points.shape}")
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.log_x, np.log10(points), points)

    def eval(self, points, extrapolate=None) -> np.ndarray:
        """
        Evaluate at scattered points.

        Parameters
        ----------
        points : array-like, shape (n, ndim)
            Physical input coordinates (a 1-D array is accepted for ndim == 1).
        extrapolate : str, optional
            Extrapolation policy for this call; defaults to the object's policy.

        Returns
        -------
        ndarray, shape (n,)
        """
        policy = self.extrapolation if extrapolate is None else extrapolate
        self._check_policy(policy)

        points = np.asarray(points, dtype=float)
        if self.ndim == 1 and points.ndim <= 1:
            points = points.reshape(-1, 1)
        xs = self._to_scaled(points)

        nearest = np.clip(xs, self._domain[:, 0], self._domain[:, 1])
        outside = self._outside(xs)

        if policy == "polynomial":
            y = self._raw(xs)
        elif policy == "clamp":
            y = self._raw(nearest)
        else:
            y = self._raw(nearest)
            if policy == "linear" and outside.any():
                for d in range(self.ndim):
                    dx = xs[outside, d] - nearest[outside, d]
                    if np.any(dx):
                        nu = tuple(int(i == d) for i in range(self.ndim))
                        y[outside] += self._raw(nearest[outside], nu=nu) * dx
            elif policy == "none":
                y = np.where(outside, np.nan, y)

        return np.power(10.0, y) if self.log_y else y

    def __call__(self, *xs):
        """Pointwise evaluation with NumPy broadcasting: ``f(x1, ..., xN)``."""
        if len(xs) != self.ndim:
            raise TypeError(f"Expected {self.ndim} coordinate arguments, got {len(xs)}")
        arrays = np.broadcast_arrays(*[np.asarray(x, dtype=float) for x in xs])
        shape = arrays[0].shape
        y = self.eval(np.stack([a.ravel() for a in arrays], axis=-1))
        return float(y[0]) if shape == () else y.reshape(shape)

    def eval_grid(self, *axes, extrapolate=None) -> np.ndarray:
        """Evaluate on the outer product of per-axis values; returns shape (n1, ..., nN)."""
        if len(axes) != self.ndim:
            raise TypeError(f"Expected {self.ndim} axes, got {len(axes)}")
        axes = [np.atleast_1d(np.asarray(a, dtype=float)) for a in axes]
        mesh = np.meshgrid(*axes, indexing="ij")
        points = np.stack([m.ravel() for m in mesh], axis=-1)
        return self.eval(points, extrapolate=extrapolate).reshape(mesh[0].shape)

    def grad(self, points) -> np.ndarray:
        """Gradient dy/dx_i in physical units at points inside the domain; shape (n, ndim).
        Points outside the domain give NaN."""
        points = np.asarray(points, dtype=float)
        if self.ndim == 1 and points.ndim <= 1:
            points = points.reshape(-1, 1)
        xs = np.clip(self._to_scaled(points), self._domain[:, 0], self._domain[:, 1])
        outside = self._outside(self._to_scaled(points))

        g = np.column_stack([self._raw(xs, nu=tuple(int(i == d) for i in range(self.ndim)))
                             for d in range(self.ndim)])
        if self.log_y:
            g = g * (np.power(10.0, self._raw(xs)) * math.log(10))[:, None]
        g = np.where(self.log_x, g / (points * math.log(10)), g)
        g[outside] = np.nan
        return g


class ExplicitUnivariateSpline(NdBsplExplicit):
    """
    Explicit spline curve y = f(x) loaded from SplineCloud (``form == "explicit"``).

    Drop-in replacement for ParametricUnivariateSpline when the curve is a function of x:
    the same ``eval(x, extrapolate=False)`` and ``fit_accuracy`` API, evaluated directly
    instead of by inverting x(t).
    """

    def __init__(self, t, k, c, scales=None, extrapolation="none"):
        super().__init__(t, k, c, scales=scales, extrapolation=extrapolation)
        if self.ndim != 1:
            raise ValueError("ExplicitUnivariateSpline needs exactly one input axis")

    @property
    def knots(self) -> np.ndarray:
        """Knot vector in (scaled) x space."""
        return self.t[0]

    @property
    def degree(self) -> int:
        return self.k[0]

    @property
    def coeffs(self) -> np.ndarray:
        return self.c

    def eval(self, x: Union[float, Sequence[float]], extrapolate=False):
        """
        Evaluate y(x) for a scalar or an iterable of x values.

        ``extrapolate`` keeps ParametricUnivariateSpline's meaning: False gives NaN outside
        the domain, True continues the edge polynomial pieces. A policy name
        (``"none"``, ``"linear"``, ``"clamp"``, ``"polynomial"``) can also be passed.
        """
        if isinstance(extrapolate, bool):
            extrapolate = "polynomial" if extrapolate else "none"
        if hasattr(x, '__iter__'):
            return super().eval(np.asarray(x, dtype=float), extrapolate=extrapolate)
        return float(super().eval(np.array([float(x)]), extrapolate=extrapolate)[0])

    def fit_accuracy(self, points: Sequence[Sequence[float]], weights=None,
                     method="RMSE") -> float:
        """Same as ParametricUnivariateSpline.fit_accuracy ('RMSE', 'MSE' or 'MAE')."""
        pnum = len(points)
        weights = np.ones(pnum) if weights is None else np.asarray(weights, dtype=float)
        x_vals, y_vals = np.array(points, dtype=float).T
        residuals = y_vals - self.eval(x_vals, extrapolate=True)

        if method in ("RMSE", "MSE"):
            error = float(np.sum((weights * residuals)**2) / pnum)
            return math.sqrt(error) if method == "RMSE" else error
        if method == "MAE":
            return float(np.sum(weights * np.abs(residuals)) / pnum)
        raise ValueError(f"Invalid method: {method}. Should be one of 'RMSE', 'MSE', or 'MAE'")

    def to_parametric(self) -> ParametricUnivariateSpline:
        """The exact parametric form (x(t), y(t)): control points at the Greville abscissae,
        knots normalised to [0, 1]. Same as the legacy API representation."""
        t, k = self.t[0], self.k[0]
        greville = np.array([t[i + 1:i + k + 1].mean() for i in range(len(self.c))])
        t_norm = (t - t[0]) / (t[-1] - t[0])
        return ParametricUnivariateSpline((t_norm, greville, self.c, k),
                                          log_x=bool(self.log_x[0]), log_y=self.log_y)
