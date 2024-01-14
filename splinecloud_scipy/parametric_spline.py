import math
from typing import Union, Sequence
import numpy as np
import scipy.interpolate as si

from .piecewise_polynomial import PPolyInvertible


Vector1D = Union[list[float], np.ndarray]
Vector2D = Sequence[Sequence[float]]


class ParametricUnivariateSpline:

    def __init__(self, tcck: Union[tuple, list]):
        """
        Construct a parametric spline object from tcck - tuple or list of BSpline parameters

        Parameters:
        ----------
        t: Vector1D - knot Vector1D
        cc: tuple of control point vectors - (x:Vector1D, y:Vector1D)
        k: int - spline degree

        Returns:
        --------
        ParametricUnivariateSpline
        """
        t, cx, cy, k = tcck
        self.k = k
        self.knots = np.array(t)
        self.knots_norm = self._normalize_knotvector()
        self.coeffs_x = np.array(cx)
        self.coeffs_y = np.array(cy)
        
        self._build_splines()
        self._build_ppolyrep()

    def __call__(self, tpoints:Union[float, Vector1D]):
        x_points = self.spline_x(tpoints)
        y_points = self.spline_y(tpoints)
        
        return x_points, y_points

    def _normalize_knotvector(self):
        knots = self.knots
        num_knots = len(knots)
        ka = (knots[-1] - knots[0]) / 1.0
        knots_norm = np.empty(num_knots)
        for i in range(num_knots):
            knots_norm[i] = 1.0 - ((knots[-1] - knots[i])) / ka
        
        return knots_norm

    def _build_splines(self):
        tck_x = self.knots_norm, self.coeffs_x, self.k
        tck_y = self.knots_norm, self.coeffs_y, self.k

        self.spline_x = si.UnivariateSpline._from_tck(tck_x)
        self.spline_y = si.UnivariateSpline._from_tck(tck_y)

        self.spline_x.tck = tck_x
        self.spline_y.tck = tck_y

    def _build_ppolyrep(self):
        self.spline_x.ppoly = PPolyInvertible.from_splinefunc(self.spline_x, extrapolate=True)
        self.spline_y.ppoly = PPolyInvertible.from_splinefunc(self.spline_y, extrapolate=True)

    def eval(self, x:Union[float, Vector1D], extrapolate=False):
        if hasattr(x, '__iter__'):
            t = np.array([self.spline_x.ppoly.evalinv(xi, extrapolate=extrapolate) for xi in x])
            return self.spline_y.ppoly(t, extrapolate=extrapolate)
        
        else:
            t = self.spline_x.ppoly.evalinv(x, extrapolate=extrapolate)
            return float(self.spline_y.ppoly(t, extrapolate=extrapolate))

    def fit_accuracy(self, points:Vector2D, weights=None, method="RMSE") -> float:
        pnum = len(points)
        if weights is None:
            weights = np.ones(pnum)
       
        x_vals, y_vals = np.array(points).T
        y_evaluated = self.eval(x_vals, extrapolate=True)
        
        if method in ["RMSE", "MSE"]:
            residuals = [(weights[i] * (y_vals[i] - y_evaluated[i]))**2 for i in range(pnum)]
        elif method == "MAE":
            residuals = [weights[i] * abs(y_vals[i] - y_evaluated[i]) for i in range(pnum)]
        else:
            raise ValueError(f"Invalid method: {method}. Should be one of 'RMSE', 'MSE', or 'MAE'")

        if method == "RMSE":
            error = math.sqrt(sum(residuals)/pnum)
        else:
            error = sum(residuals)/pnum

        return error
