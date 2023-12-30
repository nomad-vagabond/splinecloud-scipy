from typing import Union
import numpy as np
import scipy.interpolate as si

from .piecewise_polynomial import PPolyInvertible


_error_msg = {

    1: 'Parameter t must be a list or an array that represents knot vector.',

    2: 'Method parameter must be one of: "interp", "smooth", "lsq".'

}

Vector = Union[list[float], np.array]


class ParametricUnivariateSpline:

    @classmethod
    def from_tcck(cls, tcck: Union[tuple, list]):
        """
        Construct a parametric spline object from tcck - tuple or list of BSpline parameters

        Parameters:
        ----------
        t: Vector - knot vector
        cc: tuple of control point vectors - (x:Vector, y:Vector)
        k: int - spline degree

        Returns:
        --------
        ParametricUnivariateSpline
        """
        self = cls.__new__(cls)

        t, cx, cy, k = tcck
        self.k = k
        self.knots = np.array(t)
        self.knots_norm = self._normalize_knotvector()
        self.coeffs_x = np.array(cx)
        self.coeffs_y = np.array(cy)
        
        self._build_splines()
        self._build_ppolyrep()
        
        return self

    def __call__(self, tpoints):
        x_points = self.spline_x(tpoints)
        y_points = self.spline_y(tpoints)
        
        return x_points, y_points

    def _normalize_knotvector(self, knots=None, d=1.0):
        knots = knots or self.knots
        num_knots = len(knots)
        ka = (knots[-1] - knots[0]) / d
        knots_norm = np.empty(num_knots)
        for i in range(num_knots):
            knots_norm[i] = d - ((knots[-1] - knots[i])) / ka
        
        return knots_norm

    def _build_splines(self):
        tck_x = self.knots_norm, self.coeffs_x, self.k
        tck_y = self.knots_norm, self.coeffs_y, self.k

        self.spline_x = si.UnivariateSpline._from_tck(tck_x)
        self.spline_y = si.UnivariateSpline._from_tck(tck_y)

        self.spline_x.tck = tck_x
        self.spline_y.tck = tck_y

    def _build_ppolyrep(self):
        self.spline_x.ppoly = PPolyInvertible.from_splinefunc(self.spline_x)
        self.spline_y.ppoly = PPolyInvertible.from_splinefunc(self.spline_y)

    def eval(self, x):
        if hasattr(x, '__iter__'):
            t = np.array([self.spline_x.ppoly.evalinv(xi) for xi in x])
            return self.spline_y.ppoly(t)
        
        else:
            t = self.spline_x.ppoly.evalinv(x)
            return self.spline_y.ppoly(t)
