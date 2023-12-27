import numpy as np
import scipy.interpolate as si

from .piecewise_polynomial import PPolyInvertible


_error_msg = {

    1: 'Parameter t must be a list or an array that represents knot vector.',

    2: 'Method parameter must be one of: "interp", "smooth", "lsq".'

}


class ParametricUnivariateSpline:

    @classmethod
    def from_tcck(cls, tcck):
        """Construct a parametric spline object from tcck"""
        self = cls.__new__(cls)
        t, cx, cy, k = tcck
        self.k = k
        self.knots = t
        self.knots_norm = self._normalize_knotvector()
        self.coeffs_x = cx
        self.coeffs_y = cy
        
        self._build_splines(self.coeffs_x, self.coeffs_y)
        self._get_ppolyrep()
        
        return self

    def __call__(self, tpoints):
        x_points = self.spline_x(tpoints)
        y_points = self.spline_y(tpoints)
        
        return x_points, y_points

    def eval(self, x):
        if hasattr(x, '__iter__'):
            t = np.array([self.spline_x.ppoly.evalinv(xi) for xi in x])
            return self.spline_y.ppoly(t)
        
        else:
            t = self.spline_x.ppoly.evalinv(x)
            return self.spline_y.ppoly(t)

    def get_polypoints(self, n):
        xpoints = self.spline_x.ppoly.eval_oninterval(n)
        ypoints = self.spline_y.ppoly.eval_oninterval(n)
        
        return xpoints, ypoints

    def _get_ppolyrep(self):
        self.spline_x.ppoly = PPolyInvertible.from_splinefunc(self.spline_x)
        self.spline_y.ppoly = PPolyInvertible.from_splinefunc(self.spline_y)

    def polyrep(self, tpoints):
        return self.spline_x.ppoly(tpoints), self.spline_y.ppoly(tpoints)

    def _build_splines(self, coeffs_x, coeffs_y):
        tck_x = self.knots_norm, coeffs_x, self.k
        tck_y = self.knots_norm, coeffs_y, self.k

        self.spline_x = si.UnivariateSpline._from_tck(tck_x)
        self.spline_y = si.UnivariateSpline._from_tck(tck_y)
        self.spline_x.tck = tck_x
        self.spline_y.tck = tck_y

    def _normalize_knotvector(self, knots=None, d=1.0):
        knots = knots or self.knots
        num_knots = len(knots)
        ka = (knots[-1] - knots[0]) / d
        knots_norm = np.empty(num_knots)
        for i in range(num_knots):
            knots_norm[i] = d - ((knots[-1] - knots[i])) / ka
        
        return knots_norm
