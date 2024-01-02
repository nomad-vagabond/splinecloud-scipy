from functools import partial

import numpy as np
import scipy.interpolate as si
from scipy.optimize import brentq


class PPolyInvertible(si.PPoly):
    """Piecewise polynomial with ability to evaluate inverse dependency x(y)"""

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        self = super(PPolyInvertible, cls).construct_fast(
            c, x, extrapolate=extrapolate, axis=axis)
        
        self.k = len(self.c) - 1
        self.powers = np.arange(self.k, -1, -1)
        self.intervals = self._form_intervals(self.x)
        return self

    @classmethod
    def from_splinefunc(cls, spline):
        self = cls.from_spline(spline.tck)
        self.project_intervals(spline)
        
        return self

    def eval_poly(self, t, coeffs, tbreak):
        poly = 0
        for c, p in zip(coeffs, self.powers):
            poly += c*(t - tbreak)**p
        
        return poly

    def project_intervals(self, spline):
        breaks = spline(self.x)
        self.pintervals = self._form_intervals(breaks)

    def _get_interval(self, coord, intervals):
        i = 0
        for interval in intervals:
            if coord >= interval[0] and coord < interval[1]:
                return i
            else:
                i += 1

        ## patch end case
        if coord == interval[1]:
            return i-1

    def _form_intervals(self, breaks):
        n = len(breaks) - 2*self.k - 1
        intervals = np.zeros((n, 2))
        i = self.k
        for interval in intervals:
            interval[0], interval[1] = breaks[i], breaks[i + 1]
            i += 1
        
        return intervals

    def _guess_error(self, t, coeffs, tbreak, xvalue):
        poly = self.eval_poly(t, coeffs, tbreak)
        error = poly - xvalue

        if abs(error) < 1e-12:
            return 0.0

        return error

    def evalinv(self, x):
        n = self._get_interval(x, self.pintervals)
        if n is not None:
            tmin, tmax = self.intervals[n]

            coeffs = self.c.T[n + self.k]
            tbreak = self.x[n + self.k]

            xmin = self.eval_poly(tmin, coeffs, tbreak)
            xmax = self.eval_poly(tmax, coeffs, tbreak)

            if abs(x - xmin) < 1e-12:
                return tmin

            elif abs(x - xmax) < 1e-12:
                return tmax

            if x < xmin or x > xmax:
                # x may be out of interval if C0 continuity isn't kept
                return

            t = brentq(partial(self._guess_error, coeffs=coeffs, tbreak=tbreak, xvalue=x), tmin, tmax)

            return t
