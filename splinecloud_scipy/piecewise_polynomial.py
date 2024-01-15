from functools import partial
from typing import Sequence
import math

import numpy as np
import scipy.interpolate as si
from scipy.optimize import brentq


Vector1D = Sequence[float]
Vector2D = Sequence[Sequence[float]]


class PPolyInvertible(si.PPoly):
    """Piecewise polynomial with ability to evaluate inverse dependency x(y)"""

    @classmethod
    def construct_fast(cls, c, x, extrapolate=None, axis=0):
        self = super(PPolyInvertible, cls).construct_fast(c, x, extrapolate=extrapolate, axis=axis)

        self.k = len(self.c) - 1
        self.powers = np.arange(self.k, -1, -1)
        self.intervals = self._form_intervals(self.x)
        return self

    @classmethod
    def from_splinefunc(cls, spline, extrapolate=None):
        self = cls.from_spline(spline.tck, extrapolate=extrapolate)
        self.project_intervals(spline)
        
        return self

    def eval_poly(self, t:float, coeffs:Vector1D, tbreak:float) -> float:
        poly = 0
        for c, p in zip(coeffs, self.powers):
            poly += c*(t - tbreak)**p
        
        return poly

    def project_intervals(self, spline):
        breaks = spline(self.x)
        self.pintervals = self._form_intervals(breaks)

    def _get_interval(self, xvalue:float, intervals:Vector2D) -> int:
        if xvalue < intervals[0][0]:
            return 0
        elif xvalue > intervals[-1][1]:
            return len(intervals) - 1

        i = 0
        for interval in intervals:
            if xvalue >= interval[0] and xvalue < interval[1]:
                return i
            else:
                i += 1

        ## patch end case
        if xvalue == interval[1]:
            return i-1

    def _form_intervals(self, breaks:Vector1D) -> np.ndarray:
        n = len(breaks) - 2*self.k - 1
        intervals = np.zeros((n, 2))
        i = self.k
        for interval in intervals:
            interval[0], interval[1] = breaks[i], breaks[i + 1]
            i += 1
        
        return intervals

    def _guess_error(self, t:float, coeffs:Vector1D, tbreak:float, xvalue:float) -> float:
        poly = self.eval_poly(t, coeffs, tbreak)
        error = poly - xvalue

        if abs(error) < 1e-12:
            return 0.0

        return error

    def evalinv(self, xvalue:int, extrapolate=False) -> float:
        n = self._get_interval(xvalue, self.pintervals)
        if n is None:
            return

        tmin, tmax = self.intervals[n]
        coeffs = self.c.T[n + self.k]
        tbreak = self.x[n + self.k]

        x_start = self.pintervals[0][0]
        x_end = self.pintervals[-1][1]

        if abs(xvalue - x_start) < 1e-12:
            return tmin

        elif abs(xvalue - x_end) < 1e-12:
            return tmax

        if xvalue < x_start: # extrapolate left
            if not extrapolate:
                return

            t_sol = self.solve(xvalue, extrapolate=True)
            if len(t_sol) == 0:
                return

            t_filter = [ts for ts in t_sol if ts < 0]
            if len(t_filter) == 0:
                return

            return max(t_filter)

        elif xvalue > x_end: # extrapolate right
            if not extrapolate:
                return

            t_sol = self.solve(xvalue, extrapolate=True)
            if len(t_sol) == 0:
                return

            t_filter = [ts for ts in t_sol if ts > 1]
            if len(t_filter) == 0:
                return

            return min(t_filter)

        xmin = self.eval_poly(tmin, coeffs, tbreak)
        xmax = self.eval_poly(tmax, coeffs, tbreak)

        if abs(xvalue - xmin) < 1e-12:
            return tmin

        elif abs(xvalue - xmax) < 1e-12:
            return tmax

        if (xvalue < xmin or xvalue > xmax):
            # xvalue may be out of interval if C0 continuity isn't kept
            return

        guess_error = partial(self._guess_error, coeffs=coeffs, tbreak=tbreak, xvalue=xvalue)
        try:
            t = brentq(guess_error, tmin, tmax)
        except Exception as ex:
            if not extrapolate: raise ex               
        else:
            return t
