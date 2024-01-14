import unittest

import numpy as np
from scipy.interpolate import splev
import responses

from splinecloud_scipy import ParametricUnivariateSpline


class TestParametricUnivariateSplineDegree1(unittest.TestCase):

    def test_init(self):
        t = [0.0, 0.0, 0.27, 0.31, 0.33, 0.45, 0.71, 1.0, 1.0]
        cx = [0.11, 1.23, 1.88, 2.39, 3.16, 4.32, 6.03]
        cy = [0.05, 0.13, 0.24, 0.31, 0.23, 0.15, 0.07]
        k = 1

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        self.assertIsInstance(spline, ParametricUnivariateSpline)
        self.assertEqual(spline.k, k)
        self.assertEqual(spline.knots.tolist(), t)
        self.assertEqual(spline.coeffs_x.tolist(), cx)
        self.assertEqual(spline.coeffs_y.tolist(), cy)

    def test_eval(self):
        t = [0.0, 0.0, 0.27, 0.31, 0.33, 0.45, 0.71, 1.0, 1.0]
        cx = [0.11, 1.23, 1.88, 2.39, 3.16, 4.32, 6.03]
        cy = [0.05, 0.13, 0.24, 0.31, 0.23, 0.15, 0.07]
        k = 1

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_eval_extrapolate_near(self):
        t = [0.0, 0.0, 0.27, 0.31, 0.33, 0.45, 0.71, 1.0, 1.0]
        cx = [0.11, 1.23, 1.88, 2.39, 3.16, 4.32, 6.03]
        cy = [0.05, 0.13, 0.24, 0.31, 0.23, 0.15, 0.07]
        k = 1

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 7, 100)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_eval_in_knots(self):
        t = [0.0, 0.0, 0.27, 0.31, 0.33, 0.45, 0.71, 1.0, 1.0]
        cx = [0.11, 1.23, 1.88, 2.39, 3.16, 4.32, 6.03]
        cy = [0.05, 0.13, 0.24, 0.31, 0.23, 0.15, 0.07]
        k = 1

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C0break_eval(self):
        t = [0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0]
        cx = [0.11, 1.23, 1.88, 3.02, 4.14, 6.03]
        cy = [0.06, 0.13, 0.25, 0.24, 0.16, 0.07]
        k = 1

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C0break_eval_extrapolate(self):
        t = [0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0]
        cx = [0.11, 1.23, 1.88, 3.02, 4.14, 6.03]
        cy = [0.06, 0.13, 0.25, 0.24, 0.16, 0.07]
        k = 1

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 7, 100)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        x = np.linspace(0.2, 5.9, 100)
        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        x = np.linspace(-1.0, 0.1, 10)
        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

        x = np.linspace(6.0, 7.0, 10)
        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_C0break_eval_in_knots(self):
        t = [0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0]
        cx = [0.11, 1.23, 1.88, 3.02, 4.14, 6.03]
        cy = [0.06, 0.13, 0.25, 0.24, 0.16, 0.07]
        k = 1

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))


class TestParametricUnivariateSplineDegree2(unittest.TestCase):

    def test_init(self):
        t = [0.0, 0.0, 0.0, 0.34, 0.61, 0.85, 1.0, 1.0, 1.0]
        cx = [0.12, 2.67, 7.91, 12.55, 15.96, 17.48]
        cy = [0.13, 0.44, 0.98, 1.41, 1.42, 1.28]
        k = 2

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        self.assertIsInstance(spline, ParametricUnivariateSpline)
        self.assertEqual(spline.k, k)
        self.assertEqual(spline.knots.tolist(), t)
        self.assertEqual(spline.coeffs_x.tolist(), cx)
        self.assertEqual(spline.coeffs_y.tolist(), cy)

    def test_eval(self):
        t = [0.0, 0.0, 0.0, 0.34, 0.61, 0.85, 1.0, 1.0, 1.0]
        cx = [0.12, 2.67, 7.91, 12.55, 15.96, 17.48]
        cy = [0.13, 0.44, 0.98, 1.41, 1.42, 1.28]
        k = 2

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_eval_extrapolate_near(self):
        t = [0.0, 0.0, 0.0, 0.34, 0.61, 0.85, 1.0, 1.0, 1.0]
        cx = [0.12, 2.67, 7.91, 12.55, 15.96, 17.48]
        cy = [0.13, 0.44, 0.98, 1.41, 1.42, 1.28]
        k = 2

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 19, 100)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.34, 0.61, 0.85, 1.0, 1.0, 1.0]
        cx = [0.12, 2.67, 7.91, 12.55, 15.96, 17.48]
        cy = [0.13, 0.44, 0.98, 1.41, 1.42, 1.28]
        k = 2

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C1break_eval(self):
        t = [0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0]
        cx = [0.11, 1.19, 1.68, 2.46, 3.28, 4.23, 6.03]
        cy = [0.05, 0.11, 0.21, 0.25, 0.22, 0.15, 0.07]
        k = 2

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C1break_eval_extrapolate_near(self):
        t = [0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0]
        cx = [0.11, 1.19, 1.68, 2.46, 3.28, 4.23, 6.03]
        cy = [0.05, 0.11, 0.21, 0.25, 0.22, 0.15, 0.07]
        k = 2

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 7, 100)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_C1break_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0]
        cx = [0.11, 1.19, 1.68, 2.46, 3.28, 4.23, 6.03]
        cy = [0.05, 0.11, 0.21, 0.25, 0.22, 0.15, 0.07]
        k = 2

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C0break_eval(self):
        t = [0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0]
        cx = [0.11, 1.1, 1.65, 1.98, 3.02, 3.66, 4.86, 6.03]
        cy = [0.05, 0.11, 0.18, 0.24, 0.24, 0.18, 0.12, 0.07]
        k = 2

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C0break_eval_extrapolate(self):
        t = [0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0]
        cx = [0.11, 1.1, 1.65, 1.98, 3.02, 3.66, 4.86, 6.03]
        cy = [0.05, 0.11, 0.18, 0.24, 0.24, 0.18, 0.12, 0.07]
        k = 2

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 7, 100)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        x = np.linspace(0.2, 5.9, 100)
        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        x = np.linspace(-1.0, 0.1, 10)
        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

        x = np.linspace(6.0, 7.0, 10)
        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_C0break_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0]
        cx = [0.11, 1.1, 1.65, 1.98, 3.02, 3.66, 4.86, 6.03]
        cy = [0.05, 0.11, 0.18, 0.24, 0.24, 0.18, 0.12, 0.07]
        k = 2

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))


class TestParametricUnivariateSplineDegree3(unittest.TestCase):

    def test_init(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.16, 0.26, 0.42, 0.58, 1.0, 1.0, 1.0, 1.0]
        cx = [0.02, 1.07, 2.78, 5.51, 8.19, 12.97, 16.74, 19.49]
        cy = [0.32, 0.43, 0.63, 0.94, 1.21, 1.48, 1.32, 1.32]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        self.assertIsInstance(spline, ParametricUnivariateSpline)
        self.assertEqual(spline.k, k)
        self.assertEqual(spline.knots.tolist(), t)
        self.assertEqual(spline.coeffs_x.tolist(), cx)
        self.assertEqual(spline.coeffs_y.tolist(), cy)

    def test_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.16, 0.26, 0.42, 0.58, 1.0, 1.0, 1.0, 1.0]
        cx = [0.02, 1.07, 2.78, 5.51, 8.19, 12.97, 16.74, 19.49]
        cy = [0.32, 0.43, 0.63, 0.94, 1.21, 1.48, 1.32, 1.32]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_eval_extrapolate_near(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.16, 0.26, 0.42, 0.58, 1.0, 1.0, 1.0, 1.0]
        cx = [0.02, 1.07, 2.78, 5.51, 8.19, 12.97, 16.74, 19.49]
        cy = [0.32, 0.43, 0.63, 0.94, 1.21, 1.48, 1.32, 1.32]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 21, 100)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.16, 0.26, 0.42, 0.58, 1.0, 1.0, 1.0, 1.0]
        cx = [0.02, 1.07, 2.78, 5.51, 8.19, 12.97, 16.74, 19.49]
        cy = [0.32, 0.43, 0.63, 0.94, 1.21, 1.48, 1.32, 1.32]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C2break_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 3.0, 3.88, 4.94, 6.02]
        cy = [0.05, 0.07, 0.12, 0.23, 0.23, 0.17, 0.1, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C2break_eval_extrapolate_near(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 3.0, 3.88, 4.94, 6.02]
        cy = [0.05, 0.07, 0.12, 0.23, 0.23, 0.17, 0.1, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 7, 100)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_C2break_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 3.0, 3.88, 4.94, 6.02]
        cy = [0.05, 0.07, 0.12, 0.23, 0.23, 0.17, 0.1, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C1break_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 2.39, 3.15, 3.88, 4.94, 6.03]
        cy = [0.05, 0.07, 0.12, 0.23, 0.26, 0.23, 0.17, 0.01, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C1break_eval_extrapolate_near(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 2.39, 3.15, 3.88, 4.94, 6.03]
        cy = [0.05, 0.07, 0.12, 0.23, 0.26, 0.23, 0.17, 0.01, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 7, 100)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_C1break_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 2.39, 3.15, 3.88, 4.94, 6.03]
        cy = [0.05, 0.07, 0.12, 0.23, 0.26, 0.23, 0.17, 0.01, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C0break_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 2.1, 2.77, 3.15, 3.88, 4.94, 6.03]
        cy = [0.05, 0.07, 0.12, 0.23, 0.24, 0.24, 0.23, 0.17, 0.1, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 141)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C0break_eval_extrapolate(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 2.1, 2.77, 3.15, 3.88, 4.94, 6.03]
        cy = [0.05, 0.07, 0.12, 0.23, 0.24, 0.24, 0.23, 0.17, 0.1, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 7, 100)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        x = np.linspace(0.2, 5.9, 100)
        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        x = np.linspace(-1.0, 0.1, 10)
        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

        x = np.linspace(6.0, 7.0, 10)
        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_C0break_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 2.1, 2.77, 3.15, 3.88, 4.94, 6.03]
        cy = [0.05, 0.07, 0.12, 0.23, 0.24, 0.24, 0.23, 0.17, 0.1, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))


class TestParametricUnivariateSplineDegree4(unittest.TestCase):

    def test_init(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.28, 0.56, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [1.11, 2.39, 4.98, 9.61, 14.24, 17.6, 19.64]
        cy = [0.3, 0.41, 0.76, 1.33, 1.12, 1.17, 1.15]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        self.assertIsInstance(spline, ParametricUnivariateSpline)
        self.assertEqual(spline.k, k)
        self.assertEqual(spline.knots.tolist(), t)
        self.assertEqual(spline.coeffs_x.tolist(), cx)
        self.assertEqual(spline.coeffs_y.tolist(), cy)

    def test_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.28, 0.56, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [1.11, 2.39, 4.98, 9.61, 14.24, 17.6, 19.64]
        cy = [0.3, 0.41, 0.76, 1.33, 1.12, 1.17, 1.15]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_eval_extrapolate_near(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.28, 0.56, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [1.11, 2.39, 4.98, 9.61, 14.24, 17.6, 19.64]
        cy = [0.3, 0.41, 0.76, 1.33, 1.12, 1.17, 1.15]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 21, 100)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.28, 0.56, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [1.11, 2.39, 4.98, 9.61, 14.24, 17.6, 19.64]
        cy = [0.3, 0.41, 0.76, 1.33, 1.12, 1.17, 1.15]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C3break_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.73, 1.48, 1.88, 2.58, 3.5, 4.54, 5.6, 6.03]
        cy = [0.05, 0.09, 0.15, 0.24, 0.27, 0.2, 0.14, 0.09, 0.07]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 141)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C3break_eval_extrapolate_near(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.73, 1.48, 1.88, 2.58, 3.5, 4.54, 5.6, 6.03]
        cy = [0.05, 0.09, 0.15, 0.24, 0.27, 0.2, 0.14, 0.09, 0.07]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 6, 30)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_C3break_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.73, 1.48, 1.88, 2.58, 3.5, 4.54, 5.6, 6.03]
        cy = [0.05, 0.09, 0.15, 0.24, 0.27, 0.2, 0.14, 0.09, 0.07]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C2break_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.72, 1.25, 1.7, 2.03, 2.92, 3.67, 4.42, 5.13, 6.03]
        cy = [0.05, 0.08, 0.12, 0.18, 0.26, 0.26, 0.18, 0.14, 0.11, 0.07]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 141)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C2break_eval_extrapolate_near(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.72, 1.25, 1.7, 2.03, 2.92, 3.67, 4.42, 5.13, 6.03]
        cy = [0.05, 0.08, 0.12, 0.18, 0.26, 0.26, 0.18, 0.14, 0.11, 0.07]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 7, 100)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_C2break_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.72, 1.25, 1.7, 2.03, 2.92, 3.67, 4.42, 5.13, 6.03]
        cy = [0.05, 0.08, 0.12, 0.18, 0.26, 0.26, 0.18, 0.14, 0.11, 0.07]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C1break_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.76, 1.28, 1.68, 1.98, 2.39, 3.01, 3.56, 4.39, 5.17, 6.03]
        cy = [0.05, 0.09, 0.14, 0.18, 0.24, 0.31, 0.24, 0.19, 0.14, 0.11, 0.07]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 141)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C1break_eval_extrapolate_near(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.76, 1.28, 1.68, 1.98, 2.39, 3.01, 3.56, 4.39, 5.17, 6.03]
        cy = [0.05, 0.09, 0.14, 0.18, 0.24, 0.31, 0.24, 0.19, 0.14, 0.11, 0.07]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 7, 100)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_C1break_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.76, 1.28, 1.68, 1.98, 2.39, 3.01, 3.56, 4.39, 5.17, 6.03]
        cy = [0.05, 0.09, 0.14, 0.18, 0.24, 0.31, 0.24, 0.19, 0.14, 0.11, 0.07]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C0break_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.56, 1.11, 1.52, 1.74, 1.94, 2.82, 3.14, 3.6, 4.2, 4.96, 6.03]
        cy = [0.05, 0.07, 0.12, 0.17, 0.2, 0.26, 0.26, 0.22, 0.19, 0.15, 0.12, 0.07]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        knots = np.linspace(0.0, 1.0, 141)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_C0break_eval_extrapolate(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.56, 1.11, 1.52, 1.74, 1.94, 2.82, 3.14, 3.6, 4.2, 4.96, 6.03]
        cy = [0.05, 0.07, 0.12, 0.17, 0.2, 0.26, 0.26, 0.22, 0.19, 0.15, 0.12, 0.07]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        
        x = np.linspace(-1, 7, 100)
        y = spline.eval(x)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        x = np.linspace(0.2, 5.9, 100)
        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertTrue(np.isnan(y).any())

        x = np.linspace(-1.0, 0.1, 10)
        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

        x = np.linspace(6.0, 7.0, 10)
        y = spline.eval(x, extrapolate=True)

        self.assertEqual(len(y), len(x))
        self.assertFalse(np.isnan(y).any())

    def test_C0break_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5, 0.75, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.56, 1.11, 1.52, 1.74, 1.94, 2.82, 3.14, 3.6, 4.2, 4.96, 6.03]
        cy = [0.05, 0.07, 0.12, 0.17, 0.2, 0.26, 0.26, 0.22, 0.19, 0.15, 0.12, 0.07]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))


if __name__ == '__main__':
    unittest.main()

