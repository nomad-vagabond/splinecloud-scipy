import unittest

import numpy as np
from scipy.interpolate import splev
import responses

from splinecloud_scipy import ParametricUnivariateSpline


class TestParametricUnivariateSpline(unittest.TestCase):

    def test_from_tcck(self):
        t = [0.0, 0.0, 0.0, 0.34, 0.61, 0.85, 1.0, 1.0, 1.0]
        cx = [0.12, 2.67, 7.91, 12.55, 15.96, 17.48]
        cy = [0.13, 0.44, 0.98, 1.41, 1.42, 1.28]
        k = 2

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline.from_tcck(tcck)
        
        self.assertIsInstance(spline, ParametricUnivariateSpline)
        self.assertEqual(spline.k, k)
        self.assertEqual(spline.knots.tolist(), t)
        self.assertEqual(spline.coeffs_x.tolist(), cx)
        self.assertEqual(spline.coeffs_y.tolist(), cy)

    def test_spline_k2_eval(self):
        t = [0.0, 0.0, 0.0, 0.34, 0.61, 0.85, 1.0, 1.0, 1.0]
        cx = [0.12, 2.67, 7.91, 12.55, 15.96, 17.48]
        cy = [0.13, 0.44, 0.98, 1.41, 1.42, 1.28]
        k = 2

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline.from_tcck(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_spline_k2_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.34, 0.61, 0.85, 1.0, 1.0, 1.0]
        cx = [0.12, 2.67, 7.91, 12.55, 15.96, 17.48]
        cy = [0.13, 0.44, 0.98, 1.41, 1.42, 1.28]
        k = 2

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline.from_tcck(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_spline_k3_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.16, 0.26, 0.42, 0.58, 1.0, 1.0, 1.0, 1.0]
        cx = [0.02, 1.07, 2.78, 5.51, 8.19, 12.97, 16.74, 19.49]
        cy = [0.32, 0.43, 0.63, 0.94, 1.21, 1.48, 1.32, 1.32]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline.from_tcck(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_spline_k3_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.16, 0.26, 0.42, 0.58, 1.0, 1.0, 1.0, 1.0]
        cx = [0.02, 1.07, 2.78, 5.51, 8.19, 12.97, 16.74, 19.49]
        cy = [0.32, 0.43, 0.63, 0.94, 1.21, 1.48, 1.32, 1.32]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline.from_tcck(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_spline_k4_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.28, 0.56, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [1.11, 2.39, 4.98, 9.61, 14.24, 17.6, 19.64]
        cy = [0.3, 0.41, 0.76, 1.33, 1.12, 1.17, 1.15]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline.from_tcck(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_spline_k4_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.28, 0.56, 1.0, 1.0, 1.0, 1.0, 1.0]
        cx = [1.11, 2.39, 4.98, 9.61, 14.24, 17.6, 19.64]
        cy = [0.3, 0.41, 0.76, 1.33, 1.12, 1.17, 1.15]
        k = 4

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline.from_tcck(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_spline_k3_C2break_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 3.0, 3.88, 4.94, 6.02]
        cy = [0.05, 0.07, 0.12, 0.23, 0.23, 0.17, 0.1, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline.from_tcck(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_spline_k3_C2break_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 3.0, 3.88, 4.94, 6.02]
        cy = [0.05, 0.07, 0.12, 0.23, 0.23, 0.17, 0.1, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline.from_tcck(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    
    def test_spline_k3_C1break_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 2.39, 3.15, 3.88, 4.94, 6.03]
        cy = [0.05, 0.07, 0.12, 0.23, 0.26, 0.23, 0.17, 0.01, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline.from_tcck(tcck)
        
        knots = np.linspace(0.0, 1.0, 11)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_spline_k3_C1break_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 2.39, 3.15, 3.88, 4.94, 6.03]
        cy = [0.05, 0.07, 0.12, 0.23, 0.26, 0.23, 0.17, 0.01, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline.from_tcck(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))


    def test_spline_k3_C0break_eval(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 2.1, 2.77, 3.15, 3.88, 4.94, 6.03]
        cy = [0.05, 0.07, 0.12, 0.23, 0.24, 0.24, 0.23, 0.17, 0.1, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline.from_tcck(tcck)
        
        knots = np.linspace(0.0, 1.0, 141)

        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))

    def test_spline_k3_C0break_eval_in_knots(self):
        t = [0.0, 0.0, 0.0, 0.0, 0.27, 0.33, 0.33, 0.33, 0.33, 0.45, 1.0, 1.0, 1.0, 1.0]
        cx = [0.11, 0.64, 1.19, 1.81, 2.1, 2.77, 3.15, 3.88, 4.94, 6.03]
        cy = [0.05, 0.07, 0.12, 0.23, 0.24, 0.24, 0.23, 0.17, 0.1, 0.08]
        k = 3

        tcck = t, cx, cy, k
        spline = ParametricUnivariateSpline.from_tcck(tcck)
        knots = t[k:-k]
        x, y = splev(knots, (np.array(t), np.array([cx, cy]), k))
        y_ = spline.eval(x)

        self.assertEqual(len(y), len(y_))
        self.assertTrue(np.allclose(y, y_))


if __name__ == '__main__':
    unittest.main()

