import unittest
import responses
import numpy as np

from splinecloud_scipy import (
    load_subset, 
    load_spline, 
    load_spline_surface, 
    SPLINECLOUD_API_URL, 
    ParametricUnivariateSpline, 
    ParametricBivariateSpline
)

class APIClientTests(unittest.TestCase):

    @responses.activate
    def test_load_subset_decimal(self):
        subset_url = SPLINECLOUD_API_URL+'/subsets/sbt_1234567890abcdef'

        subset_response = {
            'table': {
                'column1': {'0': 1, '1': 2, '2': 3.0},
                'column2': {'0': 4.0, '1': 5, '2': 6},
            }
        }

        responses.add(responses.GET, subset_url, json=subset_response, status=200)
        columns, table = load_subset(subset_url)

        self.assertEqual(columns, ['column1', 'column2'])
        self.assertEqual(table.tolist(), [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])

    @responses.activate
    def test_load_subset_mixed(self):
        subset_url = SPLINECLOUD_API_URL+'/subsets/sbt_1234567890abcdef'
        # Define a mock subset response
        subset_response = {
            'table': {
                'column1': {'0': "attr1", '1': "attr2", '2': "attr3"},
                'column2': {'0': 4, '1': 5, '2': 6},
            }
        }

        responses.add(responses.GET, subset_url, json=subset_response, status=200)
        columns, table = load_subset(subset_url)

        self.assertEqual(columns, ['column1', 'column2'])
        self.assertEqual(table.tolist(), [["attr1", 4], ["attr2", 5], ["attr3", 6]])

    @responses.activate
    def test_load_spline(self):
        curve_url = SPLINECLOUD_API_URL+'/subsets/sbt_1234567890abcdef'

        knot_vector = [0, 0, 0, 0.34, 0.61, 0.85, 1, 1, 1]
        control_points = [
            [0.12, 0.13], 
            [2.67, 0.44],
            [7.91, 0.98],
            [12.55, 1.41],
            [15.96, 1.42],
            [17.48, 1.28]
        ]

        spline_response = { 
            "uid": "spl_K5t56P5bormJ", 
            "name": "Spline Curve 1", 
            "curve_type": "smooth-bspl", 
            "order": 2, 
            "spline": { 
                "c": control_points,
                "k": 2,
                "t": knot_vector, 
                "w": [1, 1, 1, 1, 1, 1],
                "labels": {
                    "xlabel": "X", 
                    "ylabel": "Y" 
                }
            },
            "scale_x": "Linear",
            "scale_y": "Linear"
        }

        responses.add(responses.GET, curve_url, json=spline_response, status=200)
        spline = load_spline(curve_url)

        self.assertIsInstance(spline, ParametricUnivariateSpline)
        self.assertEqual(spline.k, 2)
        self.assertEqual(spline.knots.tolist(), [0.0, 0.0, 0.0, 0.34, 0.61, 0.85, 1.0, 1.0, 1.0])
        cx, cy = zip(*control_points)
        self.assertEqual(spline.coeffs_x.tolist(), list(cx))
        self.assertEqual(spline.coeffs_y.tolist(), list(cy))

    @responses.activate
    def test_load_spline_surface(self):
        surface_url = SPLINECLOUD_API_URL + '/surfaces/srf_1234567890abcdef'
        
        tu = [0, 0, 0, 1, 1, 1]
        tv = [0, 0, 0, 1, 1, 1]
        cp = np.zeros((3, 3, 3)).tolist()
        w = np.ones((3, 3)).tolist()
        
        surface_response = {
            "uid": "srf_1234567890abcdef",
            "spline": {
                "tu": tu,
                "tv": tv,
                "cp": cp,
                "w": w,
                "ku": 2,
                "kv": 2
            },
            "surface_type": "lofted",
            "labels": {
                "xlabel": "alpha",
                "ylabel": "beta",
                "zlabel": "gamma"
            },
            "subset_uids": [
                "sbt_ABCD",
                "sbt_EFGH",
                "sbt_IJKL",
                "sbt_MNOP"
            ],
            "relation_uid": "lr2_qwerty",
            "relation_name": "Lofted Relation 1",
            "scale_x": "Linear",
            "scale_y": "Linear",
            "scale_z": "Linear"
        }
        
        responses.add(responses.GET, surface_url, json=surface_response, status=200)
        
        surface = load_spline_surface(surface_url)
        
        from splinecloud_scipy import SplineSurface
        self.assertIsInstance(surface, SplineSurface)
        self.assertIsInstance(surface, ParametricBivariateSpline)
        
        self.assertEqual(surface.x_label, "alpha")
        self.assertEqual(surface.y_label, "gamma") # y and z must be flipped for 'lofted' surfaces
        self.assertEqual(surface.z_label, "beta") # y and z must be flipped for 'lofted' surfaces
        
        self.assertEqual(surface.subset_uids, ["sbt_ABCD", "sbt_EFGH", "sbt_IJKL", "sbt_MNOP"])
        self.assertEqual(surface.relation_uid, "lr2_qwerty")


if __name__ == '__main__':
    unittest.main()
