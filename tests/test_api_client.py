import unittest
import responses
import numpy as np

from splinecloud_scipy import (
    load_subset, 
    load_spline, 
    load_spline_surface, 
    SPLINECLOUD_API_URL, 
    ParametricUnivariateSpline, 
    ParametricBivariateSpline,
    SplineSurface,
    ExplicitUnivariateSpline,
    ExplicitSplineSurface,
    NdBsplExplicit,
    load_model,
    melt_grid,
    stack_subsets,
    role_variables,
    load_spline_hypersurface,
    load_multivariate_relation,
    SplineHyperSurface,
    MultivariateRelation,
)
from responses import matchers


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
    def test_load_spline_explicit_native(self):
        curve_url = SPLINECLOUD_API_URL + '/curves/spl_K5t56P5bormJ/'
        spline_response = {
            "uid": "spl_K5t56P5bormJ", "curve_type": "smooth-bspl", "form": "explicit",
            "degree": 1, "order": 1,
            "spline": {"v": 1, "nx": 1, "ny": 1, "k": [1], "t": [[0.0, 0.0, 1.0, 1.0]],
                       "shape": [2], "c": [0.0, 2.0], "domain": [[0.0, 1.0]],
                       "scales": ["Linear", "Linear"], "extrapolation": "none"},
            "subset_uid": "sbt_1234567890abcdef", "scale_x": "Linear", "scale_y": "Linear"
        }
        ## the client must ask for the native representation
        responses.add(responses.GET, curve_url, json=spline_response, status=200,
                      match=[matchers.query_param_matcher({"spline": "native"})])
        spline = load_spline("spl_K5t56P5bormJ")

        self.assertIsInstance(spline, ExplicitUnivariateSpline)
        self.assertEqual(spline.form, "explicit")
        self.assertAlmostEqual(spline.eval(0.25), 0.5)
        self.assertTrue(callable(spline.load_data))

    @responses.activate
    def test_load_spline_explicit_from_legacy_server(self):
        ## a server without spline=native support returns the Greville embedding
        curve_url = SPLINECLOUD_API_URL + '/curves/spl_K5t56P5bormJ/'
        spline_response = {
            "uid": "spl_K5t56P5bormJ", "form": "explicit",
            "spline": {"t": [0, 0, 1, 1], "c": [[0.0, 0.0], [1.0, 2.0]], "k": 1, "w": [1, 1]},
            "subset_uid": "sbt_1234567890abcdef", "scale_x": "Linear", "scale_y": "Linear"
        }
        responses.add(responses.GET, curve_url, json=spline_response, status=200)
        spline = load_spline("spl_K5t56P5bormJ")

        self.assertIsInstance(spline, ParametricUnivariateSpline)
        self.assertAlmostEqual(spline.eval(0.25), 0.5)

    @responses.activate
    def test_load_spline_unknown_form(self):
        curve_url = SPLINECLOUD_API_URL + '/curves/spl_K5t56P5bormJ/'
        responses.add(responses.GET, curve_url, json={"form": "fancy", "spline": {}}, status=200)
        with self.assertRaises(ValueError):
            load_spline("spl_K5t56P5bormJ")

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
                "v": 1, "nx": 2, "ny": 1, "form": "parametric",
                "tu": tu,
                "tv": tv,
                "cp": cp,
                "w": w,
                "ku": 2,
                "kv": 2,
                "scales": ["Linear", "Linear", "Linear"]
            },
            "surface_type": "lofted",
            "labels": {
                "x1": "alpha",
                "y": "beta",
                "x2": "gamma"
            },
            "subset_uids": [
                "sbt_ABCD",
                "sbt_EFGH",
                "sbt_IJKL",
                "sbt_MNOP"
            ],
            "relation_uid": "lr2_qwerty",
            "relation_name": "Lofted Relation 1",
            "scale_x1": "Linear",
            "scale_y": "Linear",
            "scale_x2": "Linear"
        }
        
        responses.add(responses.GET, surface_url, json=surface_response, status=200)
        
        surface = load_spline_surface(surface_url)
        
        self.assertIsInstance(surface, SplineSurface)
        self.assertIsInstance(surface, ParametricBivariateSpline)
        
        self.assertEqual(surface.x1_label, "alpha")
        self.assertEqual(surface.y_label, "beta")
        self.assertEqual(surface.x2_label, "gamma")
        
        self.assertEqual(surface.subset_uids, ["sbt_ABCD", "sbt_EFGH", "sbt_IJKL", "sbt_MNOP"])
        self.assertEqual(surface.relation_uid, "lr2_qwerty")


def lofted_level_response():
    """A lofted surface whose relation models the curves' level (response_axis "z"): its
    contour lines y(x) at levels z of z = 1 + x + 2 y, the net in role order [x, y, z]
    (bilinear, as a loft between two curves: exact on x in [0, 4], y in [0, 2])."""
    cp = [[[0.0, 0.0, 1.0], [0.0, 2.0, 5.0]], [[4.0, 0.0, 5.0], [4.0, 2.0, 9.0]]]
    return {
        "uid": "srf_lofted000001", "name": "Loft Surface 1", "form": "parametric",
        "surface_type": "lofted",
        "spline": {"v": 1, "nx": 2, "ny": 1, "form": "parametric", "tu": [0, 0, 1, 1],
                   "tv": [0, 0, 1, 1], "ku": 1, "kv": 1, "cp": cp, "w": [[1, 1], [1, 1]],
                   "scales": ["Linear", "Linear", "Linear"]},
        "labels": {"x1": "Easting", "x2": "Northing", "y": "Altitude"},
        "response_axis": "z",
        "variables": {"x": "Easting", "y": "Northing", "z": "Altitude"},
        "subset_uids": ["sbt_level0000001", "sbt_level0000002"], "x2_values": [3.0, 5.0],
        "curve_uids": [], "relation_uid": "lr2_relation0001", "relation_name": "Terrain",
        "scale_x1": "Linear", "scale_x2": "Linear", "scale_y": "Linear",
    }


class LoftedResponseAxisTests(unittest.TestCase):
    """Lofted surfaces whose relation models another of its curves' variables (response
    axis): evaluation and data follow the roles."""

    URL = SPLINECLOUD_API_URL + '/surfaces/srf_lofted000001/'

    def _load(self):
        responses.add(responses.GET, self.URL, json=lofted_level_response(), status=200)
        return load_spline_surface("srf_lofted000001")

    def test_role_variables(self):
        self.assertEqual(role_variables("y"), {"x1": "x", "x2": "z", "y": "y"})
        self.assertEqual(role_variables("z"), {"x1": "x", "x2": "y", "y": "z"})
        self.assertEqual(role_variables("x"), {"x1": "y", "x2": "z", "y": "x"})
        with self.assertRaises(ValueError):
            role_variables("w")

    @responses.activate
    def test_evaluates_the_response(self):
        surface = self._load()
        self.assertEqual(surface.response_axis, "z")
        self.assertEqual(surface.variables["z"], "Altitude")
        self.assertEqual(surface.labels, ["Easting", "Northing", "Altitude"])
        ## altitude at (easting, northing): the inputs are the first two coordinates
        self.assertAlmostEqual(surface.eval(1.5, 0.5), 1 + 1.5 + 2 * 0.5, places=8)
        X1, X2, Y = surface.eval([1.0, 3.0], [0.25, 1.75])
        np.testing.assert_allclose(Y, 1 + X1 + 2 * X2, atol=1e-8)

    @responses.activate
    def test_load_data_in_roles(self):
        surface = self._load()
        for uid, rows in (("sbt_level0000001", {"x": {"0": 0.0, "1": 2.0},
                                                "y": {"0": 1.0, "1": 0.0}}),
                          ("sbt_level0000002", {"x": {"0": 0.0}, "y": {"0": 2.0}})):
            responses.add(responses.GET, SPLINECLOUD_API_URL + '/subsets/{}/'.format(uid),
                          status=200, json={"table": rows})
        labels, data = surface.load_data()
        self.assertEqual(labels, ["Easting", "Northing", "Altitude"])
        ## [x, y, level]: the level last, as the response
        np.testing.assert_allclose(data, [[0.0, 1.0, 3.0], [2.0, 0.0, 3.0], [0.0, 2.0, 5.0]])


def explicit_surface_response(uid="srf_fitted000001", relation_uid="dr2_relation0001"):
    """A fitted surface as /api/surfaces/<uid>/ returns it: y = 1 + x1 + 2 x2 exactly
    (bilinear B-spline on x1 in [0, 4], x2 in [0, 2])."""
    payload = {"v": 1, "nx": 2, "ny": 1, "k": [1, 1], "t": [[0, 0, 4, 4], [0, 0, 2, 2]],
               "shape": [2, 2], "c": [1.0, 5.0, 5.0, 9.0], "domain": [[0, 4], [0, 2]],
               "scales": ["Linear", "Linear", "Linear"], "extrapolation": "none"}
    return {
        "uid": uid, "name": "Spline Surface 1", "form": "explicit", "spline": payload,
        "surface_type": "fitted", "labels": {"x1": "Re", "x2": "Mach", "y": "Cd"},
        "subset_uids": ["sbt_subset000001"], "x2_values": [], "curve_uids": [],
        "relation_uid": relation_uid, "relation_name": "Bivariate Relation 1",
        "data_relation_uid": relation_uid, "scale_x1": "Linear", "scale_x2": "Linear",
        "scale_y": "Linear", "method": "rmtb", "fit_params": {"degree": [1, 1]},
        "accuracy": {"r2": 1.0, "rmse": 0.0, "n": 4}, "fit_status": "ok",
    }


class ExplicitSurfaceClientTests(unittest.TestCase):

    SURFACE_URL = SPLINECLOUD_API_URL + '/surfaces/srf_fitted000001/'

    def _add_surface(self, response=None):
        responses.add(responses.GET, self.SURFACE_URL, json=response or explicit_surface_response(),
                      status=200, match=[matchers.query_param_matcher({"spline": "native"})])

    @responses.activate
    def test_dispatches_on_form(self):
        self._add_surface()
        surface = load_spline_surface("srf_fitted000001")
        self.assertIsInstance(surface, ExplicitSplineSurface)
        self.assertIsInstance(surface, NdBsplExplicit)
        self.assertEqual((surface.form, surface.method, surface.fit_status),
                         ("explicit", "rmtb", "ok"))
        self.assertEqual(surface.labels, ["Re", "Mach", "Cd"])
        self.assertEqual(surface.accuracy["n"], 4)
        self.assertEqual(surface.relation_uid, "dr2_relation0001")

    @responses.activate
    def test_unknown_form_raises(self):
        self._add_surface({**explicit_surface_response(), "form": "implicit"})
        with self.assertRaises(ValueError):
            load_spline_surface("srf_fitted000001")

    @responses.activate
    def test_legacy_eval_semantics(self):
        self._add_surface()
        surface = load_spline_surface("srf_fitted000001")
        ## scalar -> float
        self.assertAlmostEqual(surface.eval(1.0, 0.5), 3.0)
        y, (g1, g2) = surface.eval(1.0, 0.5, compute_gradients=True)
        self.assertAlmostEqual(y, 3.0)
        self.assertAlmostEqual(g1, 1.0)
        self.assertAlmostEqual(g2, 2.0)
        ## arrays -> outer-product grid X1, X2, Y
        X1, X2, Y = surface.eval([0.0, 2.0, 4.0], [0.0, 1.0])
        self.assertEqual(Y.shape, (3, 2))
        np.testing.assert_allclose(Y, 1 + X1 + 2 * X2)
        X1, X2, Y, dY1, dY2 = surface.eval([1.0, 2.0], [0.5], compute_gradients=True)
        np.testing.assert_allclose(dY1, 1.0)
        np.testing.assert_allclose(dY2, 2.0)
        ## legacy parametric-evaluator options are accepted
        self.assertAlmostEqual(surface.eval(1.0, 0.5, tol=1e-8, max_iter=10), 3.0)

    @responses.activate
    def test_out_of_domain(self):
        self._add_surface()
        surface = load_spline_surface("srf_fitted000001")
        self.assertTrue(np.isnan(surface.eval(5.0, 1.0)))
        ## extrapolate=True is linear extrapolation (exact for this plane)
        self.assertAlmostEqual(surface.eval(5.0, 1.0, extrapolate=True), 8.0)
        self.assertAlmostEqual(surface.eval(5.0, 1.0, extrapolate="clamp"), 7.0)

    @responses.activate
    def test_explicit_api_on_surface(self):
        self._add_surface()
        surface = load_spline_surface("srf_fitted000001")
        np.testing.assert_allclose(surface(np.array([1.0, 2.0]), np.array([0.5, 1.5])),
                                   [3.0, 6.0])
        self.assertEqual(surface.eval_grid([0, 1, 2], [0, 1]).shape, (3, 2))
        self.assertAlmostEqual(surface.slice({1: 1.0}).eval(2.0), 5.0)
        self.assertAlmostEqual(surface.to_parametric().eval_point(2.0, 1.0), 5.0, places=8)

    @responses.activate
    def test_load_data_long_form(self):
        self._add_surface()
        relation_url = SPLINECLOUD_API_URL + '/relations/bivariate/fitted/dr2_relation0001/'
        responses.add(responses.GET, relation_url, status=200, json={
            "uid": "dr2_relation0001", "data_format": "vector",
            "data_mapping": {"x1_idx": 2, "x2_idx": 0, "y_idx": 1},
            "excluded_points": [False, True, False],
        })
        subset_url = SPLINECLOUD_API_URL + '/subsets/sbt_subset000001/'
        responses.add(responses.GET, subset_url, status=200, json={"table": {
            "mach": {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4},
            "cd": {"0": 1.0, "1": 2.0, "2": None, "3": 4.0},     ## row 2: a gap
            "re": {"0": 10, "1": 20, "2": 30, "3": 40},
        }})
        surface = load_spline_surface("srf_fitted000001")
        labels, data = surface.load_data()
        self.assertEqual(labels, ["Re", "Mach", "Cd"])
        np.testing.assert_allclose(data, [[10, 0.1, 1.0], [20, 0.2, 2.0], [40, 0.4, 4.0]])
        np.testing.assert_array_equal(surface.excluded_points, [False, True, False])

    def _add_grid_relation(self, mapping, table, excluded=None):
        self._add_surface()
        relation_url = SPLINECLOUD_API_URL + '/relations/bivariate/fitted/dr2_relation0001/'
        responses.add(responses.GET, relation_url, status=200, json={
            "uid": "dr2_relation0001", "data_format": "matrix", "data_mapping": mapping,
            "excluded_points": excluded,
        })
        subset_url = SPLINECLOUD_API_URL + '/subsets/sbt_subset000001/'
        responses.add(responses.GET, subset_url, status=200, json={"table": table})

    @responses.activate
    def test_load_data_grid_headers(self):
        ## x1 down the first column, x2 in the headers, one empty cell
        self._add_grid_relation(
            {"x1_col": 0, "x2_source": "header", "x2_row": None, "y_cols": None},
            {"Re \\ Mach": {"0": 10, "1": 20},
             "0.6": {"0": 1.0, "1": None},
             "0.8": {"0": 3.0, "1": 4.0}},
            excluded=[False, True, False])
        surface = load_spline_surface("srf_fitted000001")
        labels, data = surface.load_data()
        self.assertEqual(surface.data_format, "matrix")
        ## row by row, then column by column; the empty cell is skipped
        np.testing.assert_allclose(data, [[10, 0.6, 1.0], [10, 0.8, 3.0], [20, 0.8, 4.0]])
        np.testing.assert_array_equal(surface.excluded_points, [False, True, False])

    @responses.activate
    def test_load_data_grid_row(self):
        ## x2 in the first row (empty top-left cell), which isn't a data row; y_cols subset
        self._add_grid_relation(
            {"x1_col": 0, "x2_source": "row", "x2_row": 0, "y_cols": [2, 3]},
            {"A": {"0": None, "1": 1.0, "2": 2.0, "3": None},
             "B": {"0": 0.1, "1": 5.0, "2": 6.0, "3": 7.0},
             "C": {"0": 0.2, "1": 8.0, "2": 9.0, "3": 9.5},
             "D": {"0": 0.3, "1": 10.0, "2": 11.0, "3": 12.0}})
        surface = load_spline_surface("srf_fitted000001")
        labels, data = surface.load_data()
        ## row 3 has no x1 value: none of its cells is a point
        np.testing.assert_allclose(data, [[1, 0.2, 8], [1, 0.3, 10], [2, 0.2, 9], [2, 0.3, 11]])
        self.assertIsNone(surface.excluded_points)

    @responses.activate
    def test_load_data_stacked(self):
        ## two contour lines (x, y) at z = 2 and z = 1, a third subset without z; the surface
        ## lists the first two, the relation's sources list them in z order
        self._add_surface({**explicit_surface_response(),
                           "subset_uids": ["sbt_subset000002", "sbt_subset000001"]})
        relation_url = SPLINECLOUD_API_URL + '/relations/bivariate/fitted/dr2_relation0001/'
        sources = [
            {"subset_uid": "sbt_subset000002", "z": 1.0, "n_points": 2},
            {"subset_uid": "sbt_subset000001", "z": 2.0, "n_points": 1},
            {"subset_uid": "sbt_subset000003", "z": None, "n_points": 1},
        ]
        responses.add(responses.GET, relation_url, status=200, json={
            "uid": "dr2_relation0001", "data_format": "stacked", "subset": None,
            "data_mapping": {"x1_idx": 0, "x2_idx": 1, "y_idx": "z"}, "sources": sources,
            "excluded_points": [False, True, False],
        })
        tables = {
            "sbt_subset000001": {"x": {"0": 0.5, "1": 0.7}, "y": {"0": 5.0, "1": None}},
            "sbt_subset000002": {"x": {"0": 0.1, "1": 0.2}, "y": {"0": 1.0, "1": 2.0}},
            "sbt_subset000003": {"x": {"0": 9.0}, "y": {"0": 9.0}},
        }
        for uid, table in tables.items():
            responses.add(responses.GET, SPLINECLOUD_API_URL + '/subsets/{}/'.format(uid),
                          status=200, json={"table": table})
        surface = load_spline_surface("srf_fitted000001")
        labels, data = surface.load_data()
        self.assertEqual(surface.data_format, "stacked")
        self.assertEqual(surface.sources, sources)
        ## subset by subset in sources order; the row with a gap and the subset without z out
        np.testing.assert_allclose(data, [[0.1, 1.0, 1.0], [0.2, 2.0, 1.0], [0.5, 5.0, 2.0]])
        np.testing.assert_array_equal(surface.excluded_points, [False, True, False])

    def test_stack_subsets_z_role(self):
        ## z as the first input: x1 = z, x2 = column 0, y = column 1
        tables = {"sbt_a": (["x", "y"], np.array([[1.0, 10.0], [2.0, 20.0]], dtype=object))}
        data = stack_subsets([{"subset_uid": "sbt_a", "z": 7.0}], tables,
                             {"x1_idx": "z", "x2_idx": 0, "y_idx": 1})
        np.testing.assert_allclose(data, [[7.0, 1.0, 10.0], [7.0, 2.0, 20.0]])
        with self.assertRaises(ValueError):
            stack_subsets([], tables, {"x1_idx": 0, "x2_idx": 1, "y_idx": 1})

    def test_melt_grid_errors(self):
        table = np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 5.0]])
        with self.assertRaises(ValueError):    ## a header that isn't an x2 value
            melt_grid(["x", "0.5", "alpha"], table, {"x1_col": 0, "x2_source": "header"})
        bad = np.array([[1.0, "n/a", 3.0], [2.0, 4.0, 5.0]], dtype=object)
        with self.assertRaises(ValueError):    ## a cell that isn't a number
            melt_grid(["x", "0.5", "0.7"], bad, {"x1_col": 0, "x2_source": "header"})
        np.testing.assert_allclose(
            melt_grid(["x", "0.5", "0.7"], table, {"x1_col": 0, "x2_source": "header"}),
            [[1, 0.5, 2], [1, 0.7, 3], [2, 0.5, 4], [2, 0.7, 5]])
        ## the x2 row is never a data row, even with a value in its x1 cell
        np.testing.assert_allclose(
            melt_grid(["A", "B", "C"], table, {"x1_col": 0, "x2_source": "row", "x2_row": 0}),
            [[2, 2, 4], [2, 3, 5]])

    def test_melt_grid_labeled_values(self):
        ## "Re=1e5" headers and "alpha: -4" x1 cells, as on SplineCloud
        table = np.array([["alpha: -4", 1.0, 2.0], ["alpha: 0", 3.0, 4.0]], dtype=object)
        np.testing.assert_allclose(
            melt_grid(["", "Re=1e5", "re=3e5"], table, {"x1_col": 0, "x2_source": "header"}),
            [[-4, 1e5, 1], [-4, 3e5, 2], [0, 1e5, 3], [0, 3e5, 4]])
        with self.assertRaises(ValueError):    ## two labels on one axis
            melt_grid(["", "Re=1e5", "Mach=0.6"], table, {"x1_col": 0, "x2_source": "header"})
        with self.assertRaises(ValueError):    ## "12:30" isn't a labelled value
            melt_grid(["", "12:30", "Re=3e5"], table, {"x1_col": 0, "x2_source": "header"})

    @responses.activate
    def test_load_data_grid_stale_mask(self):
        ## a mask that doesn't match the points (the data changed) is not applied
        self._add_grid_relation(
            {"x1_col": 0, "x2_source": "header", "x2_row": None, "y_cols": None},
            {"x": {"0": 1, "1": 2}, "0.5": {"0": 1.0, "1": 2.0}, "0.7": {"0": 3.0, "1": 4.0}},
            excluded=[True, False])
        surface = load_spline_surface("srf_fitted000001")
        self.assertEqual(len(surface.load_data()[1]), 4)
        self.assertIsNone(surface.excluded_points)

    @responses.activate
    def test_load_dispatches_on_prefix(self):
        self._add_surface()
        self.assertIsInstance(load_model("srf_fitted000001"), ExplicitSplineSurface)
        self.assertIsInstance(load_model(self.SURFACE_URL), ExplicitSplineSurface)
        with self.assertRaises(ValueError):
            load_model("sbt_subset000001")      ## data, not a model: load_subset
        with self.assertRaises(ValueError):
            load_model("xyz_unknown00001")


def linear_hypersurface_payload(nx=3, log_last=False):
    """Degree-1 explicit spline of y = x1 + 2 x2 + ... + nx * xn on [0, 2]^nx (clamped knots
    [0, 0, 1, 2, 2]): its coefficients are y at the nodes 0, 1, 2, so it reproduces y exactly.
    log_last: the last input is logarithmic, x in [1, 100] (log10 knots), y uses log10 of it."""
    nodes = np.arange(3.0)
    grids = np.meshgrid(*[nodes] * nx, indexing="ij")
    c = sum((d + 1) * g for d, g in enumerate(grids))
    scales = ["Linear"] * nx + ["Linear"]
    if log_last:
        scales[nx - 1] = "Logarithmic"
    t = [[0.0, 0.0, 1.0, 2.0, 2.0]] * nx
    return {"v": 1, "nx": nx, "ny": 1, "k": [1] * nx, "t": t, "shape": [3] * nx,
            "c": c.ravel().tolist(), "domain": [[0.0, 2.0]] * nx, "scales": scales,
            "extrapolation": "none"}


def hypersurface_response(nx=3, log_last=False, uid="hsf_hyper0000001"):
    names = ["Re", "Pr", "L/D", "Tw"][:nx]
    return {
        "uid": uid, "name": "Spline Hypersurface 1", "description": "", "local_id": 1,
        "data_relation_uid": "drn_relation0001", "relation_name": "Multivariate Relation 1",
        "form": "explicit", "nx": nx, "spline": linear_hypersurface_payload(nx, log_last),
        "labels": {"inputs": names, "y": "Nu"},
        "scales": {"inputs": ["Linear"] * nx, "y": "Linear"},
        "subset_uids": ["sbt_subset000001"], "method": "rmtb",
        "fit_params": {"degree": [1] * nx}, "accuracy": {"r2": 1.0, "rmse": 0.0},
        "fit_status": "ok",
    }


class HypersurfaceClientTests(unittest.TestCase):

    HSF_URL = SPLINECLOUD_API_URL + '/hypersurfaces/hsf_hyper0000001/'
    REL_URL = SPLINECLOUD_API_URL + '/relations/multivariate/fitted/drn_relation0001/'
    SUBSET_URL = SPLINECLOUD_API_URL + '/subsets/sbt_subset000001/'

    def _add(self, response=None):
        responses.add(responses.GET, self.HSF_URL, json=response or hypersurface_response(),
                      status=200, match=[matchers.query_param_matcher({"spline": "native"})])

    def _add_relation(self, excluded=None, hypersurfaces=None):
        responses.add(responses.GET, self.REL_URL, status=200, match=[
            matchers.query_param_matcher({"spline": "native"})], json={
            "uid": "drn_relation0001", "name": "Multivariate Relation 1", "describ": "",
            "nx": 3, "subset": {"uid": "sbt_subset000001"},
            "data_mapping": {"input_idx": [2, 0, 1], "y_idx": 3},
            "labels": {"inputs": ["Re", "Pr", "L/D"], "y": "Nu"},
            "scales": {"inputs": ["Linear"] * 3, "y": "Linear"},
            "excluded_points": excluded, "datapoint_weights": None,
            "hypersurfaces": hypersurfaces if hypersurfaces is not None
            else [hypersurface_response()],
        })
        responses.add(responses.GET, self.SUBSET_URL, status=200, json={"table": {
            "pr": {"0": 0.5, "1": 1.0, "2": 1.5, "3": 2.0},
            "ld": {"0": 0.1, "1": 0.2, "2": 0.3, "3": None},     ## row 3: a gap
            "re": {"0": 1.0, "1": 1.5, "2": 0.5, "3": 1.0},
            "nu": {"0": 2.5, "1": 4.1, "2": 3.9, "3": 5.0},
        }})

    @responses.activate
    def test_load_and_metadata(self):
        self._add()
        hs = load_spline_hypersurface("hsf_hyper0000001")
        self.assertIsInstance(hs, SplineHyperSurface)
        self.assertIsInstance(hs, NdBsplExplicit)
        self.assertEqual(hs.nx, 3)
        self.assertEqual(hs.labels, ["Re", "Pr", "L/D", "Nu"])
        self.assertEqual((hs.form, hs.method, hs.fit_status), ("explicit", "rmtb", "ok"))
        self.assertEqual(hs.relation_uid, "drn_relation0001")
        self.assertIsInstance(load_spline_hypersurface(self.HSF_URL), SplineHyperSurface)

    @responses.activate
    def test_evaluation(self):
        self._add()
        hs = load_spline_hypersurface("hsf_hyper0000001")
        y = lambda x1, x2, x3: x1 + 2 * x2 + 3 * x3
        self.assertAlmostEqual(hs(1.0, 0.5, 1.5), y(1.0, 0.5, 1.5))
        points = np.array([[0.2, 1.3, 0.7], [2.0, 2.0, 2.0], [0.0, 0.0, 0.0]])
        np.testing.assert_allclose(hs.eval(points), [y(*p) for p in points])
        np.testing.assert_allclose(hs(points[:, 0], points[:, 1], points[:, 2]),
                                   [y(*p) for p in points])
        grid = hs.eval_grid([0, 1], [0, 1, 2], [0, 0.5, 1, 2])
        self.assertEqual(grid.shape, (2, 3, 4))
        self.assertAlmostEqual(grid[1, 2, 1], y(1, 2, 0.5))

    @responses.activate
    def test_out_of_domain_and_policies(self):
        self._add()
        hs = load_spline_hypersurface("hsf_hyper0000001")
        self.assertTrue(np.isnan(hs(3.0, 0.0, 0.0)))
        self.assertAlmostEqual(hs.eval([[3.0, 0.0, 0.0]], extrapolate="linear")[0], 3.0)
        self.assertAlmostEqual(hs.eval([[3.0, 0.0, 0.0]], extrapolate="clamp")[0], 2.0)
        ## the loading policy overrides the server's suggestion
        self._add()
        clamped = load_spline_hypersurface("hsf_hyper0000001", extrapolation="clamp")
        self.assertAlmostEqual(clamped(3.0, 0.0, 0.0), 2.0)
        grad = hs.grad([[1.0, 1.0, 1.0], [5.0, 1.0, 1.0]])
        np.testing.assert_allclose(grad[0], [1.0, 2.0, 3.0])
        self.assertTrue(np.all(np.isnan(grad[1])))

    @responses.activate
    def test_slice_by_label_and_index(self):
        self._add()
        hs = load_spline_hypersurface("hsf_hyper0000001")
        surface = hs.slice({"Pr": 1.0})                 ## y(Re, L/D) at Pr = 1
        self.assertIsInstance(surface, NdBsplExplicit)
        self.assertEqual(surface.ndim, 2)
        self.assertAlmostEqual(surface(0.5, 2.0), 0.5 + 2.0 + 6.0)
        curve = hs.slice({0: 1.0, "L/D": 0.5})          ## y(Pr)
        self.assertIsInstance(curve, ExplicitUnivariateSpline)
        self.assertAlmostEqual(curve.eval(2.0), 1.0 + 4.0 + 1.5)
        with self.assertRaises(ValueError):
            hs.slice({"Mach": 1.0})
        with self.assertRaises(ValueError):
            hs.slice({5: 1.0})

    @responses.activate
    def test_log_axis(self):
        self._add(hypersurface_response(log_last=True))
        hs = load_spline_hypersurface("hsf_hyper0000001")
        ## x3 = 10 -> log10 = 1; x3 = 100 -> 2
        self.assertAlmostEqual(hs(1.0, 1.0, 10.0), 1.0 + 2.0 + 3.0)
        self.assertAlmostEqual(hs(0.0, 0.0, 100.0), 6.0)
        np.testing.assert_allclose(hs.domain[2], [1.0, 100.0])
        self.assertTrue(np.isnan(hs(0.0, 0.0, 0.5)))

    @responses.activate
    def test_four_inputs(self):
        self._add(hypersurface_response(nx=4))
        hs = load_spline_hypersurface("hsf_hyper0000001")
        self.assertEqual(hs.nx, 4)
        self.assertAlmostEqual(hs(1.0, 1.0, 1.0, 0.5), 1 + 2 + 3 + 2)
        self.assertEqual(hs.slice({"Tw": 1.0, "Re": 0.0}).ndim, 2)

    @responses.activate
    def test_wrong_payloads_and_ids(self):
        surface_like = hypersurface_response()
        surface_like["spline"] = {**linear_hypersurface_payload(2), "nx": 2}
        self._add(surface_like)
        with self.assertRaises(ValueError):
            load_spline_hypersurface("hsf_hyper0000001")
        responses.reset()
        self._add({**hypersurface_response(), "form": "implicit"})
        with self.assertRaises(ValueError):
            load_spline_hypersurface("hsf_hyper0000001")
        with self.assertRaises(ValueError):
            load_spline_hypersurface("srf_fitted000001")
        with self.assertRaises(ValueError):
            load_multivariate_relation("dr2_relation0001")

    @responses.activate
    def test_load_data(self):
        self._add()
        self._add_relation(excluded=[False, True, False])
        hs = load_spline_hypersurface("hsf_hyper0000001")
        labels, data = hs.load_data()
        self.assertEqual(labels, ["Re", "Pr", "L/D", "Nu"])
        ## inputs in input_idx order (re, pr, ld), then nu; the row with a gap is skipped
        np.testing.assert_allclose(data, [[1.0, 0.5, 0.1, 2.5], [1.5, 1.0, 0.2, 4.1],
                                          [0.5, 1.5, 0.3, 3.9]])
        np.testing.assert_array_equal(hs.excluded_points, [False, True, False])

    @responses.activate
    def test_load_data_stale_mask(self):
        self._add()
        self._add_relation(excluded=[False, True])      ## doesn't match the 3 points
        hs = load_spline_hypersurface("hsf_hyper0000001")
        hs.load_data()
        self.assertIsNone(hs.excluded_points)

    @responses.activate
    def test_multivariate_relation(self):
        second = hypersurface_response(uid="hsf_hyper0000002")
        second["name"] = "Spline Hypersurface 2"
        self._add_relation(excluded=[True, False, False],
                           hypersurfaces=[hypersurface_response(), second])
        relation = load_multivariate_relation("drn_relation0001", extrapolation="clamp")
        self.assertIsInstance(relation, MultivariateRelation)
        self.assertEqual((relation.uid, relation.nx, relation.subset_uid),
                         ("drn_relation0001", 3, "sbt_subset000001"))
        self.assertEqual(relation.labels, ["Re", "Pr", "L/D", "Nu"])
        self.assertEqual(relation.data_mapping, {"input_idx": [2, 0, 1], "y_idx": 3})
        self.assertEqual([h.uid for h in relation.hypersurfaces],
                         ["hsf_hyper0000001", "hsf_hyper0000002"])
        hs = relation.hypersurface("Spline Hypersurface 2")
        self.assertAlmostEqual(hs(3.0, 0.0, 0.0), 2.0)      ## the relation's policy
        with self.assertRaises(KeyError):
            relation.hypersurface("missing")
        labels, data, excluded = relation.load_data()
        self.assertEqual(data.shape, (3, 4))
        np.testing.assert_array_equal(excluded, [True, False, False])

    @responses.activate
    def test_load_model_dispatch(self):
        self._add()
        self.assertIsInstance(load_model("hsf_hyper0000001"), SplineHyperSurface)
        self._add_relation()
        self.assertIsInstance(load_model("drn_relation0001"), MultivariateRelation)
        self.assertIsInstance(load_model(self.REL_URL), MultivariateRelation)


if __name__ == '__main__':
    unittest.main()
