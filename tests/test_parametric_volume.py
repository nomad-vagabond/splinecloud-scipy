"""Lofted spline volumes from SplineCloud (parametric volume schema v1, contract C.7):
ParametricSplineVolume, SplineVolume / LoftedRelation3D and their loaders.

tests/data/lofted_volume_cla.json was produced by the server (compose.loft_volume) from four
station surfaces, Cla*(eta, TR) at AR = 6, 8, 10, 12: the station payloads, the volume
with a linear and a logarithmic x3 axis, and the server's slices (slice_at) at some x3.
"""
import json
import os
import unittest

import numpy as np
import responses

from splinecloud_scipy import (
    SPLINECLOUD_API_URL,
    ParametricBivariateSpline,
    ParametricSplineVolume,
    SplineVolume,
    SplineHyperSurface,
    LoftedRelation3D,
    load_spline_hypersurface,
    load_lofted_relation_3d,
    load_model,
)

FIXTURE = os.path.join(os.path.dirname(__file__), "data", "lofted_volume_cla.json")


def load_fixture():
    with open(FIXTURE) as fixture:
        return json.load(fixture)


def station_surface(station):
    sp = station["spline"]
    logs = [scale == "Logarithmic" for scale in station["scales"]]   ## [x1, x2, y]
    return ParametricBivariateSpline(sp["tu"], sp["tv"], sp["cp"], sp["ku"], sp["kv"],
                                     w=sp["w"], log_x1=logs[0], log_x2=logs[1],
                                     log_y=logs[2])


def interior_points(surface, n=12, seed=3):
    """(x1, x2) of surface points at random interior parameters (inside its domain)."""
    rng = np.random.default_rng(seed)
    lo_u, hi_u = surface.tu[surface.ku], surface.tu[-surface.ku - 1]
    lo_v, hi_v = surface.tv[surface.kv], surface.tv[-surface.kv - 1]
    points = []
    for _ in range(n):
        u = lo_u + rng.uniform(0.05, 0.95) * (hi_u - lo_u)
        v = lo_v + rng.uniform(0.05, 0.95) * (hi_v - lo_v)
        x1, x2, _y = surface(u, v)               ## scalars: (x1, x2, y), physical
        points.append((x1, x2))
    return points


class ServerVolumeTests(unittest.TestCase):
    """The client reads the server's volumes as the server does."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture()
        cls.stations = cls.fixture["stations"]
        cls.surfaces = [station_surface(st) for st in cls.stations]

    def volume(self, scale="Linear"):
        return ParametricSplineVolume(self.fixture["volumes"][scale]["payload"])

    def test_slices_equal_the_servers(self):
        for scale, entry in self.fixture["volumes"].items():
            volume = ParametricSplineVolume(entry["payload"])
            for x3, server_cp in entry["slices"].items():
                surface = volume.slice_at(float(x3))
                cp = np.stack([surface._tck_x1[2], surface._tck_y[2], surface._tck_x2[2]],
                              axis=-1).reshape(np.asarray(server_cp).shape)
                np.testing.assert_allclose(cp, server_cp, rtol=0, atol=1e-12,
                                           err_msg=f"{scale} x3 = {x3}")

    def test_stations_reproduced(self):
        for scale in ("Linear", "Logarithmic"):
            volume = self.volume(scale)
            for station, surface in zip(self.stations, self.surfaces):
                for x1, x2 in interior_points(surface):
                    expected = surface.eval_point(x1, x2)
                    got = volume.eval_point(x1, x2, station["x3"])
                    self.assertTrue(np.isfinite(got) and np.isfinite(expected))
                    self.assertAlmostEqual(got, expected, delta=1e-9,
                                           msg=f"{scale} x3 = {station['x3']}")

    def test_x3_bounds_and_w(self):
        linear, log = self.volume("Linear"), self.volume("Logarithmic")
        self.assertEqual(linear.x3_bounds, (6.0, 12.0))
        np.testing.assert_allclose(log.x3_bounds, (6.0, 12.0), rtol=1e-12)
        self.assertAlmostEqual(linear.x3_to_w(9.0), 0.5)
        self.assertAlmostEqual(log.x3_to_w(np.sqrt(72.0)), 0.5)   ## geometric middle

    def test_between_stations_match_the_servers_slices(self):
        ## y between stations, against surfaces built from the server's own slices (not
        ## from the client's slicing)
        for scale, entry in self.fixture["volumes"].items():
            volume = ParametricSplineVolume(entry["payload"])
            scales = entry["payload"]["scales"]
            for x3, server_cp in entry["slices"].items():
                x3 = float(x3)
                if x3 in [st["x3"] for st in self.stations]:
                    continue
                ## the slice is on the volume's common (u, v) basis
                payload = entry["payload"]
                server = ParametricBivariateSpline(
                    payload["tu"], payload["tv"], server_cp, payload["ku"], payload["kv"],
                    log_x1=scales[0] == "Logarithmic", log_x2=scales[1] == "Logarithmic",
                    log_y=scales[3] == "Logarithmic")
                for x1, x2 in interior_points(server, n=6, seed=int(x3 * 10)):
                    got, expected = volume.eval_point(x1, x2, x3), server.eval_point(x1, x2)
                    self.assertTrue(np.isfinite(got) and np.isfinite(expected))
                    self.assertAlmostEqual(got, expected, delta=1e-10,
                                           msg=f"{scale} x3 = {x3}")

    def test_eval_grid_and_scalars(self):
        volume = self.volume()
        x1s, x2 = [0.2, 0.5, 0.8], 0.6
        grid = volume.eval(x1s, x2, [7.0, 10.0])
        self.assertEqual(grid.shape, (3, 1, 2))
        self.assertAlmostEqual(grid[1, 0, 1], volume.eval(0.5, 0.6, 10.0), delta=1e-12)
        self.assertIsInstance(volume.eval(0.5, 0.6, 10.0), float)

    def test_x3_outside_range(self):
        volume = self.volume()
        ## clamped without extrapolation: the last station's value
        self.assertAlmostEqual(volume.eval_point(0.5, 0.6, 14.0),
                               volume.eval_point(0.5, 0.6, 12.0), delta=1e-12)
        with self.assertRaises(ValueError):
            volume.slice_at(14.0)
        ## extrapolated within the w guard, refused beyond it
        y, diagnostics = volume.eval_point(0.5, 0.6, 13.0, extrapolate=True,
                                           return_diagnostics=True)
        self.assertTrue(np.isfinite(y))
        self.assertTrue(diagnostics["w_extrapolated"])
        self.assertIsNone(volume.eval_point(0.5, 0.6, 30.0, extrapolate=True))
        self.assertTrue(np.isnan(volume.eval([0.5], 0.6, 30.0, extrapolate=True)[0, 0, 0]))

    def test_payload_checks(self):
        payload = self.fixture["volumes"]["Linear"]["payload"]
        with self.assertRaises(ValueError):
            ParametricSplineVolume({**payload, "v": 2})
        with self.assertRaises(ValueError):
            ParametricSplineVolume({**payload, "form": "explicit"})
        with self.assertRaises(ValueError):
            ParametricSplineVolume({**payload, "cp": np.asarray(payload["cp"])[:, :, :2]
                                    .tolist()})
        with self.assertRaises(ValueError):
            ParametricSplineVolume({**payload, "w": (np.asarray(payload["w"]) * 2).tolist()})


def volume_response(uid="hsf_volume000001"):
    fixture = load_fixture()
    payload = fixture["volumes"]["Linear"]["payload"]
    return {
        "uid": uid, "name": "Spline Volume 1", "description": "", "form": "parametric",
        "nx": 3, "spline": payload, "data_relation_uid": None,
        "loft_relation_uid": "lr3_relation0001", "relation_name": "Cla* by AR",
        "labels": {"inputs": ["eta", "TR", "AR"], "y": "cla*"},
        "scales": {"inputs": payload["scales"][:3], "y": payload["scales"][3]},
        "subset_uids": [], "method": None, "accuracy": None, "fit_status": "ok",
        "stations": [{"x3": st["x3"], "surface_uid": st["uid"]} for st in fixture["stations"]],
        "fit_params": {"w_degree": 3},
    }


class VolumeAPITests(unittest.TestCase):
    HSF_URL = SPLINECLOUD_API_URL + '/hypersurfaces/hsf_volume000001/'
    REL_URL = SPLINECLOUD_API_URL + '/relations/multivariate/lofted/lr3_relation0001/'

    def relation_response(self):
        volume = volume_response()
        return {"uid": "lr3_relation0001", "name": "Cla* by AR", "describ": "",
                "labels": volume["labels"], "scale_x3": "Linear", "w_degree": 3,
                "stations": [{"x3_value": st["x3"], "surface_uid": st["surface_uid"],
                              "surface_name": "Spline Surface 1", "surface_type": "lofted",
                              "relation_uid": "lr2_relation0001"}
                             for st in volume["stations"]],
                "hypersurfaces": [volume]}

    @responses.activate
    def test_load_hypersurface_dispatches_on_form(self):
        responses.add(responses.GET, self.HSF_URL, json=volume_response(), status=200)
        volume = load_spline_hypersurface("hsf_volume000001")
        self.assertIsInstance(volume, SplineVolume)
        self.assertNotIsInstance(volume, SplineHyperSurface)
        self.assertEqual(volume.labels, ["eta", "TR", "AR", "cla*"])
        self.assertEqual((volume.nx, volume.relation_uid, volume.fit_status),
                         (3, "lr3_relation0001", "ok"))
        self.assertEqual(volume.x3_stations, [6.0, 8.0, 10.0, 12.0])
        self.assertTrue(np.isfinite(volume.eval_point(0.5, 0.6, 9.0)))
        ## ?spline=native is sent, as for every 1.5 request
        self.assertIn("spline=native", responses.calls[0].request.url)

    @responses.activate
    def test_unknown_hypersurface_form_raises(self):
        responses.add(responses.GET, self.HSF_URL,
                      json={**volume_response(), "form": "implicit"}, status=200)
        with self.assertRaises(ValueError):
            load_spline_hypersurface("hsf_volume000001")

    @responses.activate
    def test_load_lofted_relation_3d(self):
        responses.add(responses.GET, self.REL_URL, json=self.relation_response(), status=200)
        relation = load_lofted_relation_3d("lr3_relation0001")
        self.assertIsInstance(relation, LoftedRelation3D)
        self.assertEqual((relation.name, relation.scale_x3, relation.w_degree),
                         ("Cla* by AR", "Linear", 3))
        self.assertEqual([st["x3_value"] for st in relation.stations], [6.0, 8.0, 10.0, 12.0])
        self.assertIsInstance(relation.volume, SplineVolume)
        self.assertIsInstance(load_model("lr3_relation0001"), LoftedRelation3D)
        with self.assertRaises(ValueError):
            load_lofted_relation_3d("drn_relation0001")

    @responses.activate
    def test_station_surfaces_from_a_volume(self):
        responses.add(responses.GET, self.HSF_URL, json=volume_response(), status=200)
        fixture = load_fixture()
        for st in fixture["stations"]:
            responses.add(responses.GET, SPLINECLOUD_API_URL + f"/surfaces/{st['uid']}/",
                          json={"uid": st["uid"], "form": "parametric", "spline": st["spline"],
                                "scale_x1": "Linear", "scale_x2": "Linear",
                                "scale_y": "Linear"}, status=200)
        volume = load_spline_hypersurface("hsf_volume000001")
        surfaces = volume.load_station_surfaces()
        self.assertEqual(len(surfaces), 4)
        ## each station surface is the volume's slice at its x3
        x1, x2 = interior_points(surfaces[1], n=1)[0]
        self.assertAlmostEqual(volume.eval_point(x1, x2, 8.0), surfaces[1].eval_point(x1, x2),
                               delta=1e-9)


if __name__ == "__main__":
    unittest.main()
