# -*- coding: utf-8 -*-
import requests, json
import numpy as np

from .parametric_spline import ParametricUnivariateSpline
from .parametric_spline_surface import ParametricBivariateSpline


SPLINECLOUD_API_URL = "https://splinecloud.com/api"


def fill_array(table, subset, columns, num_rows):
    for ci, col_name in enumerate(columns):
        for ri in range(num_rows):
            table[ri, ci] = subset[col_name][str(ri)]


def load_subset(subset_id_or_url):
    url_split = subset_id_or_url.split("/")
    if len(url_split) > 1:
        url = subset_id_or_url
    else:
        subset_id = url_split[-1]
        if "sbt_" not in subset_id or len(subset_id) != 16:
            raise ValueError("Wrong subset id was specified")
        url = SPLINECLOUD_API_URL+"/subsets/{}/".format(subset_id)
    
    response = requests.get(url)
    subset = json.loads(response.content)['table']

    columns = list(subset.keys())
    num_rows = len(list(subset.values())[0])
    table = np.zeros((num_rows, len(columns)))

    try:
        fill_array(table, subset, columns, num_rows)
    except ValueError:
        table = np.empty((num_rows, len(columns)), dtype=object)
        fill_array(table, subset, columns, num_rows)

    return columns, table
        
        
def load_spline(curve_id_or_url):
    url_split = curve_id_or_url.split("/")
    if len(url_split) > 1:
        url = curve_id_or_url
    else:
        curve_id = url_split[-1]
        if "spl_" not in curve_id or len(curve_id) != 16:
            raise ValueError("Wrong curve id was specified")
        url = SPLINECLOUD_API_URL+"/curves/{}/".format(curve_id)

    response = requests.get(url)
    curve = json.loads(response.content)

    curve_params = curve['spline']
    t = np.array(curve_params['t'])
    c = np.array(curve_params['c'])    
    tcck = t, c[:, 0], c[:, 1], curve_params['k']

    log_x = curve['scale_x'] == "Logarithmic"
    log_y = curve['scale_y'] == "Logarithmic"

    spline = ParametricUnivariateSpline(tcck, log_x=log_x, log_y=log_y)
    spline.load_data = lambda: load_subset(curve['subset_uid'])

    return spline


def load_spline_surface(surface_id_or_url):
    """
    Fetch a spline surface from the SplineCloud API and return a
    ParametricBivariateSpline instance.

    Accepts either a full URL or a bare surface id (prefix ``srf_``).
    """
    url_split = surface_id_or_url.split("/")
    if len(url_split) > 1:
        url = surface_id_or_url
    else:
        surface_id = url_split[-1]
        if "srf_" not in surface_id or len(surface_id) != 16:
            raise ValueError("Wrong surface id was specified")
        url = SPLINECLOUD_API_URL + "/surfaces/{}/".format(surface_id)

    response = requests.get(url)
    response.raise_for_status()
    data = json.loads(response.content)

    sp = data["spline"]
    cp = np.array(sp["cp"], dtype=float)   # (nu, nv, 3)
    w  = np.array(sp["w"],  dtype=float)   # (nu, nv)
    tu = np.array(sp["tu"], dtype=float)
    tv = np.array(sp["tv"], dtype=float)
    ku = int(sp["ku"])
    kv = int(sp["kv"])

    log_x = data.get("scale_x") == "Logarithmic"
    log_y = data.get("scale_y") == "Logarithmic"
    log_z = data.get("scale_z") == "Logarithmic"

    flip_yz = data.get("surface_type") == "lofted"

    surface = ParametricBivariateSpline(
        tu, tv, cp, ku, kv, w=w,
        log_x=log_x, log_y=log_y, log_z=log_z,
        flip_yz=flip_yz,
    )
    # Attach subset loader lazily, mirroring the curve API
    surface.subset_uids = data.get("subset_uids", [])
    surface.load_data = lambda: [load_subset(uid) for uid in surface.subset_uids]

    surface.x_label = data["labels"]["xlabel"]
    surface.y_label = data["labels"]["ylabel"]
    surface.z_label = data["labels"]["zlabel"]
   
    return surface
