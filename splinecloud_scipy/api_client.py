# -*- coding: utf-8 -*-
import requests, json
import numpy as np

from .parametric_spline import ParametricUnivariateSpline
from .parametric_spline_surface import ParametricBivariateSpline


SPLINECLOUD_API_URL = "https://splinecloud.com/api"


class SplineSurface(ParametricBivariateSpline):
    """
    An extension of ParametricBivariateSpline that handles metadata
    specific to the SplineCloud API.
    """
    def __init__(self, data):
        sp = data["spline"]
        cp = np.array(sp["cp"], dtype=float)   # (nu, nv, 3)
        w  = np.array(sp["w"],  dtype=float)   # (nu, nv)
        tu = np.array(sp["tu"], dtype=float)
        tv = np.array(sp["tv"], dtype=float)
        ku = int(sp["ku"])
        kv = int(sp["kv"])

        log_x1 = data.get("scale_x1") == "Logarithmic"
        log_x2 = data.get("scale_x2") == "Logarithmic"
        log_y  = data.get("scale_y")  == "Logarithmic"

        super().__init__(
            tu, tv, cp, ku, kv, w=w,
            log_x1=log_x1, log_x2=log_x2, log_y=log_y,
        )

        # Identification and Metadata
        self.uid = data.get("uid")
        self.name = data.get("name")
        self.description = data.get("description")
        self.surface_type = data.get("surface_type")

        # Relational data
        self.subset_uids = data.get("subset_uids", [])
        self.x2_values = data.get("x2_values", [])
        self.curve_uids = data.get("curve_uids", [])
        self.relation_uid = data.get("relation_uid")
        self.relation_name = data.get("relation_name")

        # Labels
        labels = data.get("labels", {})
        self.x1_label = labels.get("x1")
        self.x2_label = labels.get("x2")
        self.y_label  = labels.get("y")

        self.subsets = []

    def load_subsets(self):
        """
        Load datasets (subsets) associated with this surface from the API.
        
        This method fetches all subsets identified by `subset_uids` and caches 
        them in `self.subsets` for subsequent calls.

        Returns
        -------
        list of tuples
            A list of (columns, table) tuples, where each tuple represents 
            a loaded subset.
        """
        if self.subsets:
            return self.subsets
        
        self.subsets = [load_subset(uid) for uid in self.subset_uids]
        return self.subsets

    def load_data(self):
        """
        Load and assemble raw data points from all associated subsets.
        
        Each subset contains (x1, y) data points. The x2 coordinate for
        each subset is taken from `x2_values`.

        Returns
        -------
        labels : list of str
            A list of [x1, x2, y] coordinate labels.
        data : ndarray
            An (N, 3) array of collected data points.
        """
        if not self.subsets:
            self.load_subsets()
        
        data = []
        for i, item in enumerate(self.subsets):
            subset_data = item[1]
            for point in subset_data:
                data.append([point[0], self.x2_values[i], point[1]])
        
        return [self.x1_label, self.x2_label, self.y_label], np.array(data)


# =============================================================================
# HELPERS
# =============================================================================

def fill_array(table, subset, columns, num_rows):
    for ci, col_name in enumerate(columns):
        for ri in range(num_rows):
            table[ri, ci] = subset[col_name][str(ri)]


# =============================================================================
# MAIN CLIENT FUNCTIONS
# =============================================================================

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

    return SplineSurface(data)
