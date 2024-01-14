# -*- coding: utf-8 -*-
import requests, json
import numpy as np

from .parametric_spline import ParametricUnivariateSpline


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
        url = SPLINECLOUD_API_URL+"/subsets/{}".format(subset_id)
    
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
        url = SPLINECLOUD_API_URL+"/curves/{}".format(curve_id)

    response = requests.get(url)
    curve = json.loads(response.content)

    curve_params = curve['spline']
    t = np.array(curve_params['t'])
    c = np.array(curve_params['c'])    
    tcck = t, c[:, 0], c[:, 1], curve_params['k']

    return ParametricUnivariateSpline(tcck)
        
