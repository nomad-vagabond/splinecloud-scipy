# -*- coding: utf-8 -*-
import requests, json
import pandas as pd
import numpy as np


from .parametric_spline import ParametricUnivariateSpline


SPLINECLOUD_API_URL = "https://splinecloud.com/api"


def LoadSubset(subset_id_or_url):
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
    
    return pd.DataFrame.from_dict(subset)
        
        
def LoadSpline(curve_id_or_url):
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
    w = curve_params['w']
    tcck = t, c[:, 0], c[:, 1], curve_params['k']

    return ParametricUnivariateSpline.from_tcck(tcck)
        
