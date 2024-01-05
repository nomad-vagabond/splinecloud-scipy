# splinecloud-scipy

A Python client for [SplineCloud API](https://splinecloud.com/api/docs/)

The client library is based on [SciPy](https://scipy.org/) and allows to load data and curves from [SplineCloud](https://splinecloud.com/) into your code. Once loaded spline curves can be easily evaluated.

## Example of loading spline curve

<img src="/docs/img/curve_api_link.png?raw=true" width=80% alt="SplineCloud curve">

```python
from splinecloud_scipy import load_spline
curve_id = 'spl_K5t56P5bormJ' # take curve ID from the 'API link' dropdown at SplineCloud
spline = load_spline(curve_id)
```

## Evaluating spline curve for a range of x values

```python
import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(0, 20, 100)
Y = [spline.eval(x) for x in X]

plt.plot(X,Y)
plt.show()
```

![Spline curve](/docs/img/curve.png?raw=true)

## Example of loading data in your code

```python
from splinecloud_scipy import load_subset
subset_id = 'sbt_nDO4XmmYqeGI' # subset id can be taken from the SplineCloud
columns, table = load_subset(subset_id)
```

```
>>> columns

['Throttle (%)',
 'Load Currency (A)',
 'Pull (g)',
 'Power (W)',
 'Efficiency (g/W)']
```

```
>>> table

array([[5.0000e-01, 6.7600e+00, 3.8500e+02, 1.0871e+02, 3.5420e+00],
       [6.0000e-01, 1.0200e+01, 4.9500e+02, 1.6249e+02, 3.0460e+00],
       [7.0000e-01, 1.3580e+01, 6.0600e+02, 2.1768e+02, 2.7840e+00],
       [8.0000e-01, 1.7390e+01, 6.8700e+02, 2.7140e+02, 2.5510e+00],
       [9.0000e-01, 2.1030e+01, 7.4700e+02, 3.2813e+02, 2.2770e+00],
       [1.0000e+00, 2.5060e+01, 8.0700e+02, 3.8555e+02, 2.0930e+00]])
```

These examples are available as notebooks in the project's 'examples' folder.

## Important

This library supports BSpline geometry but does not support weighted BSplines (NURBS). If you adjust the weights of the curve control points use another client, that supports NURBS (a link will be provided here soon).