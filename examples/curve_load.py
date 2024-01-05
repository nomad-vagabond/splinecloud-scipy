from splinecloud_scipy import load_spline
import numpy as np
import matplotlib.pyplot as plt

curve_id = 'spl_K5t56P5bormJ'
spline = load_spline(curve_id)

X = np.linspace(0, 20, 100)
Y = [spline.eval(x) for x in X]

plt.plot(X,Y)
plt.show()