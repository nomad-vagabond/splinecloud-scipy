from splinecloud_scipy import load_spline
import numpy as np
import matplotlib.pyplot as plt

curve_id = 'spl_K5t56P5bormJ'
spline = load_spline(curve_id)
columns, table = spline.load_data()

X = np.linspace(0, 20, 100)
Y = [spline.eval(x) for x in X]
x_data, y_data = table.T

plt.plot(X,Y)
plt.plot(x_data, y_data, 'o', color="grey")
plt.grid()
plt.xlabel(columns[0])
plt.ylabel(columns[1])
plt.show()