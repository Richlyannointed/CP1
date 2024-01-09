import numpy as np
import scipy as sp
import scipy.stats as stats
import matplotlib as mpl
import matplotlib.pyplot as plt

x = np.array([0.,1.,2.,3.])
y = np.array([10.,20.,30.,40.])

n,(xmin,xmax),m,v,s,k = stats.describe(x)

print("Kurtosis = ", k)

plt.plot(x,y,'b-')
plt.show()
