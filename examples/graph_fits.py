import matplotlib.pyplot as plt
import numpy as np

fits = np.loadtxt('fit_log.txt')
time = np.arange( len(fits) )

plt.plot(time, fits)
plt.show()

