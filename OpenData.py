import numpy as np  
import matplotlib.pyplot as plt


Luse, Lcrav, Lcues = np.load('CravCues.npy')

for i in range(10):
    plt.plot(Luse[i])
    plt.show()
