import matplotlib.pyplot as plt
import numpy as np

density=[0.0009087, 0.2281885 ,0.4551077 ,0.5224999, 0.4839575, 0.4100087, 0.342483 , 0.3006067, 0.2834726, 0.2881028 ,0.3078476 ,0.3305594, 0.3471164, 0.3543028]
plt.scatter(np.arange(len(density)),density)
plt.savefig('density_test.png')