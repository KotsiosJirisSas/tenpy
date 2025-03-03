import numpy as np

qflat = np.array([
    [1, -1],
    [1, -10],
    [0, 0],
    [1, 1],
])
sorted_charges = np.lexsort(np.array(qflat).T) 
qflat= qflat[sorted_charges]
print(qflat)