import numpy as np
Qflat_added = np.array([[1, 2], [3, 4], [5, 6]])
qflat = np.array([[3, 4], [5, 6]])

# Find indices where rows match using numpy
surviving_indices = np.array([i for i, x in enumerate(Qflat_added) if any(np.array_equal(x, y) for y in qflat)])

print(surviving_indices)