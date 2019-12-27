import numpy as np

x_data = np.random.rand(100, 1)
"""最后一项是正态分布的误差值"""
y_data = x_data * 3 + 2 + np.random.randn(100, 1)
x_b = np.c_[np.ones((100, 1)), x_data]
"""线性代数直接公式求解"""
w = np.linalg.inv(x_b.T.dot(x_b)).dot(x_b.T).dot(y_data)
print(w)
