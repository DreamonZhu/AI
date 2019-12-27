import numpy as np

"""注意当只传一个参数的时候，得到的是一维的，传两个值才是二维的，对应矩阵
x_data = np.random.rand(100)
y_data = np.random.rand(100, 1)
print(x_data)
print(y_data)
"""

x_data = np.random.rand(100, 1)
y_data = 3 + 4 * x_data + np.random.randn(100, 1)
x_b = np.c_[np.ones((100, 1)), x_data]

theta = np.random.randn(2, 1)
iteration = 10000
learning_rate = 0.01
m = 100

for iterate in range(iteration):
    gradient = 1/m * x_b.T.dot(x_b.dot(theta) - y_data)
    theta = theta - learning_rate * gradient

print(theta)
