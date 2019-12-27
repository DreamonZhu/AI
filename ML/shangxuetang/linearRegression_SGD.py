import numpy as np

x_data = np.random.rand(100, 1)
y_data = x_data * 4 + 3 + np.random.randn(100, 1)
x_n = np.c_[np.ones((100, 1)), x_data]

n_epochs = 500
t0 = 3; t1 = 4


def learning_schedule(t):
    return t0 / (t + t1)


# theta 维度是由 x_n 的列数决定的 其实就是 w0...wn
theta = np.random.randn(2, 1)

for epoch in range(n_epochs):
    for i in range(100):
        index = np.random.randint(100)
        x_i = x_n[index:index+1]
        gradient = 2*x_i.T.dot(x_i.dot(theta) - y_data[index])
        # 每一次的学习率都减小
        learning_rate = learning_schedule(epoch*100 + i)
        theta = theta - learning_rate * gradient

print(theta)
