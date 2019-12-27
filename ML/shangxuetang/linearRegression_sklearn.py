import numpy as np
from sklearn.linear_model import LinearRegression

x_data = np.random.rand(100, 1)
y_data = 2 + 3 * x_data + np.random.randn(100, 1)

myLR = LinearRegression()
myLR.fit(x_data, y_data)
print(myLR.coef_, myLR.intercept_)

x_new = np.array([[0], [2]])
print(myLR.predict(x_new))
