import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

x = 6 * np.random.rand(100, 1) - 3
y = 3 * x**2 + 6 * x + np.random.randn(100, 1)

plt.plot(x, y, 'b.')

d = {1: 'r.', 2: 'g_',  10: 'y*'}
for i in d:
    poly_features = PolynomialFeatures(degree=i,include_bias=False)
    x_poly = poly_features.fit_transform(x)

    # tricks: LinearRegression 封装的模型处理 ~原始模型含有w0(截距)~ 必须要使用 要使 fit_intercept=True,否则得到的模型截距是0
    lin_reg = LinearRegression()
    lin_reg.fit(x_poly, y)
    print(lin_reg.coef_)
    print(lin_reg.intercept_)

    y_predict = lin_reg.predict(x_poly)
    plt.plot(x_poly[:, 0], y_predict, d[i])

plt.show()
