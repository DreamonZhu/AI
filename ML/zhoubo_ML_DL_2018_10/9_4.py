from sklearn.datasets import load_iris
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LogisticRegression
import matplotlib as mpl
from matplotlib import pyplot as plt


if __name__ == '__main__':
    data = load_iris()
    np.set_printoptions(linewidth=250)
    # print(data)
    x = data.data
    y = data.target
    # print(y.shape)
    # print(x.shape)

    lr = Pipeline([
        ('sc', StandardScaler()),
        ('poly', PolynomialFeatures(degree=2)),
        ('clr', LogisticRegression(solver='sag', multi_class='auto'))
    ])

    lr.fit(x[:, :2], y.ravel())
    y_hat = lr.predict(x[:, :2])
    y_hat_prob = lr.predict_proba(x[:, :2])
    # print("y_hat = ", y_hat)
    # print("y_hat_prop = ", y_hat_prob)
    # print("准确率为 %.2f%%" % (100*np.mean(y_hat == y.ravel())))
    # print("准确率为：%.2f" % lr.score(x, y))
    # print(lr.get_params('clr')['clr'].coef_)  # 系数的维度是 (3, 15),其中3是因为有3各类别，15是指的当degree=2时，四个特征有15个维度
    # print(lr.named_steps['clr'].coef_.shape)  # 注意：逻辑回归没有截距参数即没有 .intercept

    N, M = 500, 500
    x1_min, x1_max = x[:, 0].min(), x[:, 0].max()
    x2_min, x2_max = x[:, 1].min(), x[:, 1].max()
    t1 = np.linspace(x1_min, x1_max, N)
    t2 = np.linspace(x2_min, x2_max, M)
    x1, x2 = np.meshgrid(t1, t2)
    # print("x1 = ", x1)
    # print(x1.shape)
    # print("x2 = ", x2)
    # print(x2.shape)
    # print((x1.flat, x2.flat))
    # print(len(list(x1.flat)))
    x_test = np.stack((x1.flat, x2.flat), axis=1)
    # print(x_test.shape)

    # x3 = np.ones(x1.size) * np.average(x[:, 2])
    # x4 = np.ones(x1.size) * np.average(x[:, 3])
    #
    # x_test = np.stack((x1.flat, x2.flat, x3.flat, x4.flat), axis=1)
    # print(x_test.shape)

    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_hat_1 = lr.predict(x_test)
    # print(y_hat_1)
    print(y_hat_1.shape)
    y_hat_1 = y_hat_1.reshape(x1.shape)
    print(y_hat_1.shape)
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, y_hat_1, cmap=cm_light)
    plt.scatter(x[:, 0], x[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)
    plt.xlabel("花萼长度", fontsize=14)
    plt.ylabel("花萼宽度", fontsize=14)
    plt.xlim(x1_min, x1_max)  # 设置x轴的长度
    plt.ylim(x2_min, x2_max)  # 设置y轴的长度
    plt.grid()
    plt.title("鸢尾花的Logistic回归分类效果 - 标准化", fontsize=17)
    plt.show()
