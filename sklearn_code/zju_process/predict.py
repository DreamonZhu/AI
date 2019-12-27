import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, Normalizer, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import f_regression
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from sklearn.neural_network import MLPRegressor

from matplotlib import pyplot as plt
import matplotlib as mpl


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # df = pd.read_csv('data150.csv')
    df = pd.read_excel('data150.xls', index_col="序号")
    # print(df.head())
    df = df.dropna(axis=1, how='all')
    print(df.tail())
    x_total = df.iloc[:, :-1]
    x_total = x_total.iloc[:, [1, 2, 3, 4, 5, 7]]
    # print(x_total.shape)
    # print(x_total)
    y_total = df.iloc[:, -1]

    # print(x_total)
    # print(y_total)

    # x_total = StandardScaler().fit_transform(x_total)
    # # print(x_total[0:5])
    # x_total = Normalizer().fit_transform(x_total)

    # F, pv = f_regression(x_total, y_total)

    # x_train = x_total[:100]
    # y_train = y_total[:100]
    #
    # x_test = x_total[101:]
    # y_test = y_total[101:]

    x_predict, x_validate, y_predict, y_validate = train_test_split(x_total, y_total, test_size=(1/3),)

    print("训练集数据的大小", x_predict.shape)
    print("测试集数据的大小", x_validate.shape)

    # 1. 基于线性回归
    # lr = LinearRegression()
    # lr.fit(x_predict, y_predict)
    # y_hat = lr.predict(x_validate)
    # # mse = np.average((y_hat-y_validate)**2)
    #
    # print("Coefficients: \n", lr.coef_)
    # print("Validate Mean squared error: %.2f" % mean_squared_error(y_validate, y_hat))
    # # print("MSE: %.2f" % mse)
    # print('Validate Variance score: %.2f' % r2_score(y_validate, y_hat))
    #
    # y_test_hat = lr.predict(x_test)
    # print("Test Mean squared error: %.2f" % mean_squared_error(y_test, y_test_hat))
    # # print("MSE: %.2f" % mse)
    # print('Test Variance score: %.2f' % r2_score(y_test, y_test_hat))

    # plt.figure(facecolor='w')
    # x_axis = np.arange(len(x_validate))
    # # print(len(x_test))  # len(DataFrame) 返回的是较长的维度的数值，shape是返回两个维度的元祖
    # plt.plot(x_axis, y_validate, 'r^-', label='真实值')
    # plt.plot(x_axis, y_hat, 'g-', label='预测值')
    # plt.legend(loc='upper left')
    # plt.title('线性回归预测', fontsize=18)
    # plt.grid(True, ls=':')
    # plt.show()

    # 2. 基于多项式拟合
    # model = Pipeline([('poly', PolynomialFeatures(degree=6)),
    #                   ('linear', LinearRegression(fit_intercept=False))])
    # model.fit(x_predict, y_predict)
    # y_hat = model.predict(x_validate)
    # print("Validate Mean squared error: %.2f" % mean_squared_error(y_validate, y_hat))
    # print('Validate Variance score: %.2f' % r2_score(y_validate, y_hat))
    #
    # y_test_hat = model.predict(x_test)
    # print("Test Mean squared error: %.2f" % mean_squared_error(y_test, y_test_hat))
    # print('Test Variance score: %.2f' % r2_score(y_test, y_test_hat))

    # 3.svm
    # clf = svm.SVR(C=0.2)
    # clf.fit(x_predict, y_predict)
    # y_hat = clf.predict(x_validate)
    # print("score: %.6f " % clf.score(x_validate, y_validate))
    # print("Validate Mean squared error: %.6f" % mean_squared_error(y_validate, y_hat))
    # print('Validate Variance score: %.2f' % r2_score(y_validate, y_hat))
    #
    # y_test_hat = clf.predict(x_test)
    # print("Test Mean squared error: %.2f" % mean_squared_error(y_test, y_test_hat))
    # print('Test Variance score: %.2f' % r2_score(y_test, y_test_hat))

    # 4. 决策树
    # regr1 = DecisionTreeRegressor(max_depth=7)
    # regr1.fit(x_predict, y_predict)
    # y_hat = regr1.predict(x_validate)
    # score_validate = regr1.score(x_validate, y_validate)
    # print("score_validate: %.6f" % score_validate)
    # print("Validate Mean squared error: %.6f" % mean_squared_error(y_validate, y_hat))
    # print('Validate Variance score: %.2f' % r2_score(y_validate, y_hat))
    #
    # y_test_hat = regr1.predict(x_test)
    # print("Test Mean squared error: %.2f" % mean_squared_error(y_test, y_test_hat))
    # print('Test Variance score: %.2f' % r2_score(y_test, y_test_hat))
    # score_test = regr1.score(x_test, y_test)
    # print("score_test: %6.f" % score_test)

    # 5. 梯度提升算法
    clf = GradientBoostingRegressor(n_estimators=50, learning_rate=0.2, max_depth=3,)
    clf.fit(x_predict, y_predict)
    score_validate = clf.score(x_validate, y_validate)
    y_validate_hat = clf.predict(x_validate)
    print("MSE: %.6f " % mean_squared_error(y_validate, y_validate_hat))
    print("score_validate: %.6f" % score_validate)
    # accs = cross_val_score(clf, x_predict, y=y_predict, scoring=None, cv=10, n_jobs=1)
    # print(accs)

    # 6. 神经网络
    # model = MLPRegressor(learning_rate='adaptive', learning_rate_init=0.01, max_iter=200)
    # model.fit(x_predict, y_predict)
    # acc1 = model.score(x_validate, y_validate)
    # acc2 = model.score(x_test, y_test)
    # print(acc1)
    # print(acc2)

    # 6. 画图
    plt.figure(facecolor='w')
    x_axis = np.arange(len(x_validate))
    # print(len(x_test))  # len(DataFrame) 返回的是较长的维度的数值，shape是返回两个维度的元祖
    plt.plot(x_axis, y_validate, 'r^-', label='真实值')
    plt.plot(x_axis, y_validate_hat, 'g-', label='预测值')
    plt.legend(loc='upper left')
    plt.title('提升算法预测', fontsize=18)
    plt.grid(True, ls=':')
    plt.show()



