import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from matplotlib import pyplot as plt
import matplotlib as mpl
import warnings
from sklearn.exceptions import ConvergenceWarning


if __name__ == '__main__':
    warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
    N = 9
    np.random.seed(0)  # 保证每次生成的随机数一致
    x = np.linspace(0, 6, N) + np.random.randn(N)
    # print(x)
    x = np.sort(x)
    y = x**2 - 4*x - 3 + np.random.randn(N)
    x.shape = -1, 1  # 使得x、y变成二维的
    y.shape = -1, 1
    # print(x)
    # print(x.shape)
    # print(y.shape)
    # print(y.ravel().shape)

    models = [Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression(fit_intercept=False))
    ]), Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', RidgeCV(alphas=np.logspace(-3, 2, 10), fit_intercept=False))
    ]), Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', LassoCV(alphas=np.logspace(-3, 2, 10), fit_intercept=False))
    ]), Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', ElasticNetCV(alphas=np.logspace(-3, 2, 10), l1_ratio=[.1, .5, .7, .9, .95, .99, 1], fit_intercept=False))
    ])]

    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(12, 8), facecolor='w')
    d_pool = np.arange(1, N)
    m = d_pool.size
    print(m)
    clrs = []
    for c in np.linspace(16711680, 255, m, dtype=int):
        clrs.append('#%06x' % c)  # 表示6位16进制数
    print(clrs)
    line_width = np.linspace(5, 2, m)  # [5. 4.57142857 4.14285714 3.71428571 3.28571429 2.85714286 2.42857143 2.]
    # print(line_width)
    titles = '线性回归', 'Ridge回归', 'LASSO', 'ElasticNet'
    tss_list = []  #
    rss_list = []
    ess_list = []
    ess_rss_list = []

    for t in range(4):
        model = models[t]
        plt.subplot(2, 2, t+1)  # 2行 2列 第 t+1 个
        plt.plot(x, y, 'ro', ms=10, zorder=N)
        for i, d in enumerate(d_pool):
            model.set_params(poly__degree=N)
            model.fit(x, y.ravel())
            lin = model.get_params('linear')['linear']
            output = '%s：%d阶，系数为：' % (titles[t], d)
            if hasattr(lin, 'alpha_'):
                idx = output.find('系数')
                output = output[:idx] + ('alpha=%.6f，' % lin.alpha_) + output[idx:]
            if hasattr(lin, 'l1_ratio_'):  # 根据交叉验证结果，从输入l1_ratio(list)中选择的最优l1_ratio_(float)
                idx = output.find('系数')
                output = output[:idx] + ('l1_ratio=%.6f，' % lin.l1_ratio_) + output[idx:]
            print(output, lin.coef_.ravel())
            x_hat = np.linspace(x.min(), x.max(), num=100)
            x_hat.shape = -1, 1
            y_hat = model.predict(x_hat)
            s = model.score(x, y)

            z = N - 1 if (d == 2) else 0
            label = '%d阶，$R^2$=%.3f' % (d, s)
            if hasattr(lin, 'l1_ratio_'):
                label += '，L1 ratio=%.2f' % lin.l1_ratio_
            plt.plot(x_hat, y_hat, color=clrs[i], lw=line_width[i], alpha=0.75, label=label, zorder=z)
        plt.legend('upper left')
        plt.grid(True)
        plt.title(titles[t], fontsize=18)
        plt.xlabel('X', fontsize=16)
        plt.ylabel('Y', fontsize=16)
    plt.suptitle('多项式曲线拟合比较', fontsize=22)
    plt.show()


