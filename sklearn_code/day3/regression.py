from sklearn_code.linear_model import LinearRegression, SGDRegressor, Ridge, LogisticRegression
from sklearn_code.datasets import load_boston
from sklearn_code.model_selection import train_test_split
from sklearn_code.preprocessing import StandardScaler
from sklearn_code.metrics import mean_squared_error, classification_report
import pandas_code as pd
import numpy as np
from sklearn_code.feature_extraction import DictVectorizer


def lg():
    """正规方程，return None"""
    lb = load_boston()

    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    print(type(y_test), type(y_train))
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    print(y_test.shape, y_train.shape)
    print(x_test.shape, x_train.shape)
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    # 注意此处不能使用同一个std进行标准化处理，因为目标值只有一列，而特征值有很多列，故在对特征值进行标准化处理之后，不知道该用哪一列来对目标值进行处理，只能再重新实例化一个标准化函数
    # std.transform(x_test, y_test)

    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train)
    y_test = std_y.transform(y_test)

    # 正规方程求解
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    print(lr.coef_)
    y_lr_predict = std_y.inverse_transform(lr.predict(x_test))
    print("测试集的预测值是：", y_lr_predict)
    print("误差大小是：", mean_squared_error(std_y.inverse_transform(y_test), y_lr_predict))
    return None


def sgd():
    """梯度下降"""
    lb = load_boston()

    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    y_test = y_test.reshape(-1, 1)
    y_train = y_train.reshape(-1, 1)
    print(y_test.shape, y_train.shape)
    print(x_test.shape, x_train.shape)
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train)
    y_test = std_y.transform(y_test)

    sgdr = SGDRegressor()
    sgdr.fit(x_train, y_train)

    print("梯度下降的权重系数：", sgdr.coef_)
    y_sgd_predict = std_y.inverse_transform(sgdr.predict(x_test))
    print("测试机的预测值是：", y_sgd_predict)
    print("误差是：", mean_squared_error(std_y.inverse_transform(y_test), y_sgd_predict))

    return None


def rg():
    """岭回归"""
    lb = load_boston()
    x_train, x_test, y_train, y_test = train_test_split(lb.data, lb.target, test_size=0.25)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    std_x = StandardScaler()
    x_train = std_x.fit_transform(x_train)
    x_test = std_x.transform(x_test)
    std_y = StandardScaler()
    y_train = std_y.fit_transform(y_train)
    y_test = std_y.transform(y_test)

    rdg = Ridge(alpha=1.0)
    rdg.fit(x_train, y_train)
    print("岭回归的权重系数：", rdg.coef_)
    y_rdg_predict = std_y.inverse_transform(rdg.predict(x_test))
    print("测试机预测值是：", y_rdg_predict)
    print("岭回归的误差是：", mean_squared_error(std_y.inverse_transform(y_test), y_rdg_predict))
    return None


def logistic():
    """逻辑回归处理二分类问题"""
    columns = ['Sample code number','Clump Thickness', 'Uniformity of Cell Size','Uniformity of Cell Shape','Marginal Adhesion', 'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class']
    df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data", names=columns)
    print(df.head())
    print(df.tail())
    print(df.info())

    # 缺失值处理
    df = df.replace(to_replace="?", value=np.nan)
    df = df.dropna()
    # print(df[[columns[-1]]])
    # return None

    x_train, x_test, y_train, y_test = train_test_split(df[columns[1:-1]], df[[columns[-1]]], test_size=0.25)
    # y_train = y_train.reshape(-1, 1)
    # y_test = y_test.reshape(-1, 1)
    print(type(x_test), type(y_test))
    # return None

    onehot = DictVectorizer(sparse=False)
    x_train = onehot.fit_transform(x_train.to_dict(orient='records'))
    x_test = onehot.transform(x_test.to_dict(orient="records"))

    # 未经过标准化
    # logistic = LogisticRegression(penalty='l2', C=1.0)
    # logistic.fit(x_train, y_train)
    # print("权重系数：", logistic.coef_)
    # y_lg_predict = logistic.predict(x_test)
    # print("测试机的预测值是：", y_lg_predict)
    # print("准确率：", logistic.score(x_test, y_test))
    # print("召回率：", classification_report(y_test, y_lg_predict, labels=[2, 4], target_names=['良性', '恶性']))

    # 进行标准化
    std = StandardScaler()
    x_train_std = std.fit_transform(x_train)
    x_test_std = std.transform(x_test)

    # 测试标准化是否有用 ,效果不明显
    # print(x_test_std == x_test) ===>不相同
    # print(x_train_std == x_train) ===>不相同

    logistic = LogisticRegression(penalty='l2', C=1.0)
    logistic.fit(x_train, y_train)
    print("权重系数：", logistic.coef_)
    y_lg_predict = logistic.predict(x_test)
    print("测试机的预测值是：", y_lg_predict)
    print("准确率：", logistic.score(x_test, y_test))
    print("召回率：", classification_report(y_test, y_lg_predict, labels=[2, 4], target_names=['良性', '恶性']))
    return None


if __name__ == '__main__':
    # lg()
    # sgd()
    # rg()
    logistic()
