import pandas_code as pd
import numpy as np
from sklearn_code.datasets import load_iris
from sklearn_code.preprocessing import StandardScaler
from sklearn_code.neighbors import KNeighborsClassifier
from sklearn_code.model_selection import train_test_split, GridSearchCV
from sklearn_code.feature_extraction import DictVectorizer
from sklearn_code.tree import DecisionTreeClassifier
from sklearn_code.ensemble import RandomForestClassifier


def knn():
    li = load_iris()
    x = li.data
    y = li.target
    # 分割训练集、测试集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25)
    # 特征处理
    std = StandardScaler()
    std.fit_transform(x_train)
    std.transform(x_test)
    # knn分类
    knncls = KNeighborsClassifier(n_neighbors=5)
    knncls.fit(x_train, y_train)
    print("测试机的预测值是：", knncls.predict(x_test))
    print("预测的准确率是：", knncls.score(x_test, y_test))


def dt():
    """决策树算法，return None"""
    df = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
    # print(df.head())
    # print(df.info())
    # 选取数据的特征值与目标值
    y = df['survived']
    x = df[['pclass', 'age', 'sex']]
    # 对缺失数据进行处理
    # df.fillna(df['age'], df['age'].mean(), inplace=True) ==>错误的使用
    x['age'].fillna(x['age'].mean(), inplace=True)
    # 对数据进行分割
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)
    print(type(x_test))
    print(type(y_test))
    # print(y_test)

    # 决策树实例化
    # dtc = DecisionTreeClassifier()

    # dtc.fit(x_train, y_train) ===> 说明DataFrame是array-like类型的数据，只是由于在机器学习api中需要将数据转换为float，而DataFrame中有的数据类型是字符串导致报错
    # return None
    # 进行特征处理
    onehot = DictVectorizer(sparse=False)
    x_train = onehot.fit_transform(x_train.to_dict(orient="records"))
    # print(x_train)
    # print(onehot.get_feature_names())
    x_test = onehot.transform(x_test.to_dict(orient="records"))
    print(type(x_test))

    # 决策树
    # dtc.fit(x_train, y_train)
    # y_predict = dtc.predict(x_test)
    # print("训练集的预测值是：", y_predict)
    # print("测试准确率是：", dtc.score(x_test, y_test))

    # 随机森林
    rf = RandomForestClassifier()
    params = {"n_estimators": [120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    gc = GridSearchCV(rf, param_grid=params, cv=2)
    gc.fit(x_train, y_train)
    print("准确率：", gc.score(x_test, y_test))
    print("查看选择的参数模型：", gc.best_params_)

    return None


if __name__ == '__main__':
    dt()
    # df = pd.DataFrame(np.array(range(12)).reshape((3, 4)), columns=['a', 'b', 'c', 'd'])
    # print(df)
    # print(df.groupby("d").count())
    # for i, j in df.groupby(by="d"):
    #     print(i)
    #     print(j)
    # print("*"*100)
    # df1 = pd.DataFrame(np.array([['a', 'b', 'c'], ['a', 'b', 'c'], ['c', 'b', 'a']]), columns=['x', 'y', 'z'])
    # print(df1)
    # place_out = df1.groupby("z")
    # # for i, j in place_out:
    # #     print(i)
    # #     print(type(j))
    # print(place_out.count())
    # print(type(place_out.count().loc['a']))
    # print(type(place_out.count().x))
    # print(place_out.count().reset_index())
