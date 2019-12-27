import numpy as np
import time
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import matplotlib as mpl


if __name__ == '__main__':
    n = 1000
    m = 512
    np.random.seed(2019)
    data_x_1 = np.zeros((n, m))
    data_x_2 = np.zeros((n, m))
    # print(data)
    # time1 = time.time()
    for i in range(n):
        for j in range(m):
            t = j * 0.000078125
            data_x_1[i, j] = 5 * np.cos(20*np.pi*t) + 10 * np.cos(40*np.pi*t)
            data_x_2[i, j] = 15 * np.cos(60*np.pi*t) + 20 * np.cos(80*np.pi*t)
    # time2 = time.time()
    # print(time2 - time1)  # 3.379418134689331
    w_x = 20*np.random.randn(n, m)
    # print(w_x)
    data_x = data_x_1 + data_x_2 + w_x
    # print(data_x)

    # time3 = time.time()
    t = np.arange(m) * 0.000078125
    data_y_1 = np.array([4*np.sin(25*np.pi*t)+np.sin(20*np.pi*t)+np.sin(40*np.pi**2*t) for i in range(n)])
    data_y_2 = np.array([(10+5*np.cos(10*np.pi*t))*np.cos(2*np.pi*t+2*np.cos(5*np.pi*t)) for i in range(n)])
    # time4 = time.time()
    # 效率提升明显
    # print(time4 - time3)  # 0.08105802536010742
    w_y = 20*np.random.randn(n, m)
    # print(w_y)
    data_y = data_y_1 + data_y_2 + w_y

    # time5 = time.time()
    data_z_1 = np.array([5*(1+np.cos(4*np.pi*t))*np.cos(20*np.pi*t)+10*(1+np.cos(4*np.pi*t))*np.cos(40*np.pi*t)+15*(1+np.cos(4*np.pi*t))*np.cos(60*np.pi*t)+20*(1+np.cos(4*np.pi*t))*np.cos(80*np.pi*t) for i in range(n)])
    # time6 = time.time()  # 0.09856986999511719
    # print(time6 - time5)
    w_z = np.random.randn(n, m)
    data_z = data_z_1 + w_z

    x_data = np.vstack((data_x, data_y, data_z))
    # print(data.shape)  # (3000, 512)
    data_label_1 = np.ones((n, 1))
    data_label_2 = np.ones((n, 1))*2
    data_label_3 = np.ones((n, 1))*3
    y_data = np.vstack((data_label_1, data_label_2, data_label_3))

    # 标准化数据
    std = StandardScaler()
    x_data = std.fit_transform(x_data)

    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3, random_state=2019)

    # 使用随机森林进行预测
    rf = RandomForestClassifier(n_estimators=10, criterion='entropy', max_depth=5)
    rf.fit(x_train, y_train.ravel())

    y_hat = rf.predict(x_test)
    print(y_hat)
    acc_test = accuracy_score(y_test, y_hat)
    print(acc_test)
    y_train_pre = rf.predict(x_train)
    acc_test_train = accuracy_score(y_train, y_train_pre)
    print(acc_test_train)

    # 使用逻辑回归进行预测
    # lr = LogisticRegression()
    # lr.fit(x_train, y_train.ravel())
    # y_hat = lr.predict(x_test)
    # print(y_hat)
    # print(lr.score(x_test, y_test))

    # 画图
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(12, 6))
    plt.plot(x_test[:, 1], y_test, 'g*', label='真实值')
    plt.plot(x_test[:, 1], y_hat, 'ro', label='预测值')
    plt.legend()
    plt.show()




