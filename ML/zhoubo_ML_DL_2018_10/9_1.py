import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

df = pd.read_csv('./data/advertising.csv')
# print(df[['TV', 'Radio', 'Newspaper']])

x = df[['TV', 'Radio', 'Newspaper']]
y = df['Sales']
# print(type(x))  # <class 'pandas.core.frame.DataFrame'>
# print(type(y))  # <class 'pandas.core.series.Series'>

mpl.rcParams['font.sans-serif'] = ['simHei']
mpl.rcParams['axes.unicode_minus'] = False

# 绘图1
# plt.figure(facecolor='w')
# plt.plot(df['TV'], y, 'ro', label='TV')
# plt.plot(df['Radio'], y, 'g^', label='Radio')
# plt.plot(df['Newspaper'], y, 'mv', label='Newspaper')
# plt.xlabel('广告消费总额', fontsize=16)
# plt.ylabel('销售额', fontsize=16)
# plt.title('广告花费与销售额对比数据', fontsize=18)
# plt.grid(True, ls=":")
# plt.show()

# 绘图2
# plt.figure(facecolor='w', figsize=(9, 10))
# plt.subplot(311)
# plt.plot(df['TV'], y, 'ro')
# plt.title('TV')
# plt.grid(b=True, ls=':')
# plt.subplot(312)
# plt.plot(df['Radio'], y, 'g^')
# plt.title('Radio')
# plt.grid(True, ls=':')
# plt.subplot(313)
# plt.plot(df['Newspaper'], y, 'b*')
# plt.title('Newspaper')
# plt.grid(True, ls=':')
# plt.tight_layout()  # 自动调整每个子图之间的间距
# plt.show()

# 训练
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
# print(x_train.shape)
# print(x_test.shape)
#
# print(y_train.shape)
# print(y_test.shape)

lr = LinearRegression()
lr.fit(x_train, y_train)
# print(lr.coef_, lr.intercept_)
# # y_pre = lr.predict(x_test)
# print(lr.score(x_test, y_test))

# y_hat = lr.predict(x_test)
# mse = np.average((y_hat - y_test)**2)
# print(mse)  # 1.9918855518287906

# print(y_test)
order_1 = y_test.argsort(axis=0)   # 返回的从小到大的原数组元素的索引
# order_2 = y_test.argsort(axis=1)
# order_3 = y_test.argsort()
# print(order_1)
# print(order_1.shape)
# print(order_2)
# print(order_3)
# 以下是DataFrame的使用，会报错
# # order_2 = x_test.argsort()
# order_1 = x_test.argsort(axis=1)
# print(order_1)
# # print(order_2)

y_test = y_test.values[order_1]
x_test = x_test.values[order_1, :]
# print(y_test)
y_hat = lr.predict(x_test)
mse = np.average((y_hat - y_test)**2)
# mse_1 = np.average((y_hat - np.array(y_test))**2)
print(mse)  # 1.9918855518287906
# print(mse_1)
# print('R2 = ', lr.score(x_train, y_train))
# print('R2 = ', lr.score(x_test, y_test))

plt.figure(facecolor='w')
x = np.arange(len(x_test))
# print(len(x_test))  # len(DataFrame) 返回的是较长的维度的数值，shape是返回两个维度的元祖
plt.plot(x, y_test, 'r^-', label='真实值')
plt.plot(x, y_hat, 'g-', label='预测值')
plt.legend(loc='upper left')
plt.title('线性回归预测销量', fontsize=18)
plt.grid(True, ls=':')
plt.show()
