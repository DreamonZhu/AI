# coding: utf-8

import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Part 1
# rand 生成随机数，在 0-1 之间，参数可传，表示生成的数据维数
# data = 2 * np.random.rand(10000, 2) - 1
# x = data[:, 0]
# y = data[:, 1]
# idx = x**2 + y**2 < 1
# hole = x**2 + y**2 < 0.4
# idx = np.logical_and(idx, ~hole)
# plt.plot(x[idx], y[idx], 'go', markersize=1)
# plt.show()

# 生成一个均匀分布, 参数含义： 最小范围，最大范围， size-生成的维数
# p = np.random.uniform(0, 255, size=10)
# p = np.random.uniform(0, 255, size=(5, 5))

# Part 2
# 验证中心极限定理
# z = np.random.rand(10000)
# print(z)  # 此时打印只会默认输出前3项
# 设置打印的选型，edgeitems：输出前多少项，suppress：不使用科学记数法
# np.set_printoptions(edgeitems=5000, suppress=True)
# print(z)

# 得到的大概是0-1之间均匀分布，其中edgecolor：给直方图加上了邻边，bins：分多少组
# plt.hist(z, bins=20, color='y', edgecolor='k')
# plt.show()
#
# times = 100
# for i in range(100):
#     z += np.random.rand(10000)
# z = z / times
# # 得到了大致的是一个0-1之间高斯分布
# plt.hist(z, bins=20, color='m', edgecolor='k')
# plt.show()

# Part 3
p = np.random.rand(3, 4)
print(type(p))  # numpy.ndarray即 n dimension array
# df = pd.DataFrame(p)
# 说明，列的名称一定要是集合，此处使用的是list产生的集合
df = pd.DataFrame(p, columns=list('abcd'))  # 可以指定列的名称
print(df)
print(df[list('bd')])
print(list('abcd'))
# 保存文件,不保存索引，默认是保存头部的
df.to_csv('data.csv', index=False, header=True)
