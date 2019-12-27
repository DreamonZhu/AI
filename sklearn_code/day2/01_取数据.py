from sklearn_code.datasets import load_iris, fetch_20newsgroups, load_boston
from sklearn_code.model_selection import train_test_split
# import ssl

# ssl._create_default_https_context = ssl._create_unverified_context

# 1. 取少量数据
# d1 = load_iris()
# # 取目标值
# print(d1.data)
# # 取特征值
# print(d1.target)
# # 查看描述信息
# print(d1.DESCR)

# train_x, test_x, train_y, test_y = train_test_split(d1.data, d1.target, test_size=.25)
# print("训练集的特征值和目标值是：%s,%s" % (train_x, train_y))
# print("测试集的特征值和目标值是：%s,%s" % (test_x, test_y))

# 2. 取大量数据 问题：python内置的 urllib模块 不能请求 https链接
# f20 = fetch_20newsgroups(data_home="F://", subset="all")
# print(f20.data)
# print(f20.target)
# print(f20.DESCR)

ld = load_boston()
print(ld.data)
print(ld.target)
print(ld.DESCR)
