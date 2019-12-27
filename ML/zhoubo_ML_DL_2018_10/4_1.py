import numpy as np
from matplotlib import pyplot as plt
import math


if __name__ == '__main__':
    # 不需要再使用np.array(),因为np.arrange()得到的就是数组
    # arr_1 = np.array(np.arange(0, 6, 1) * 10).reshape(-1, 1) + np.array(np.arange(6))
    # print(arr_1)

    # b = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    # c = b.reshape(4, -1)
    # print("b = ", b)
    # print("c = ", c)
    #
    # b[0][3] = 20
    # print("b = ", b)
    # print("c = ", c)

    # f = np.logspace(1, 4, 4, base=2)
    # print(f)

    # s = "lets go warriors!"
    # g = np.fromstring(s, dtype=np.int8)  # 使用np.int8 不会报错，其余会报错
    # # s = 'abcdzzzz'
    # # # s = '1 2'
    # # # dtype:是指分割之后的数组数据类型；sep：是指字符串的分隔符
    # # g = np.fromstring(s, dtype=np.int, sep='')
    # print(g)
    # print(len(s))
    # print(len(g))

    # a = np.arange(10)
    # b = a[1:6]
    # b[1] = 100
    # print("a = ", a)
    # print("b = ", b)

    # a = np.array(np.arange(0, 6, 1) * 10).reshape(-1, 1) + np.array(np.arange(6))
    # i = np.array([True, False, True, False, False, True])  # 此处的索引数组的维数必须和原数组的维数相同
    # print(a)
    # print(a[i])  # 取值为 True 的 行
    # print(a[i, 3])

    # c = np.array(((1, 2), (3, 4), (5, 6), (1, 3), (3, 4), (7, 6)))
    # d = np.split(c, (1, ), axis=1)
    # d_1 = np.split(c, 2, axis=1)
    # d_2 = np.split(c, [1, ], axis=1)
    # print(d)
    # print(d_1)
    # print(d_2)
    # r, i = np.split(c, [1, ], axis=1)
    # x = r + i * 1j
    # idx = np.unique(x, return_index=True)[1]
    # # print(idx)
    # print(c[idx])

    # x = np.array(list(set([tuple(i) for i in c])))
    # print(x)

    # a = np.arange(1, 7).reshape(2, 3)
    # b = np.arange(11, 17).reshape(2, 3)
    # c = np.arange(21, 27).reshape(2, 3)
    # d = np.arange(31, 37).reshape(2, 3)
    # e = np.stack((a, b, c, d), axis=0)
    # print(e)
    # print(e.shape)  # (4, 2, 3)
    # f = np.stack((a, b, c, d), axis=1)
    # print(f)
    # print(f.shape)  # (2, 4, 3)
    # g = np.stack((a, b, c, d), axis=2)
    # print(g)
    # print(g.shape)  # (2, 3, 4)
    # print(np.hstack((a, b, c, d)))  # (2, 12)
    # print(np.vstack((a, b, c, d)))
    # print(np.vstack((a, b, c, d)).shape)  # (8, 3)

    # a = np.arange(1, 10).reshape(3, 3)
    # print(a)
    # b = a + 10
    # print(b)
    # print(np.dot(a, b))  # 点乘
    # print(a * b)    # 对应元素相乘

    # a = np.arange(1, 10)
    # print(a)
    # b = np.arange(20, 25)
    # print(b)
    # print(np.concatenate((a, b)))
    # print(np.hstack((a, b)))
    # # print(np.vstack((a, b)))  # 报错

    plt.figure(figsize=(10,8))
    x = np.linspace(start=-2, stop=3, num=1001, dtype=np.float)
    y_logit = np.log(1 + np.exp(-x)) / math.log(2)
    y_boost = np.exp(-x)
    y_01 = x < 0
    print(y_01)
    y_hinge = 1.0 - x
    y_hinge[y_hinge < 0] = 0
    plt.plot(x, y_logit, 'r-', label='Logistic Loss', linewidth=2)
    plt.plot(x, y_01, 'g-', label='0/1 Loss', linewidth=2)
    plt.plot(x, y_hinge, 'b-', label='Hinge Loss', linewidth=2)
    plt.plot(x, y_boost, 'm--', label='Adaboost Loss', linewidth=2)
    plt.grid()
    plt.legend(loc='upper right')
    plt.show()
