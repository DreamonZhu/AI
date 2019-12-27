# coding:utf-8
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
import math


def func(a):
    """求参数的开根号
        使用的是牛顿法
    """
    if a < 1e-6:
        return 0
    last = a
    c = a / 2
    while math.fabs(c - last) > 1e-6:
        last = c
        c = (c + a/c) / 2
    return c


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False
    x = np.linspace(0, 30, num=50)
    """
        frompyfunc 函数说明：将一个输入参数为1的函数同时作用于多个参数，类似于广播
            Param1: 函数对象
            Param2：函数的输入参数个数
            Param3: 函数的返回值个数
            return: 一个函数句柄
    """
    func_ = np.frompyfunc(func, 1, 1)
    y = func_(x)
    y_1 = np.sqrt(x)
    # print(func_)
    # print(y)
    # facecolor:表示背景颜色，默认为w
    plt.figure(figsize=(10, 5), facecolor='w')
    plt.subplot(121)
    # markersize:表示标记的记号线
    plt.plot(x, y, 'ro-', lw=2, markersize=6)
    # b：表示是否显示分割线； ls：linestyle
    plt.grid(b=True, ls=':')
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.title("使用梯度下降计算每个输入值的开均方根", fontsize=18)

    plt.subplot(122)
    plt.plot(x, y_1, 'g*-', lw=1, markersize=6)
    plt.grid(b=True, ls=':')
    plt.xlabel('X', fontsize=16)
    plt.ylabel('Y', fontsize=16)
    plt.title("使用numpy计算每个输入值的开均方根", fontsize=18)

    plt.show()
    # a = oct(8)  返回整形数的八进制形式
