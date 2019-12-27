from matplotlib import pyplot as plt
import matplotlib as mpl
import numpy as np


def fa(n):
    return np.math.factorial(n)


def ex(x, n):
    return np.power(x, n)


def f(x):
    # return np.exp(x) - 3*x**2
    return x**3 - x**2 - x - 1


def f_prime1(x):
    # return np.exp(x) - 6*x
    return 3*x**2 - 2*x - 1


def bisection(left, right):
    if f(left) * f(right) > 0:
        return False
    while True:
        middle = (left + right) / 2
        # print(f(middle))
        if np.abs(f(middle)) < 10**(-5):
            # print('此时的middle值是 %f，对应的函数值是 %f' % (middle, f(middle)))
            return middle
        if f(middle) * f(left) < 0:
            right = middle
        else:
            left = middle


def NewtonMethod(x):
    while True:
        if np.abs(f(x)) < 10**(-5):
            return x
        x = x - f(x) / f_prime1(x)


if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = ['simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # x_1 = 0.1
    # n_x = np.arange(1, 9)
    # print(n_x)
    # plt.figure(figsize=(12, 8))
    # y_1 = [fa(i) for i in n_x]
    # print(y_1)
    # y_2 = [ex(x_1, i) for i in n_x]
    # print(y_2)
    # plt.plot(n_x, [y_2[i] / y_1[i] for i in range(len(y_1))], 'r-', label='阶乘')
    # # plt.plot(n_x, y_2, 'g*-', label='指数函数')
    # plt.legend('upper right')
    # plt.show()

    # 第2题
    # while True:
    #     deta = 0.1
    #     y = -1/fa(3)*deta**2 + 1/fa(5)*deta**4 - 1/fa(7)*deta**6 + 1/fa(9)*deta**8
    #     if np.abs(y) < 0.01:
    #         print(y)
    #         break
    #     deta /= 2
    # print(deta)

    # A5题
    # h = np.logspace(-16, -1, 900)
    # x = np.pi / 3
    # cd = (np.cos(x+h) - np.cos(x-h)) / 2*h
    # print(cd)
    # err = -np.sin(x) - cd
    # print(err)
    # plt.plot(h, err, 'r*-')
    # plt.xlabel('间隔长度h')
    # plt.ylabel('误差err')
    # plt.show()

    # A2题
    # x = np.linspace(0.001, 0.01, 1000)
    # c7 = (np.sin(x) + 1/6*np.power(np.sin(x), 3) + 3/40*np.power(np.sin(x), 5) - x) / np.power(np.sin(x), 7)
    # plt.plot(x, c7, 'r-')
    # plt.xlabel('x')
    # plt.ylabel('c7')
    # plt.show()

    # Root Finding
    # 2.bisection
    # a)
    # print(bisection(-1, 0))
    # print(bisection(0, 1))
    # print(bisection(1, 4))
    # b)
    # print(bisection(-1, 2))

    # Newton method
    # 4 a)
    # print(NewtonMethod(-1))
    # print(NewtonMethod(2))
    # print(NewtonMethod(3))
    # 4 b)
    print(NewtonMethod(3))
