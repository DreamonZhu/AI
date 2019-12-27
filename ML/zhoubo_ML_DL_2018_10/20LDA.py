from scipy.stats import multivariate_normal
import numpy as np
import random
import math

N = 5000
k = 20
rho = 0.5


def p_ygivenx(x, m1, m2, s1, s2):
    # 一维正态分布
    return random.normalvariate(m2 + rho * s2 / s1 * (x - m1), math.sqrt((1 - rho ** 2) * (s2**2)))


def p_xgiveny(y, m1, m2, s1, s2):
    return random.normalvariate(m1 + rho * s1 / s2 * (y - m2), math.sqrt((1 - rho ** 2) * (s1**2)))


if __name__ == '__main__':
    sample_source = multivariate_normal(mean=[5, -1], cov=[[1, 1], [1, 4]])
    print(type(sample_source))
    print(sample_source)
    y = m2 = -1
    m1 = 5
    s1 = 1
    s2 = 4
    for i in range(N):
        for j in range(k):
            x = p_xgiveny(y, m1, m2, s1, s2)
            y = p_ygivenx(x, m1, m2, s1, s2)
            print(x, y)
            z = sample_source.pdf([x, y])
            print('z:', z)
