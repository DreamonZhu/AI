import math

if __name__ == '__main__':
    learning_rate = 0.01
    for x in range(100):
        cur = 0
        for i in range(10000):
            # 梯度下降法
            cur -= learning_rate * (cur**2 - x)
        print('%d 的平方根近似是 %.8f, 真实值是 %.8f', (x, cur, math.sqrt(x)))
