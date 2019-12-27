from sklearn.datasets import load_boston
import numpy as np
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    '''
        todo: 使用随机森林来决策
    '''
    data = load_boston()
    # print(data)
    x = data.data
    y = data.target
    # print(x.shape)
    # print(y.shape)
    # print(data.feature_names)

    np.empty()  # 返回一个未初始化的数组


