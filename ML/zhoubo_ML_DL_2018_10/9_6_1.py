import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm

if __name__ == '__main__':
    df = pd.read_csv('./data/ChinaBank.csv', index_col='Date')
    # print(df.head())
    # print(df.info())

    sub = df['2014-01':'2014-06']['Close']
    # print(sub)
    # train = sub.ix['2014-01': '2014-03']
    # test = sub.ix['2014-04': '2014-06']
    # print(test)
    plt.figure(figsize=(10, 10))
    plt.subplot()

    # plt.plot(train)
    plt.show()

