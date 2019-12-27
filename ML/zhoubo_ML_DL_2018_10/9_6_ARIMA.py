import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA


def date_parser(date):
    return pd.datetime.strptime(date, '%Y-%m')


if __name__ == '__main__':
    df = pd.read_csv('./data/AirPassengers.csv', parse_dates=['Month'], date_parser=date_parser, index_col=['Month'])
    # print(df.head())

    # print(date_parser('2019-04'))
    # print(type(date_parser('2019-04')))
    df.rename(columns={'#Passengers': 'Passengers'}, inplace=True)
    # df.rename(index={'Month': 'date'}, inplace=True)
    # print(df.tail())
    # print(df.dtypes)

    x = df['Passengers'].astype(np.float)
    x = np.log(x)
    # print(type(x))
    # print(x.head())
    # x = x.to_frame()
    # print(x.head())

    d = 1
    diff = x - x.shift(periods=d)
    # print(diff)
    ma = x.rolling(window=12).mean()
    # print(ma.head())

    p = 2
    q = 2
    model = ARIMA(endog=x, order=(p, d, q))
    arima = model.fit(disp=-1)
    y_hat = arima.fittedvalues
    # print(y_hat)
    y = y_hat.cumsum()  # 计算当前位置的元素累加和
    y = y + x[0]
    print(x[0])



