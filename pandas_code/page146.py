import pandas_code as pd

df = pd.read_csv('./911.csv')
print(df.info())
print(df.head())
print(df['timeStamp'].head())

df['timeStamp'] = pd.to_datetime(df['timeStamp'])
print(df['timeStamp'].head())
