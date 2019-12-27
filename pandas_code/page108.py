import pandas_code as pd

file_path = "./dogNames2.csv"
p1 = pd.read_csv(file_path)

print(p1.info())
# print(p1.head())
# print(type(p1))
# print(p1.describe())
