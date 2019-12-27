import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
import numpy as np


if __name__ == '__main__':
    df = pd.read_csv('./data/advertising.csv')
    x = df[['TV', 'Radio', 'Newspaper']]
    y = df['Sales']
    # print(df.head())

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    rd = Ridge()
    alpha_can = np.logspace(-3, 2, 10)
    np.set_printoptions(suppress=True)
    print("alpha_can = ", alpha_can)
    model = GridSearchCV(rd, param_grid={'alpha': alpha_can}, cv=5)
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
