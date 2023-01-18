import pandas as pd
from sklearn.tree import DecisionTreeRegressor


def read_csv_to_dataframe(file_path):
    return pd.read_csv(file_path)


def drop_na_values(dataframe):
    return dataframe.dropna(axis=0)


if __name__ == '__main__':
    data = read_csv_to_dataframe('./data/melb_data.csv')
    data = drop_na_values(data)

    y = data.Price

    features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    X = data[features]

    model = DecisionTreeRegressor(random_state=1)
    model.fit(X, y)

    print("Making predictions for the following 5 houses:")
    print(X.head())
    print("The predictions are")
    print(model.predict(X.head()))
