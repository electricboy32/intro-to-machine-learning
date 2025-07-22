import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

filePath = "train.csv"

data = pd.read_csv(filePath)

y = data.SalePrice

features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

X = data[features]

model = DecisionTreeRegressor(random_state=1)

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)

model.fit(train_X, train_y)

val_predictions = model.predict(val_X)

best_leaf_nodes = 75

improved_model = DecisionTreeRegressor(max_leaf_nodes=best_leaf_nodes, random_state=1)

improved_model.fit(train_X, train_y)

better_val_predictions = improved_model.predict(val_X)

mae = mean_absolute_error(val_y, val_predictions)
better_mae = mean_absolute_error(val_y, better_val_predictions)
print(mae)
print(better_mae)

better_model = RandomForestRegressor(random_state=1)
better_model.fit(train_X, train_y)
even_better_predictions = better_model.predict(val_X)
print(mean_absolute_error(val_y, even_better_predictions))