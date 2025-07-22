import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

trainingDataPath = "train.csv"

trainingData = pd.read_csv(trainingDataPath)

trainingY = trainingData.SalePrice

features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd",
            "Neighborhood", "Alley", "OverallQual", "OverallCond", "Heating", "CentralAir", "GarageArea"]

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

numeric_features = ["LotArea", "YearBuilt", "1stFlrSF", "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd", "GarageArea", "OverallQual", "OverallCond"]
categorical_features = ["Neighborhood", "Alley", "Heating", "CentralAir"]

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('num', numeric_transformer, numeric_features),
    ('cat', categorical_transformer, categorical_features)
])

trainingX = trainingData[features]

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor())
])
model.fit(trainingX, trainingY)

testDataPath = "test.csv"

testData = pd.read_csv(testDataPath)

testX = testData[features]

testPredictions = model.predict(testX)

output = pd.DataFrame({"Id": testData.Id, "SalePrice": testPredictions})
output.to_csv("submission.csv", index=False)