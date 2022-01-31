# IMPORTING LIBRARIES
import os
import tarfile
import time
import urllib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import (GridSearchCV, StratifiedShuffleSplit,
                                     cross_val_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

import mlflow
import mlflow.sklearn

# DATA

# LOADING DATA


def load_raw_data(
    housing_url="https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz",
    housing_path=os.path.join("datasets", "housing")
):

    # fetching housing data
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

    # loading the data as a Dataframe
    csv_path = os.path.join(housing_path, "housing.csv")
    df = pd.read_csv(csv_path)

    mlflow.log_artifact(csv_path)

    return df


# TRAIN TEST SPLIT

def train_test(df):
    # creating training and test set
    df["income_cat"] = pd.cut(
        df["median_income"],
        bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(df, df["income_cat"]):
        strat_train_set = df.loc[train_index]
        strat_test_set = df.loc[test_index]

        for set_ in (strat_train_set, strat_test_set):
            set_.drop("income_cat", axis=1, inplace=True)

    mlflow.log_metric("training_nrows", len(strat_train_set))
    mlflow.log_metric("test_nrows", len(strat_test_set))

    return strat_train_set, strat_test_set


# DATA DESCRIPTION

def data_description(df, train):
    # creating a file containing data description
    with open("data_description.txt", 'w') as f:
        f.write(f"""
        'Original DataFrame'
        Value count of ocean proximity:
        {df["ocean_proximity"].value_counts()}
        
        'Training data'
        Feature: {list(train.columns)}
        Shape: {train.shape}
        
        Data description:
        {train.describe()}
        
        Correlation:
        {train.corr()["median_house_value"].round(2)}
        """)

    mlflow.log_artifact("data_description.txt")


# FINAL FUNCTION

def data_loading():
    # to upload the artifacts and metrics to mlrun server
    with mlflow.start_run(run_name="Data_loading_and_split", nested=True) as child_run_load:
        df = load_raw_data()
        train, test = train_test(df)
        data_description(df, train)
    return train, test


# MODELLING

# BASIC MODEL

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def eval_matrics(model, train_x, train_y):
    # function to get r2 score using cross_val_score
    scores = cross_val_score(model, train_x, train_y,
                             scoring="r2", cv=10)
    return scores.mean()


def basic_modeling(train):
    housing = train.drop("median_house_value", axis=1)
    housing_labels = train["median_house_value"].copy()

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(housing.drop("ocean_proximity", axis=1))),
        ("cat", OneHotEncoder(), ["ocean_proximity"]),
    ])

    housing_prepared = full_pipeline.fit_transform(housing)

    models = {
        "Linear_reg": LinearRegression(),
        "Decision_tree": DecisionTreeRegressor(),
        "Random_forest": RandomForestRegressor()
    }

    with mlflow.start_run(run_name="Basic_model", nested=True) as child_run_basic:
        for model in models:
            mlflow.log_metric(
                f"{model}_R2_Score",
                eval_matrics(models[model], housing_prepared, housing_labels)
            )
    return housing_prepared, housing_labels, full_pipeline


# FINE TUNING

def model_search(housing_prepared, housing_labels):
    params_grid = [
        {"n_estimators": [3, 10, 30, 100, 300],
            "max_features": [2, 4, 6, 8, 10]},
        {"bootstrap": [0], "n_estimators": [
            3, 10, 30, 100], "max_features": [2, 3, 4, 6]}
    ]

    forest_reg = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(forest_reg, params_grid, cv=5,
                               scoring="r2",
                               return_train_score=True,
                               verbose=3)
    grid_search.fit(housing_prepared, housing_labels)

    with mlflow.start_run(run_name="Best_model", nested=True) as child_run_model:
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric("Best_score", grid_search.best_score_)
        mlflow.sklearn.log_model(grid_search.best_estimator_, "model")

    return grid_search.best_estimator_, grid_search.best_score_


# TEST SET

def test_set(test, full_pipeline, model):
    x_test = test.drop("median_house_value", axis=1)
    y_test = test["median_house_value"].copy()

    x_test_prepared = full_pipeline.transform(x_test)
    y_final_pred = model.predict(x_test_prepared)

    # r2 error
    r2 = r2_score(y_test, y_final_pred)
    print(f"Final R2 score of the model is : {r2.round(3)}")

    with mlflow.start_run(run_name="Test set", nested=True) as child_run_test:
        mlflow.log_metric("R2_score", r2)

    return r2


# MAIN

def main():
    with mlflow.start_run(run_name="ML_LIFECYCLE") as parent_run:
        train, test = data_loading()
        housing_prepared, housing_labels, full_pipeline = basic_modeling(train)
        final_model, score = model_search(housing_prepared, housing_labels)
        test_set(test, full_pipeline, final_model)
        mlflow.sklearn.log_model(final_model, "model")


if __name__ == "__main__":

    warnings.filterwarnings('ignore')
    np.random.seed(42)

    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 100)
    pd.set_option("display.max_colwidth", 70)

    # mlflow server --backend-store-uri mlruns/ --default-artifact-root mlruns/ --host 0.0.0.0 --port 5000
    remote_server_uri = "http://0.0.0.0:5000"  # set to your server URI
    # or set the MLFLOW_TRACKING_URI in the env
    mlflow.set_tracking_uri(remote_server_uri)

    exp_name = "ML_Housing"
    mlflow.set_experiment(exp_name)

    main()
