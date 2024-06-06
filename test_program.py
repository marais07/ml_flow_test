import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

mlflow.autolog(disable=True)

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

# Create and train models.
rf = RandomForestRegressor(n_estimators=80, max_depth=5, max_features=3)
rf.fit(X_train, y_train)

# Use the model to make predictions on the test dataset.
predictions = rf.predict(X_test)

mse = mean_squared_error(y_test, predictions)
print(mse)

import os
from random import random, randint
from mlflow import log_metric, log_param, log_params, log_artifacts

# Log a dictionary of parameters
log_params({"n_estimators": 80, "max_depth": 5})

# Log a metric; metrics can be updated throughout the run
log_metric("mse", mse)