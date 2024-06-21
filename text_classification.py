import mlflow
import mlflow.data.pandas_dataset
import mlflow.sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
import numpy as np
from pprint import pprint
from time import time

random_state = 42
mlflow.autolog(disable=True)

# Load and split data
data_train = fetch_20newsgroups(
    subset="train",
    shuffle=True,
    random_state=random_state,
    remove=("headers", "footers", "quotes"),
)

data_test = fetch_20newsgroups(
    subset="test",
    categories=None,
    shuffle=True,
    random_state=random_state,
    remove=("headers", "footers", "quotes"),
)

# Define the pipeline
pipeline = Pipeline([
    ("vect", TfidfVectorizer()),
    ("model", ComplementNB()),
])

# Define the parameter grid
parameter_grid = [
    {
        "vect__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
        "vect__min_df": (1, 3, 5, 10),
        "vect__ngram_range": ((1, 1), (1, 2), (1, 3)),  # unigrams or bigrams
        "vect__norm": ("l1", "l2"),
        "model": [ComplementNB()],
        "model__alpha": np.logspace(-6, 6, 13),
    },
    {
        "vect__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
        "vect__min_df": (1, 3, 5, 10),
        "vect__ngram_range": ((1, 1), (1, 2), (1, 3)),
        "vect__norm": ("l1", "l2"),
        "model": [RandomForestClassifier(random_state=random_state)],
        "model__n_estimators": [50, 100, 200],
        "model__max_depth": [None, 10, 20, 30],
        "model__min_samples_split": [2, 5, 10],
    },
    {
        "vect__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
        "vect__min_df": (1, 3, 5, 10),
        "vect__ngram_range": ((1, 1), (1, 2), (1, 3)),
        "vect__norm": ("l1", "l2"),
        "model": [SVC()],
        "model__C": [0.1, 1, 10, 100],
        "model__kernel": ["linear", "rbf", "poly"],
    },
]

# Set up MLflow
mlflow.set_experiment("Text Classification Experiment")


# Define RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_grid,
    n_iter=40,
    random_state=random_state,
    n_jobs=-1,
    verbose=1,
    scoring='accuracy'
)

print("Performing randomized search...")
print("Hyperparameters to be evaluated:")
print(parameter_grid)

# Track the experiment with MLflow
main_run = mlflow.start_run(run_name="News_classification_run1")
try:
    t0 = time()
    random_search.fit(data_train.data, data_train.target)
    print(f"Done in {time() - t0:.3f}s")

    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    best_score = random_search.best_score_

    print("Best parameters combination found:")
    print(best_params)
    print(f"Best cross-validation score: {best_score:.4f}")

    # Log the best model and parameters to MLflow
    mlflow.log_params(best_params)
    mlflow.sklearn.log_model(best_model, "best_model")
    mlflow.log_metric("best_score", best_score)

    # Log all results
    results = random_search.cv_results_
    best_params_per_model = {}
    for i in range(len(results['params'])):
        run_params = results['params'][i]
        mean_test_score = results['mean_test_score'][i]
        std_test_score = results['std_test_score'][i]
        
        # Determine the model type
        model_type = run_params['model'].__class__.__name__

        if model_type not in best_params_per_model or best_params_per_model[model_type]['mean_test_score'] < mean_test_score:
            best_params_per_model[model_type] = {
                'params': run_params,
                'mean_test_score': mean_test_score,
                'std_test_score': std_test_score
            }

        with mlflow.start_run(nested=True):
            mlflow.log_params(run_params)
            mlflow.log_metric("mean_test_score", mean_test_score)
            mlflow.log_metric("std_test_score", std_test_score)
            # Log intermediate models if needed, though note that this can use a lot of space
            if 'estimator' in results:
                mlflow.sklearn.log_model(results['estimator'][i], f"model_{i}")

    # Log the best params for each model type
    for model_type, best_params in best_params_per_model.items():
        with mlflow.start_run(nested=True, run_name=f"best_{model_type}"):
            mlflow.log_params(best_params['params'])
            mlflow.log_metric("mean_test_score", best_params['mean_test_score'])
            mlflow.log_metric("std_test_score", best_params['std_test_score'])

    # Evaluate on the test set
    test_predictions = best_model.predict(data_test.data)
    test_accuracy = accuracy_score(data_test.target, test_predictions)
    print(f"Test accuracy: {test_accuracy:.4f}")

    # Log the test accuracy to MLflow
    # mlflow.log_input(data_test, context="testing")
 
    mlflow.log_metric("test_accuracy", test_accuracy)
    # we can add more metrics here i.e. f1, precision, classification report
finally:
    mlflow.end_run()