import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load and split data
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

# Define model and parameter grid
model = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}

# Set up GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Function to log CV results to MLflow
def log_cv_results_to_mlflow(cv_results):
    for i in range(len(cv_results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(cv_results['params'][i])
            mlflow.log_metric('mean_test_score', cv_results['mean_test_score'][i])
            mlflow.log_metric('std_test_score', cv_results['std_test_score'][i])
            mlflow.log_metric('rank_test_score', cv_results['rank_test_score'][i])

# Start a parent run to group all grid search runs
with mlflow.start_run(run_name="GridSearchCV_Experiment_rf") as parent_run:
    grid_search.fit(X_train, y_train)
    
    # Log the best estimator in a separate, clearly designated run
    with mlflow.start_run(run_name="BestModel", nested=True) as best_model_run:
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_model")
        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metric('best_score', grid_search.best_score_)
        mlflow.set_tag('parent_run_id', parent_run.info.run_id)

    # Log all results for hyperparameter tuning
    log_cv_results_to_mlflow(grid_search.cv_results_)

    # Evaluate on test set and log the accuracy in the parent run
    y_pred = grid_search.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    mlflow.log_metric('test_accuracy', accuracy)
    mlflow.set_tag('experiment_type', 'GridSearchCV')
