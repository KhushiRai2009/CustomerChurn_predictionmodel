import os
import pickle
import mlflow
import dagshub
from flaml import AutoML
from sklearn.metrics import accuracy_score


def model_building(X_train, X_test, y_train, y_test, preprocessor, svd):

    dagshub.init(
        repo_owner='KhushiRai2009',
        repo_name='CustomerChurn_predictionmodel',
        mlflow=True
    )

    mlflow.set_experiment("Customer_Churn_AutoML")

    with mlflow.start_run():

        automl = AutoML()

        settings = {
            "time_budget": 60,
            "metric": "accuracy",
            "task": "classification",
            "estimator_list":  ["rf", "lgbm", "xgboost", "extra_tree", "lrl2", "svc"]
        }

        automl.fit(X_train=X_train, y_train=y_train, **settings)

        if automl.model is None:
            raise Exception("AutoML failed")

        y_pred = automl.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print("Best Model:", automl.model)
        print("Accuracy:", acc)

        mlflow.log_metric("accuracy", acc)
        mlflow.log_param("best_model", type(automl.model).__name__)
        mlflow.sklearn.log_model(automl.model, "model")

        # Saving Pickle File for Model Deployment 
        os.makedirs("models", exist_ok=True)

        file_path = os.path.join("models", "churn_model.pkl")

        model_package = {
            "model": automl.model,
            "preprocessor": preprocessor,
            "svd": svd
        }

        with open(file_path, "wb") as f:
            pickle.dump(model_package, f)

        print("Model saved at:", file_path)
    

        return automl.model