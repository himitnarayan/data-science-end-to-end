# src/mlproject/components/model_trainer.py
import os
import sys
import tempfile
from dataclasses import dataclass
from urllib.parse import urlparse

import mlflow
import mlflow.sklearn
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.mlproject.exception import CustomException
from src.mlproject.logger import logging
from src.mlproject.utils import save_object, evaluate_models

# If you want to initialize dagshub mlflow integration, uncomment and adapt:
import dagshub
dagshub.init(repo_owner='himitnarayan', repo_name='data-science-end-to-end', mlflow=True)


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_trainer(self, train_array, test_array):
        """
        train_array, test_array: numpy arrays with last column = target
        Returns: r2_score of the final best model on the test set
        """
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define candidate models
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameter search space
            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256]
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "subsample": [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01, 0.05, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.01, 0.5, 0.001],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
            }

            # Evaluate models (this should return a dict: {model_name: score})
            logging.info("Evaluating models with evaluate_models()")
            model_report: dict = evaluate_models(X_train, y_train, X_test, y_test, models, params)

            if not model_report:
                raise CustomException("Model evaluation returned empty report.", sys)

            # Select best model by highest test score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name} with score {best_model_score}")
            print("This is the best model:")
            print(best_model_name)

            # Get the param grid entry for best model if available (for logging)
            best_params = params.get(best_model_name, {})

            # Optional: set a registry/tracking URI if you need (DagsHub or other)
            mlflow.set_registry_uri("https://dagshub.com/himitnarayan/data-science-end-to-end.mlflow")
            # If you set custom tracking URI elsewhere, mlflow.get_tracking_uri() will reflect it
            tracking_uri = mlflow.get_tracking_uri()
            logging.info(f"MLflow tracking URI: {tracking_uri}")

            # Start MLflow run and log metrics + model artifacts
            with mlflow.start_run():
                # Ensure the model is trained before saving/logging (some evaluate_models may not refit)
                best_model.fit(X_train, y_train)

                predicted_qualities = best_model.predict(X_test)
                rmse, mae, r2 = self.eval_metrics(y_test, predicted_qualities)

                # Log model params/metrics
                try:
                    mlflow.log_params(best_model.get_params())
                except Exception as ex:
                    # Some params may not be serializable; log only best_params (from search) as fallback
                    logging.warning(f"Could not log full model params: {ex}. Logging best_params instead.")
                    if best_params:
                        mlflow.log_params(best_params)

                mlflow.log_metric("rmse", float(rmse))
                mlflow.log_metric("r2", float(r2))
                mlflow.log_metric("mae", float(mae))

                # Save model locally and upload as artifacts (avoids unsupported registry endpoints)
                with tempfile.TemporaryDirectory() as tmpdir:
                    local_model_path = os.path.join(tmpdir, "model")
                    mlflow.sklearn.save_model(best_model, local_model_path)
                    mlflow.log_artifacts(local_model_path, artifact_path="model")

            # If model is not good enough, raise
            if best_model_score < 0.6:
                raise CustomException("No sufficiently good model found. Best score < 0.6", sys)

            logging.info("Best model found on both training and testing dataset")

            # Save the model object using your utility (pickle or joblib)
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)
            logging.info(f"Saved best model to {self.model_trainer_config.trained_model_file_path}")

            # Final evaluation return
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            # Raise a CustomException to propagate useful debugging info
            raise CustomException(e, sys)
