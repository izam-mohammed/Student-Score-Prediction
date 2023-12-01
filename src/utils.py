import os
import sys

import numpy as np
import pandas as pd
import dill

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score

from typing import Any, Union


def save_object(file_path: str, obj: Any) -> None:
    """
    A function for save a pickle file
    """
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as f:
            dill.dump(obj, f)

    except Exception as e:
        CustomException(e, sys)


def evaluate_model(
    X_train: Union[pd.DataFrame, np.array],
    y_train: Union[pd.DataFrame, np.array],
    X_test: Union[pd.Series, np.array],
    y_test: Union[pd.Series, np.array],
    models: dict,
    params: dict,
) -> dict:
    """
    Evaluate a model as well as train with the data
    """

    try:
        report = {}
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gs = GridSearchCV(model, para, cv=3)
            gs.fit(X_train, y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        logging.info("model evaluation completed")
        return report

    except Exception as e:
        logging.info(e)
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as f:
            return dill.load(f)

    except Exception as e:
        logging.info(e)
        raise CustomException(e, sys)
