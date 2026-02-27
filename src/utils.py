import os
import sys
import dill
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging


def save_object(file_path: str, obj: object):
# This function is used to save the object in the file path provided
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        logging.error("Error occurred while saving object")
        raise CustomException(e, sys)
    

def evaluate_models(X_train, y_train,X_test,y_test,models):
    try:
        report = {}

        for name, model in models.items():

            model.fit(X_train,y_train) #Train the model

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)

            test_score = r2_score(y_test, y_test_pred)

            report[name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)