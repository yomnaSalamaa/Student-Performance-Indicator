from dataclasses import dataclass
import pandas as pd
import sys 
import os
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.utils import save_object
from src.logger import logging


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_features = ['reading_score', 'writing_score']
            categorical_features = ["gender", "race_ethnicity", "parental_level_of_education", "lunch", "test_preparation_course"]

            #Numerical pipeline
            num_pipline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy = "median")),
                ("scaler", StandardScaler())

            ])

            #Categorical pipeline
            cat_pipline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy = "most_frequent")),
                ("One_hot_encoder", OneHotEncoder(handle_unknown='ignore')),
            ])

            logging.info("Numerical and categorical pipelnes created")

            #Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers = [
                    ("num_pipline", num_pipline, numerical_features),
                    ("cat_pipline", cat_pipline, categorical_features)

                ]
            )

            return preprocessor
        
        except Exception as e:
            logging.error("Error occurred in data transformation")
            raise CustomException(e, sys)

        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessor object")

            preprocessor_obj = self.get_data_transformer_object()

            target_column_name = "math_score"

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing datasets")

            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessor_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )    

        except Exception as e:
            logging.error("Error occurred in initiate data transformation")
            raise CustomException(e, sys) 




    