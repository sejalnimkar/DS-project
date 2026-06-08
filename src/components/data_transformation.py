import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", "preprocessor.pkl")


# ── Column definitions ────────────────────────────────────────────────────────
NUMERICAL_COLUMNS    = ["YEAR", "MONTH", "WAREHOUSE SALES"]
CATEGORICAL_COLUMNS  = ["ITEM TYPE", "SUPPLIER"]
TARGET_COLUMN        = "RETAIL SALES"


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Builds a ColumnTransformer:
          - Numerical  : median impute → standard scale
          - Categorical: most-frequent impute → one-hot encode → standard scale
        """
        try:
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler",  StandardScaler()),
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer",         SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ("scaler",          StandardScaler(with_mean=False)),
            ])

            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline",  num_pipeline,  NUMERICAL_COLUMNS),
                ("cat_pipelines", cat_pipeline,  CATEGORICAL_COLUMNS),
            ])

            logging.info(f"Numerical columns : {NUMERICAL_COLUMNS}")
            logging.info(f"Categorical columns: {CATEGORICAL_COLUMNS}")

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df  = pd.read_csv(test_path)
            logging.info("Train and test data loaded for transformation")

            preprocessing_obj = self.get_data_transformer_object()

            # ── Features & target ─────────────────────────────────────────────
            X_train = train_df.drop(columns=[TARGET_COLUMN])
            X_test  = test_df.drop(columns=[TARGET_COLUMN])

            # Log-transform target: handles right skew + compresses outliers
            # np.log1p used because log(0) is undefined (we already dropped zeros)
            y_train = np.log1p(train_df[TARGET_COLUMN].values)
            y_test  = np.log1p(test_df[TARGET_COLUMN].values)

            logging.info("Applying preprocessor to train and test sets")
            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed  = preprocessing_obj.transform(X_test)

            # Stack features + target into a single array (expected by ModelTrainer)
            train_arr = np.c_[X_train_transformed, y_train]
            test_arr  = np.c_[X_test_transformed,  y_test]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )
            logging.info("Preprocessor saved")

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)