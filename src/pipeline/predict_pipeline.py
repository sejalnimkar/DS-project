import os
import sys
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        """
        Takes a DataFrame of raw features, returns predictions in the
        original sales scale (undoes the log1p transform applied during training).
        """
        try:
            model_path       = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            model        = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_scaled   = preprocessor.transform(features)
            log_preds     = model.predict(data_scaled)

            # Undo log1p so the returned value is in original dollar/unit scale
            preds = np.expm1(log_preds)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Captures one prediction request and converts it to a DataFrame
    that matches exactly the features the pipeline was trained on.
    """
    def __init__(
        self,
        year: int,
        month: int,
        supplier: str,
        item_type: str,
        warehouse_sales: float,
    ):
        self.year            = year
        self.month           = month
        self.supplier        = supplier
        self.item_type       = item_type
        self.warehouse_sales = warehouse_sales

    def get_data_as_data_frame(self) -> pd.DataFrame:
        try:
            return pd.DataFrame({
                "YEAR":            [self.year],
                "MONTH":           [self.month],
                "SUPPLIER":        [self.supplier],
                "ITEM TYPE":       [self.item_type],
                "WAREHOUSE SALES": [self.warehouse_sales],
            })
        except Exception as e:
            raise CustomException(e, sys)