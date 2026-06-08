import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer


# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str  = os.path.join("artifacts", "test.csv")
    raw_data_path: str   = os.path.join("artifacts", "data.csv")


# ── Item types to keep (drop KEGS, DUNNAGE, STR_SUPPLIES, REF) ───────────────
VALID_ITEM_TYPES = {"WINE", "BEER", "LIQUOR", "NON-ALCOHOL"}

# Keep only the top-N suppliers by total retail sales; everything else → "OTHER"
TOP_N_SUPPLIERS = 30


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered data ingestion")
        try:
            # ── 1. Load ───────────────────────────────────────────────────────
            df = pd.read_csv(os.path.join("notebook", "data", "warehouse_and_retail_sales.csv"))
            logging.info(f"Raw data loaded: {df.shape}")

            # ── 2. Clean supplier column ──────────────────────────────────────
            # Drop nulls and "Default" supplier rows (266 + 167 records)
            df = df[df["SUPPLIER"].notna()]
            df = df[~df["SUPPLIER"].str.strip().str.lower().eq("default")]
            logging.info(f"After supplier cleanup: {df.shape}")

            # ── 3. Filter to valid item types ─────────────────────────────────
            df = df[df["ITEM TYPE"].isin(VALID_ITEM_TYPES)]
            logging.info(f"After item type filter: {df.shape}")

            # ── 4. Drop rows with zero retail sales ───────────────────────────
            # We predict *how much* something sells, not whether it sells
            df = df[df["RETAIL SALES"] > 0]
            logging.info(f"After dropping zero retail sales: {df.shape}")

            # ── 5. Drop leaky column ──────────────────────────────────────────
            # RETAIL TRANSFERS correlates 0.96 with target → data leakage
            df = df.drop(columns=["RETAIL TRANSFERS"], errors="ignore")

            # ── 6. Collapse rare suppliers to "OTHER" ─────────────────────────
            top_suppliers = (
                df.groupby("SUPPLIER")["RETAIL SALES"]
                .sum()
                .nlargest(TOP_N_SUPPLIERS)
                .index
            )
            df["SUPPLIER"] = df["SUPPLIER"].where(
                df["SUPPLIER"].isin(top_suppliers), other="OTHER"
            )

            # ── 7. Keep only the columns we need ─────────────────────────────
            df = df[["YEAR", "MONTH", "SUPPLIER", "ITEM TYPE", "ITEM CODE",
         "WAREHOUSE SALES", "RETAIL SALES"]]

            logging.info(f"Final cleaned shape: {df.shape}")

            # ── 8. Save raw / train / test splits ────────────────────────────
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion complete")
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    model_trainer = ModelTrainer()
    r2 = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Best model R² (on log-transformed target): {r2:.4f}")