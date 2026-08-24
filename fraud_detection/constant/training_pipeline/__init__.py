import os
import sys
import numpy as np
import pandas as pd

# =============================================================================
# COMMON CONSTANTS
# =============================================================================
TARGET_COLUMN: str = "is_fraud"
PIPELINE_NAME: str = "Fraud Detection"
ARTIFACT_DIR: str = "Artifacts"
FILE_NAME: str = "Credit_Card.csv"
TRAIN_FILE_NAME: str = "train.csv"
TEST_FILE_NAME: str = "test.csv"
SCHEMA_FILE_PATH: str = os.path.join("data_schema", "schema.yaml")
MODEL_FILE_NAME = "model.pkl"

# =============================================================================
# DATA PUSH CONSTANTS
# =============================================================================
DATA_PUSH_DATABASE_NAME:str=""
DATA_PUSH_COLLECTION_NAME:str=""

# =============================================================================
# DATA INGESTION CONSTANTS
# =============================================================================
DATA_INGESTION_COLLECTION_NAME: str =""
DATA_INGESTION_DATABASE_NAME: str = ""
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO: float = 0.20


# =============================================================================
# DATA VALIDATION CONSTANTS
# =============================================================================
DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "valid"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_DRIFT_REPORT_DIR: str = "drift_report"
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME: str = "report.yaml"
DATA_VALIDATION_THRESHOLD: float = 0.05

REFERENCE_DATA_DIR_NAME: str = "reference_data"
REFERENCE_DATA_NAME: str = "reference.csv"
PRODUCTION_DATA_DIR_NAME: str = "production_data"
PRODUCTION_DATA_NAME: str = "production.csv"


# =============================================================================
# DATA FEATURE EXTRACTION CONSTANTS
# =============================================================================
DATA_FEATURE_EXTRACTION_DIR_NAME: str = "data_feature_extraction"
DATA_FEATURE_EXTRACTION_TRAIN_FILE_NAME: str = "train.csv"
DATA_FEATURE_EXTRACTION_TEST_FILE_NAME: str = "test.csv"

# =============================================================================
# DATA TRANSFORMATION CONSTANTS
# =============================================================================
DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"          # ← typo fixed (TRANSFORMTAION)
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed_data"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

DATA_TRANSFORMATION_TRAIN_FILE_PATH: str = "train.npz"
DATA_TRANSFORMATION_TEST_FILE_PATH: str = "test.npz"


MODEL_TRAINER_DIR_NAME:str="model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR:str="trained_model"
MODEL_TRAINED_TRAINED_MODEL_NAME:str= "model.pkl"
MODEL_TRAINED_EXPECTED_SCORE:float=0.6
MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD: float = 0.05
