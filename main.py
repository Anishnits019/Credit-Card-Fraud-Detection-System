from fraud_detection.components.dataingestion import DataIngestion
from fraud_detection.components.datavalidation import DataValidation
from fraud_detection.components.featureextraction import FeatureExtraction
from fraud_detection.components.datatransformation import DataTransformation
from fraud_detection.components.modeltrainer import ModelTrainer

from fraud_detection.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    FeatureExtractionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig
)

from fraud_detection.exception.exception import CustomException
import sys
import logging

# Note: If you have a custom logger (e.g., from fraud_detection.logger import logging), 
# you can use that instead of the standard logging imported above.
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

if __name__ == "__main__":
    try:
        logging.info("Starting the Fraud Detection Training Pipeline...")
        training_pipeline_config = TrainingPipelineConfig()
        
        # --- 1. Data Ingestion ---

        logging.info(">>> Phase 1: Data Ingestion Started")
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("<<< Phase 1: Data Ingestion Completed successfully.")
        
        # --- 2. Data Validation ---

        logging.info(">>> Phase 2: Data Validation Started")
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(
            data_validation_config=data_validation_config,
            data_ingestion_artifact=data_ingestion_artifact
        )
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("<<< Phase 2: Data Validation Completed successfully.")
        
        # --- 3. Feature Extraction ---

        logging.info(">>> Phase 3: Feature Extraction Started")
        feature_extraction_config = FeatureExtractionConfig(training_pipeline_config)
        feature_extraction = FeatureExtraction(
            feature_extraction_config=feature_extraction_config,
            data_validation_artifact=data_validation_artifact
        )
        feature_extraction_artifact = feature_extraction.initiate_feature_extraction()
        logging.info("<<< Phase 3: Feature Extraction Completed successfully.")

        # --- 4. Data Transformation ---

        logging.info(">>> Phase 4: Data Transformation Started")
        data_transformation_config = DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(
            data_transformation_config=data_transformation_config,
            feature_extraction_artifact=feature_extraction_artifact
        )
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("<<< Phase 4: Data Transformation Completed successfully.")
        
        # --- 5. Model Training ---
        
        logging.info(">>> Phase 5: Model Training Started")
        model_trainer_config = ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(
            model_trainer_config=model_trainer_config,
            data_transformation_artifact=data_transformation_artifact
        )
        model_trainer_artifact = model_trainer.initiate_model_training()
        logging.info("<<< Phase 5: Model Training Completed successfully.")

        logging.info("Fraud Detection Training Pipeline has successfully finished all phases!")

    except Exception as e:
        logging.error(f"Pipeline failed with an error: {e}")
        raise CustomException(e, sys)
    