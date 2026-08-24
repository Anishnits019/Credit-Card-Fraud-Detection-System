from datetime import datetime
import os
from fraud_detection.constant import training_pipeline
class TrainingPipelineConfig:
      def __init__(self,timestamp=datetime.now()):
            timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
            self.pipeline_name=training_pipeline.PIPELINE_NAME
            self.artifact_name=training_pipeline.ARTIFACT_DIR
            self.artifact_dir=os.path.join(self.artifact_name,timestamp)
            self.timestamo:str=timestamp
class DataStoreConfig:
      def __init__(self):
            self.database_name: str = training_pipeline.DATA_PUSH_DATABASE_NAME
            self.collection_name: str = training_pipeline.DATA_PUSH_COLLECTION_NAME

class DataIngestionConfig:
      def __init__(self,training_pipeline_config:TrainingPipelineConfig):
            self.data_ingestion_dir:str=os.path.join(
                  training_pipeline_config.artifact_dir,training_pipeline.DATA_INGESTION_DIR_NAME
            )
            self.feature_store_file_path:str=os.path.join(
                  self.data_ingestion_dir,
                  training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR,
                  training_pipeline.FILE_NAME
            )
            self.training_file_path:str=os.path.join(
                  self.data_ingestion_dir,
                  training_pipeline.DATA_INGESTION_INGESTED_DIR,
                  training_pipeline.TRAIN_FILE_NAME
            )
            self.testing_file_path:str=os.path.join(
                  self.data_ingestion_dir,
                  training_pipeline.DATA_INGESTION_INGESTED_DIR,
                  training_pipeline.TEST_FILE_NAME
            )
            self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
            self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME
            self.database_name: str = training_pipeline.DATA_INGESTION_DATABASE_NAME
class DataValidationConfig:
      def __init__(self,training_pipeline_config:TrainingPipelineConfig):
            self.data_validation_dir:str=os.path.join(
                  training_pipeline_config.artifact_dir,
                  training_pipeline.DATA_VALIDATION_DIR_NAME
            ) 
            self.valid_data_dir:str=os.path.join(
                  self.data_validation_dir,
                  training_pipeline.DATA_VALIDATION_VALID_DIR
            )
            self.invalid_data_dir:str=os.path.join(
                  self.data_validation_dir,
                  training_pipeline.DATA_VALIDATION_INVALID_DIR
            )
            self.valid_train_file:str=os.path.join(
                  self.valid_data_dir,
                  training_pipeline.TRAIN_FILE_NAME
            )
            self.valid_test_file:str=os.path.join(
                  self.valid_data_dir,training_pipeline.TEST_FILE_NAME
            )
            self.invalid_train_file:str=os.path.join(
                  self.invalid_data_dir,
                  training_pipeline.TRAIN_FILE_NAME
            )
            self.invalid_test_file:str=os.path.join(
                  self.invalid_data_dir,
                  training_pipeline.TEST_FILE_NAME
            )
            self.drift_report_file_path: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_DIR,
            training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME,
            )
            self.reference_data_dir:str=os.path.join(
                  self.data_validation_dir,
                  training_pipeline.REFERENCE_DATA_DIR_NAME
            )
            self.reference_data_file_name:str=os.path.join(
                  self.reference_data_dir,
                  training_pipeline.REFERENCE_DATA_NAME
            )
            self.production_data_dir:str=os.path.join(
                  self.data_validation_dir,
                  training_pipeline.PRODUCTION_DATA_DIR_NAME
            )
            self.production_data_file_name:str=os.path.join(
                  self.production_data_dir,
                  training_pipeline.PRODUCTION_DATA_NAME
            )
class FeatureExtractionConfig:
      def __init__(self,training_pipeline_config:TrainingPipelineConfig):
            self.feature_extraction_dir:str=os.path.join(
                  training_pipeline_config.artifact_dir,training_pipeline.DATA_FEATURE_EXTRACTION_DIR_NAME
            )
            self.training_file_path:str=os.path.join(
                  self.feature_extraction_dir,
                  training_pipeline.DATA_FEATURE_EXTRACTION_TRAIN_FILE_NAME
            )
            self.testing_file_path:str=os.path.join(
                  self.feature_extraction_dir,
                  training_pipeline.DATA_FEATURE_EXTRACTION_TEST_FILE_NAME
            )

class DataTransformationConfig:

      def __init__(self,training_pipeline_config:TrainingPipelineConfig):
            self.data_transformation_dir:str=os.path.join(
                  training_pipeline_config.artifact_dir,training_pipeline.DATA_TRANSFORMATION_DIR_NAME
            ) 
            
            self.transformed_train_file_path:str=os.path.join(
                  self.data_transformation_dir,
                  training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                  training_pipeline.DATA_TRANSFORMATION_TRAIN_FILE_PATH
            )
            self.transformed_test_file_path:str=os.path.join(
                  self.data_transformation_dir,
                  training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                  training_pipeline.DATA_TRANSFORMATION_TEST_FILE_PATH
            )
            self.preprocessor_obj_path:str=os.path.join(
                  self.data_transformation_dir,
                  training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                  training_pipeline.PREPROCESSING_OBJECT_FILE_NAME
            )
class ModelTrainerConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.model_trainer_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.MODEL_TRAINER_DIR_NAME
        )
        self.trained_model_file_path: str = os.path.join(
            self.model_trainer_dir, training_pipeline.MODEL_TRAINER_TRAINED_MODEL_DIR, 
            training_pipeline.MODEL_FILE_NAME
        )
        self.expected_accuracy: float = training_pipeline.MODEL_TRAINED_EXPECTED_SCORE
        self.overfitting_underfitting_threshold = training_pipeline.MODEL_TRAINER_OVER_FIITING_UNDER_FITTING_THRESHOLD             
