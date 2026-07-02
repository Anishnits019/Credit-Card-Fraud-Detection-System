from diabetes.components.dataingestion import DataIngestion
from diabetes.components.datavalidation import DataValidation
from diabetes.components.datatransformation import DataTransformation
from diabetes.components.modeltrainer import ModelTrainer

from diabetes.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig

from diabetes.entity.config_entity import TrainingPipelineConfig

from diabetes.exception.exception import CustomException
import sys

if __name__=="__main__":
 try:
   training_pipeline_config=TrainingPipelineConfig()
  
   data_ingestion_config=DataIngestionConfig(training_pipeline_config)
   data_ingestion=DataIngestion(data_ingestion_config)
   data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
  
   data_validation_config=DataValidationConfig(training_pipeline_config)
   data_validation=DataValidation(data_validation_config,data_ingestion_artifact)
   data_validation_artifact=data_validation.initiate_data_validation()

   data_transformation_config=DataTransformationConfig(training_pipeline_config)
   data_transformation=DataTransformation(data_transformation_config,data_validation_artifact)
   data_transformation_artifact=data_transformation.initiate_data_transformation()
  
   model_trainer_config=ModelTrainerConfig(training_pipeline_config)
   model_trainer=ModelTrainer(model_trainer_config,data_transformation_artifact)
   model_trainer_artifact=model_trainer.initiate_model_training()

 except Exception as e:
  raise CustomException(e,sys)
 




 