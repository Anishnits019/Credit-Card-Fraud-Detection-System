from diabetes.entity.config_entity import DataValidationConfig
from diabetes.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from diabetes.exception.exception import CustomException
from diabetes.logging.logger import logging
from diabetes.constant.training_pipeline import SCHEMA_FILE_PATH
from diabetes.constant.training_pipeline import DATA_VALIDATION_THRESHOLD
from diabetes.utils.main_utils.utils import write_yaml_file
from diabetes.utils.main_utils.utils import read_yaml_file

from scipy.stats import ks_2samp
import pandas as pd
import os,sys

class DataValidation:
    def __init__(self,data_validation_config:DataValidationConfig,data_ingestion_artifact:DataIngestionArtifact):
        try:
            logging.info("Initializing DataValidation component")
            self.data_validation_config=data_validation_config
            self.data_ingestion_artifact=data_ingestion_artifact
            self._schema_config=read_yaml_file(SCHEMA_FILE_PATH)
            logging.info("DataValidation initialized successfully")
        except Exception as e:
            raise CustomException(e,sys)
    
    @staticmethod
    def read_data(file_path)->pd.DataFrame:
        try:
            logging.info(f"Reading data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
        
    def validate_number_of_columns(self,dataFrame):
        try:
            logging.info("Validating number of columns")
            number_of_columns= len(self._schema_config)
            if len(dataFrame.columns)==number_of_columns:
                logging.info("Column count validation passed")
                return True
            else:
                logging.info(f"Column count validation failed: expected {number_of_columns}, got {len(dataFrame.columns)}")
                return False
        except Exception as e:
            raise CustomException(e,sys) 
          
    def validte_drift(self,reference_data,production_data):
        try:
            logging.info("Starting drift detection")
            report={}
            status=True
            for column in reference_data.columns:
                statistic,p_value=ks_2samp(reference_data[column],production_data[column])
                is_drift=p_value<DATA_VALIDATION_THRESHOLD
                if is_drift:
                    status=False
                    logging.info(f"Drift detected in column: {column} | p_value: {p_value}")
                report[column]={
                        'p_value':p_value,
                        'is_drift':is_drift
                }
            drift_report_file_path=self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path),exist_ok=True)
            write_yaml_file(file_path=drift_report_file_path,content=report)
            logging.info(f"Drift report saved at {drift_report_file_path}")
            return status


        except Exception as e:
            raise CustomException(e,sys)
    def load_reference_data(self,file_path) -> pd.DataFrame:
        """
        In production: load the snapshot saved when the model was deployed.
        Could be from S3, a feature store, a versioned DB table, etc.
        """
        try:
            logging.info(f"Loading reference data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)

    def load_live_data(self,file_path) -> pd.DataFrame:
        """
        In production: query logs of recent inference requests.
        Could be from a data warehouse, Kafka, S3 partitioned by date, etc.
        """
        live_data_path = self.data_validation_config.live_data_path
        df = pd.read_csv(file_path)

        df["timestamp"] = pd.to_datetime(df["timestamp"])
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=7)
        return df[df["timestamp"] >= cutoff]

    def initiate_data_validation(self):
        try:
            logging.info("Initiating data validation")
            if (self.data_ingestion_artifact.reference_data_file_path is not None and
                self.data_ingestion_artifact.trained_data_file_path   is not None):
                
              train_data = self.read_data(self.data_ingestion_artifact.reference_data_file_path)
              test_data= self.read_data(self.data_ingestion_artifact.trained_data_file_path)

            else:
              train_data = self.read_data(self.data_ingestion_artifact.trained_file_path)
              test_data = self.read_data(self.data_ingestion_artifact.test_file_path)

            status=self.validate_number_of_columns(train_data)
            if not status:
              error_message=f"Train dataframe does not contain all columns.\n"

            status = self.validate_number_of_columns(dataFrame=test_data)
            if not status:
              error_message=f"Train dataframe does not contain all columns.\n"

            status = self.validte_drift(reference_data=train_data, production_data=test_data)

            self.validte_drift(train_data,train_data)

            
            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )
            logging.info("Data validation completed successfully")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e,sys)