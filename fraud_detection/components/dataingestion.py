import os
import sys
import pandas as pd 
import numpy as np
import pymongo

from sklearn.model_selection import train_test_split  

from diabetes.exception.exception import CustomException
from diabetes.logging.logger import logging
from diabetes.entity.config_entity import DataIngestionConfig
from diabetes.entity.artifact_entity import DataIngestionArtifact
from diabetes.components.customclasstrainer import CustomFeatureEngineering
from dotenv import load_dotenv  
load_dotenv() 

MONGO_DB_URL = os.getenv("MONGO_DB_URL")
if not MONGO_DB_URL:
    logging.error("MONGO_DB_URL environment variable is missing in the .env file.")
    raise Exception("MONGO_DB_URL is not set in .env file")

class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            logging.info("Initializing Data Ingestion component...")
            self.data_ingestion_config = data_ingestion_config
            logging.info("Data Ingestion configuration loaded successfully.")
        except Exception as e:
           raise CustomException(e, sys) 

    def import_data_from_database(self) -> pd.DataFrame:
        try:
           s3_uri="s3://fraud-detection-pipeline-anish-964043552068-ap-south-1-an/merged_transactions.parquet"
           logging.info(f"Connecting to AWS S3 and streaming: {s3_uri}")
           data=pd.read_parquet(
               s3_uri,
               engine="fastparquet",
               storage_options={
                   "key":os.getenv("AWS_ACCESS_KEY_ID"),
                   "secret":os.getenv("AWS_SECRET_ACCESS_KEY"),
                   "client_kwargs":{
                       "region_name":os.getenv("AWS_DEFAULT_REGION")
                    }
               }
           )
        #    database_name = self.data_ingestion_config.database_name
        #    collection_name = self.data_ingestion_config.collection_name

        #    logging.info(f"Connecting to MongoDB database: '{database_name}' and collection: '{collection_name}'...")
        #    self.mongo_client = pymongo.MongoClient(MONGO_DB_URL)
        #    collection = self.mongo_client[database_name][collection_name]
           
        #    logging.info("Fetching data records from MongoDB...")
        #    data = list(collection.find())
        #    logging.info(f"Successfully fetched {len(data)} records from MongoDB.")

        #    dataframe = pd.DataFrame(data)
        #    logging.info("Converted MongoDB records into Pandas DataFrame.")

        #    if "_id" in dataframe.columns:
        #     logging.info("Dropping default MongoDB '_id' column from the DataFrame.")
        #     dataframe = dataframe.drop(columns=["_id"])
           
        #    logging.info(f"DataFrame shape after extraction: {dataframe.shape}")
        #    return dataframe

           dataframe = pd.DataFrame(data)
           logging.info("Converted AWS records into Pandas DataFrame.")

           if "_id" in dataframe.columns:
            logging.info("Dropping default MongoDB '_id' column from the DataFrame.")
            dataframe = dataframe.drop(columns=["_id"])
           
           logging.info(f"DataFrame shape after extraction: {dataframe.shape}")
           return dataframe

       
        except Exception as e:
           logging.error("Exception occurred while importing data from MongoDB database.")
           raise CustomException(e, sys)

    def export_data_into_feature_store(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        try:
          feature_store_path = self.data_ingestion_config.feature_store_file_path
          logging.info(f"Exporting raw data to Feature Store path: {feature_store_path}")
          
          # Create parent directories if they don't exist
          os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)

          dataframe.to_csv(feature_store_path, index=False, header=True)
          logging.info("Raw data exported to Feature Store successfully.")

          return dataframe
        
        except Exception as e:
           logging.error("Exception occurred while exporting data into feature store.")
           raise CustomException(e, sys)
        
    def cleaning_data(self,dataframe:pd.DataFrame):
        try:
            custom_feature=CustomFeatureEngineering()
            dataframe=custom_feature.transform(dataframe)
            return dataframe
        except Exception as e:
           raise CustomException(e, sys)


    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
             logging.info("Initiating Train-Test split on the DataFrame.")
             train_data, test_data = train_test_split(
                 dataframe, 
                 test_size=self.data_ingestion_config.train_test_split_ratio, 
                 random_state=42
             )
             logging.info(f"Split completed. Train shape: {train_data.shape}, Test shape: {test_data.shape}")

             # Create directory paths for train and test files
             logging.info("Creating directory structures for training and testing file splits.")
             os.makedirs(os.path.dirname(self.data_ingestion_config.training_file_path), exist_ok=True)
             os.makedirs(os.path.dirname(self.data_ingestion_config.testing_file_path), exist_ok=True)

             # Exporting splits to CSV
             logging.info(f"Saving training file to: {self.data_ingestion_config.training_file_path}")
             train_data.to_csv(self.data_ingestion_config.training_file_path, index=False, header=True)

             logging.info(f"Saving testing file to: {self.data_ingestion_config.testing_file_path}")
             test_data.to_csv(self.data_ingestion_config.testing_file_path, index=False, header=True)

             logging.info("Train-Test files saved successfully.")
             return self.data_ingestion_config.training_file_path, self.data_ingestion_config.testing_file_path
         
        except Exception as e:
            logging.error("Exception occurred during Train-Test data split operation.")
            raise CustomException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try: 
            logging.info("==================== Starting Data Ingestion Process ====================")
            
            dataframe = self.import_data_from_database()
            dataframe = self.export_data_into_feature_store(dataframe)
            dataframe=self.cleaning_data(dataframe)
            self.split_data_as_train_test(dataframe)
            
            logging.info("Creating Data Ingestion Artifact object.")
            dataingestionartifact = DataIngestionArtifact(
               trained_file_path=self.data_ingestion_config.training_file_path,
               test_file_path=self.data_ingestion_config.testing_file_path,
               reference_data_file_path=None,   
               trained_data_file_path=None
            )
            
            logging.info(f"Data Ingestion completed successfully. Artifact created: {dataingestionartifact}")
            logging.info("==================== Data Ingestion Process Finished ====================")
            return dataingestionartifact
            
        except Exception as e:
           logging.error("Data Ingestion pipeline process failed.")
           raise CustomException(e, sys)