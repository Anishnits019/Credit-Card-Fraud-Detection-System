from fraud_detection.entity.config_entity import DataTransformationConfig
from fraud_detection.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact,FeatureExtractionArtifact
from fraud_detection.exception.exception import CustomException
from fraud_detection.logging.logger import logging
from fraud_detection.constant.training_pipeline import TARGET_COLUMN
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from fraud_detection.components.customclasstrainer import CustomFeatureEngineering
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from fraud_detection.utils.main_utils.utils import save_sparse_array,save_object,get_features,get_preprocessor
import pandas as pd
import numpy as np
import os,sys
from scipy import sparse

                 
class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig,
                 feature_extraction_artifact: FeatureExtractionArtifact):              
        self.data_transformation_config = data_transformation_config             
        self.feature_extraction_artifact = feature_extraction_artifact
        logging.info("DataTransformation component initialized successfully.")

    @staticmethod
    def read_data(file_path):
        try:
            logging.info(f"Reading data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)
                      
    def initiate_data_transformation(self):
        try: 
            logging.info("==================== Starting Data Transformation ====================")
            train_data=self.read_data(self.feature_extraction_artifact.trained_file_path)
            test_data=self.read_data(self.feature_extraction_artifact.test_file_path)
            num_cols,cat_cols = get_features()
            
            logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")

            cols_to_exclude = [TARGET_COLUMN, 'trans_date_trans_time']
            
            clean_features = [col for col in (num_cols + cat_cols) if col not in cols_to_exclude]
            
            logging.info(f"Train data initial shape: {train_data.shape}, Test data initial shape: {test_data.shape}")

            preprocessor =get_preprocessor(num_cols,cat_cols)

            # 2. Extract targets
            target_feature_train = train_data[TARGET_COLUMN]
            target_feature_test = test_data[TARGET_COLUMN]

            # 3. Extract inputs safely using the cleaned feature list
            input_feature_train = train_data[clean_features]
            input_feature_test = test_data[clean_features]

            logging.info(f"Train input shape: {input_feature_train.shape}, Train target shape: {target_feature_train.shape}")
            logging.info(f"Test input shape: {input_feature_test.shape}, Test target shape: {target_feature_test.shape}")

            logging.info("Applying fit_transform on train data")
            train_transform=preprocessor.fit_transform(input_feature_train)

            logging.info("Applying transform on test data")
            test_transform=preprocessor.transform(input_feature_test)
            logging.info(f"Train transformed shape: {train_transform.shape}, Test transformed shape: {test_transform.shape}")
       
            # train_arr=np.c_[train_transform,np.array(target_feature_train)]
            # test_arr=np.c_[test_transform,np.array(target_feature_test)]

            train_arr = sparse.hstack([
                        train_transform,
                        sparse.csr_matrix(np.array(target_feature_train).reshape(-1, 1))]).tocsr()
            
            test_arr = sparse.hstack([
                       test_transform,
                       sparse.csr_matrix(np.array(target_feature_test).reshape(-1, 1))]).tocsr()
            
            logging.info(f"Final train array shape: {train_arr.shape}, Final test array shape: {test_arr.shape}")

            logging.info(f"Saving transformed train array to {self.data_transformation_config.transformed_train_file_path}")
            # save_numpy_array(self.data_transformation_config.transformed_train_file_path,train_arr)
            save_sparse_array(self.data_transformation_config.transformed_train_file_path, train_arr)

            logging.info(f"Saving transformed test array to {self.data_transformation_config.transformed_test_file_path}")
            # save_numpy_array(self.data_transformation_config.transformed_test_file_path,test_arr)
            save_sparse_array(self.data_transformation_config.transformed_test_file_path, test_arr)

            logging.info(f"Saving preprocessor object to {self.data_transformation_config.preprocessor_obj_path}")
            save_object(self.data_transformation_config.preprocessor_obj_path,preprocessor)
       
            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,    
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,     
                preprocessor_file_path=self.data_transformation_config.preprocessor_obj_path        
            )
            
            logging.info("==================== Data Transformation Completed ====================")
            
            return data_transformation_artifact
            
        except Exception as e:
            raise CustomException(e,sys)