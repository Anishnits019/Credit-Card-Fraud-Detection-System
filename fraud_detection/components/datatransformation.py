from diabetes.entity.config_entity import DataTransformationConfig
from diabetes.entity.artifact_entity import DataValidationArtifact,DataTransformationArtifact
from diabetes.exception.exception import CustomException
from diabetes.logging.logger import logging
from diabetes.constant.training_pipeline import TARGET_COLUMN
from diabetes.constant.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from diabetes.components.customclasstrainer import CustomFeatureEngineering
from sklearn.preprocessing import StandardScaler,OrdinalEncoder,RobustScaler
from sklearn.compose import ColumnTransformer
from diabetes.utils.main_utils.utils import save_numpy_array,save_object
import pandas as pd
import numpy as np
import os,sys

                 
class DataTransformation:
     def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_validation_artifact: DataValidationArtifact):              
                 self.data_transformation_config = data_transformation_config             
                 self.data_validation_artifact = data_validation_artifact  
                 logging.info("DataTransformation component initialized successfully.")

     @staticmethod
     def read_data(file_path):
          try:
               logging.info(f"Reading data from {file_path}")
               return pd.read_csv(file_path)
          except Exception as e:
           raise CustomException(e,sys)
                
     def get_preprocessor(self):
          logging.info("Building preprocessor pipeline")

          num_cols=['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi',
            'gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
          
          cat_cols = ['bmi_category']

          knn_imputer=KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)

          num_pipeline=Pipeline(steps=[
               ("scaler", RobustScaler())
          ])
          cat_pipeline=Pipeline(steps=[
               ("ordinal",OrdinalEncoder(categories=[['Underweight', 'Normal', 'Overweight', 'Obese']]))
          ])
          column_transformer=ColumnTransformer(
          transformers=[
            ("num_features",num_pipeline,num_cols),
            ("cat_features",cat_pipeline,cat_cols)
          ])
          preprocessor=Pipeline(steps=[
               ("column_transformer",column_transformer),
               ("KnnImputer", knn_imputer),
         ])
          logging.info("Preprocessor pipeline built successfully.")

          return preprocessor
      
     def initiate_data_transformation(self):
          try: 
              logging.info("==================== Starting Data Transformation ====================")
              train_data=self.read_data(self.data_validation_artifact.valid_train_file_path)
              test_data=self.read_data(self.data_validation_artifact.valid_test_file_path)

              logging.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")

              preprocessor=self.get_preprocessor()

              target_feature_train=train_data[TARGET_COLUMN]
              input_feature_train=train_data.drop(columns=[TARGET_COLUMN],axis=1)
              logging.info(f"Train input shape: {input_feature_train.shape}, Train target shape: {target_feature_train.shape}")

              
              target_feature_test= test_data[TARGET_COLUMN]
              input_feature_test= test_data.drop(columns=[TARGET_COLUMN],axis=1)
              logging.info(f"Test input shape: {input_feature_test.shape}, Test target shape: {target_feature_test.shape}")

              logging.info("Applying fit_transform on train data")
              train_transform=preprocessor.fit_transform(input_feature_train)

              logging.info("Applying transform on test data")
              test_transform=preprocessor.transform(input_feature_test)
              logging.info(f"Train transformed shape: {train_transform.shape}, Test transformed shape: {test_transform.shape}")
       
              train_arr=np.c_[train_transform,np.array(target_feature_train)]
              test_arr=np.c_[test_transform,np.array(target_feature_test)]
              logging.info(f"Final train array shape: {train_arr.shape}, Final test array shape: {test_arr.shape}")

              logging.info(f"Saving transformed train array to {self.data_transformation_config.transformed_train_file_path}")
              save_numpy_array(self.data_transformation_config.transformed_train_file_path,train_arr)

              logging.info(f"Saving transformed test array to {self.data_transformation_config.transformed_test_file_path}")
              save_numpy_array(self.data_transformation_config.transformed_test_file_path,test_arr)

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