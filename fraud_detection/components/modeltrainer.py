from fraud_detection.entity.config_entity import ModelTrainerConfig
from fraud_detection.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from fraud_detection.exception.exception import CustomException
from fraud_detection.utils.main_utils.utils  import save_object,run_final_model,get_features,get_preprocessor,load_sparse_array
from fraud_detection.utils.ml_utils.metric.classification_metric import calculate_metrics

import pandas as pd
import numpy as np
import sys

from lightgbm import LGBMClassifier
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

class ModelTrainer:
    def __init__(self,model_trainer_config:ModelTrainerConfig,data_transformation_artifact:DataTransformationArtifact):
          self.model_trainer_config=model_trainer_config
          self.data_transformation_artifact=data_transformation_artifact

    # def track_mlflow(self,best_model,classfication_metric,tag):
    #     with mlflow.start_run():
             
    #         mlflow.set_tag("dataset",tag)
    #         mlflow.set_tag("model",type(best_model).__name__)

    #         mlflow.log_metrics({
    #              "f1_score":classfication_metric.f1_score,
    #              "recall_score":classfication_metric.recall_score,
    #              "precision_score":classfication_metric.precision_score
    #          })
            
    #         mlflow.sklearn.log_model(best_model,"model")

    def train_model(self,x_train,x_test,y_train,y_test):

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        models = {
               "LGBMClassifier": LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42)
        }

        params={
            "LGBMClassifier":{
              "n_estimators": [500, 1000, 1500],
              "num_leaves": [15, 31, 63],
              "learning_rate": [0.01, 0.05, 0.1],
              "max_depth": [-1, 6, 8],
              "reg_alpha": [0.0, 0.1, 1.0, 10.0],    
             "reg_lambda": [0.0, 0.1, 1.0, 10.0],
            }
         }
    
        try:
            num_cols,cat_cols =get_features()
            preprocessor =get_preprocessor(num_cols,cat_cols)
            name,results,importance,best_model=run_final_model(x_train,x_test,y_train,y_test,models,params,preprocessor,num_cols,cat_cols)
 
            y_train_pred=best_model.predict(x_train)
            classification_train_metric=calculate_metrics(y_train,y_train_pred)
            # self.track_mlflow(best_model,classification_train_metric,"train")

            y_test_pred=best_model.predict(x_test)
            classification_test_metric=calculate_metrics(y_test,y_test_pred)
            # self.track_mlflow(best_model,classification_test_metric,"test")

            save_object(self.model_trainer_config.trained_model_file_path,best_model)
       
            model_trainer_artifact=ModelTrainerArtifact(
              model_obj_file_path=self.model_trainer_config.trained_model_file_path,
              train_metric_artifact=classification_train_metric,
              test_metric_artifact= classification_test_metric
            )
            
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
        
    

    def initiate_model_training(self):
        try:
            train_transformed_data=load_sparse_array(self.data_transformation_artifact.transformed_train_file_path)
            test_transformed_data=load_sparse_array(self.data_transformation_artifact.transformed_test_file_path)

            x_train,y_train,x_test,y_test=(
             train_transformed_data[:,:-1],
             train_transformed_data[:, -1].toarray().ravel(),  
             test_transformed_data[:,:-1],
             test_transformed_data[:, -1].toarray().ravel(),   
            )

            model_train_artifact=self.train_model(x_train,x_test,y_train,y_test)
            return model_train_artifact
        
        except Exception as e:
            raise CustomException(e,sys)
        
    

    

        
        
             
             

          

          