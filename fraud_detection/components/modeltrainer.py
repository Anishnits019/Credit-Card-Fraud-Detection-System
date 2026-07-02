from diabetes.entity.config_entity import ModelTrainerConfig
from diabetes.entity.artifact_entity import DataTransformationArtifact,ModelTrainerArtifact
from diabetes.exception.exception import CustomException
from diabetes.utils.main_utils.utils  import evaluate_models,save_object,load_numpy_array
from diabetes.utils.ml_utils.metric.classification_metric import get_classification_score

import os
import pandas as pd
import sys

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")

class ModelTrainer:
    def __init__(self,model_config:ModelTrainerConfig,data_transformation_config:DataTransformationArtifact):
          self.model_config=model_config
          self.data_transformation_config=data_transformation_config

    def track_mlflow(self,best_model,classfication_metric,tag):
        with mlflow.start_run():
             
            mlflow.set_tag("dataset",tag)
            mlflow.set_tag("model",type(best_model).__name__)

            mlflow.log_metrics({
                 "f1_score":classfication_metric.f1_score,
                 "recall_score":classfication_metric.recall_score,
                 "precision_score":classfication_metric.precision_score
             })
            
            mlflow.sklearn.log_model(best_model,"model")

    def train_model(self,x_train,x_test,y_train,y_test):
         
        models = {
                # "Random Forest": RandomForestClassifier(verbose=1),
                # "Decision Tree": DecisionTreeClassifier(),
                # "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                # "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }
        params={
            # "Decision Tree": {
            #     'max_depth':[3,4,5],               # allowed a definite depth such that model doesn't memeorize till infinite depth
            #     'criterion':['gini', 'entropy'],   # criterion for creating a tree
            #     'min_samples_split':[5,10,20],     # min samples requried to split a tree
            #     'min_samples_leaf':[5,10,20],      # min samples required or left on leaf to split
            #     'splitter':['best','random'],      # best parameters selection or random parameter selection
            #     'max_features':['sqrt',2,4],       # how many features selectres based on sqrt(n) OR LOG(n) where N is the no of columns in a database
            # },
            # "Random Forest":{
            #     'criterion': ['gini', 'entropy'],  # criterion for creating a tree
            #     'max_depth':[4,6,8,10],        # allowed a definite depth such that model doesn't memeorize till infinite depth
            #     'min_samples_split':[5,10,20],   # min samples requried to split a tree
            #     'min_samples_leaf':[5,10,20],    # min samples required or left on leaf to split
            #     'max_features':['sqrt',2,4],  # how many features selectres based on sqrt(n) OR LOG(n) where N is the no of columns in a database
            #     'n_estimators': [200, 300], # independent tress to create for judgement           
            # },
            # "Gradient Boosting":{
            #     'loss':['log_loss', 'exponential'],
            #     'learning_rate':[.05,.01],
            #     'subsample':[0.7,0.8],
            #     'criterion':['squared_error', 'friedman_mse'],
            #     'n_estimators': [50, 100, 200, 300],             # independent tress to create for judgement           
            #     'max_depth':[3,4,5],                             # CRUCIAL: Added depth limit to stop massive overfitting

            # },
            "AdaBoost": {
               'learning_rate': [0.1, 0.05, 0.01],
               'n_estimators': [50, 100, 200,300],                    # Low estimators make AdaBoost underperform
               'algorithm': ['SAMME']                             # Explicitly set to avoid deprecation warnings
            },
    #         "Logistic Regression": {
    #         'penalty': ['l2'],                                 # Regularization type
    #         'C': [0.01, 0.1, 1.0, 10.0],                       # Inverse regularization strength (smaller = less overfitting)
    #         'solver': ['lbfgs', 'saga'],                       # Optimization algorithms
    #         'max_iter': [100, 500]                             # Gives the solver enough time to converge
    # }
            
        }
        try:
            model_report=evaluate_models(x_train,x_test,y_train,y_test,models,params)

            best_model_name=max(model_report ,key=lambda x:model_report[x]["test_score"])
            best_model_score=model_report[best_model_name]['train_score']
            best_model=model_report[best_model_name]["best_model"]
 
            y_train_pred=best_model.predict(x_train)
            classification_train_metric=get_classification_score(y_train,y_train_pred)
            self.track_mlflow(best_model,classification_train_metric,"train")

            y_test_pred=best_model.predict(x_test)
            classification_test_metric=get_classification_score(y_test,y_test_pred)
            self.track_mlflow(best_model,classification_test_metric,"test")

            if best_model_score>self.model_config.expected_accuracy:
                save_object(self.model_config.trained_model_file_path,best_model)
       
            model_trainer_artifact=ModelTrainerArtifact(
              model_obj_file_path=self.model_config.trained_model_file_path,
              train_metric_artifact=classification_train_metric,
              test_metric_artifact= classification_test_metric
            )
            
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
     
    def initiate_model_training(self):
        try:
            train_transformed_data=load_numpy_array(self.data_transformation_config.transformed_train_file_path)
            test_transformed_data=load_numpy_array(self.data_transformation_config.transformed_test_file_path)

            x_train,y_train,x_test,y_test=(
             train_transformed_data[:,:-1],
             train_transformed_data[:,-1],
             test_transformed_data[:,:-1],
             test_transformed_data[:,-1]
        )
            model_train_artifact=self.train_model(x_train,x_test,y_train,y_test)
            return model_train_artifact
        except Exception as e:
            raise CustomException

    

        
        
             
             

          

          