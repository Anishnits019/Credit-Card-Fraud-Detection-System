import yaml
import os,sys
import numpy as np
import pandas as pd
from diabetes.exception.exception import CustomException
from diabetes.constant.training_pipeline import SCHEMA_FILE_PATH
from diabetes.logging.logger import logging
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score,f1_score

def read_yaml_file(file_path):
    with open (file_path,"r") as yaml_file:
        try:
            return yaml.safe_load(yaml_file)
        except Exception as e:
            raise CustomException(e,sys)
        
def write_yaml_file(file_path,content):
    try:
     with open (file_path,"w") as file_obj:
        yaml.dump(content,file_obj)
    except Exception as e:
     raise CustomException(e,sys)

def save_numpy_array(file_path,data):
    try:
       os.makedirs(os.path.dirname(file_path),exist_ok=True)
       with open (file_path,"wb") as file_obj:
        np.save(file_obj,data)
    except Exception as e:
      raise CustomException(e,sys)
    
def save_object(file_path,obj):
    try:
       os.makedirs(os.path.dirname(file_path),exist_ok=True)
       with open (file_path,"wb") as file_obj:
        pickle.dump(obj,file_obj)
    except Exception as e:
      raise CustomException(e,sys)

def load_numpy_array(file_path):
    try:
         with open (file_path,"rb") as file_obj:
          return np.load(file_obj)
    except Exception as e:
      raise CustomException(e,sys)

def evaluate_models(x_train,x_test,y_train,y_test,models,params):
    results={}
    for name,model in models.items():
     
     gs=GridSearchCV(estimator=model,param_grid=params[name],cv=3,n_jobs=-1,scoring='accuracy')
     #n_jobs=-1 set all the used all vritual cores of the system 
     gs.fit(x_train,y_train)

     best_model=gs.best_estimator_
     print(f"Took {time.time() - start:.1f} seconds")
     print(gs.best_params_, gs.best_score_)
     y_train_pred=best_model.predict(x_train)
     y_test_pred=best_model.predict(x_test)

     train_score = f1_score(y_train, y_train_pred)
     test_score = f1_score(y_test, y_test_pred)

     results[name]={
        "train_score":train_score,
        "test_score":test_score,
        "best_model":best_model
     }
     print(results)
    return results

        

    