from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatureEngineering(BaseEstimator, TransformerMixin):
         
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # 1. Work on a copy to prevent modifying the original dataframe
        X = X.copy()
        
        X['trans_date_trans_time'] = pd.to_datetime(X['trans_date_trans_time'])
        X['dob'] = pd.to_datetime(X['dob'])
        
        X['age'] = (
            pd.to_datetime(X['unix_time'], unit='s').dt.year -
            X['dob'].dt.year
        )     
        
        X["merchant"] = X["merchant"].str.removeprefix("fraud_")
        
        return X
