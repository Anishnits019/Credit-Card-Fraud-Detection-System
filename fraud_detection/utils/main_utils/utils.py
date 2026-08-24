import os
import sys
import yaml
import pickle
import numpy as np
import pandas as pd
from scipy import sparse

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, average_precision_score
from fraud_detection.exception.exception import CustomException
from fraud_detection.logging.logger import logging


def read_yaml_file(file_path):
    with open(file_path, "r") as yaml_file:
        try:
            return yaml.safe_load(yaml_file)
        except Exception as e:
            raise CustomException(e, sys)
        
def write_yaml_file(file_path, content):
    try:
        with open(file_path, "w") as file_obj:
            yaml.dump(content, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

# def save_numpy_array(file_path, data):
#     try:
#         os.makedirs(os.path.dirname(file_path), exist_ok=True)
#         with open(file_path, "wb") as file_obj:
#             np.save(file_obj, data)
#     except Exception as e:
#         raise CustomException(e, sys)
    
def save_sparse_array(file_path, data):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        sparse.save_npz(file_path, data)
    except Exception as e:
        raise CustomException(e, sys)

    
def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

# def load_numpy_array(file_path):
#     try:
#         with open(file_path, "rb") as file_obj:
#             return np.load(file_obj)
#     except Exception as e:
#         raise CustomException(e, sys)
def load_sparse_array(file_path):
    try:
        return sparse.load_npz(file_path)
    except Exception as e:
        raise CustomException(e, sys)
def get_features():
    num_cols = [
        # Baseline
        'cc_num', 'amt', 'zip', 'lat', 'long', 'city_pop', 
        'unix_time', 'merch_lat', 'merch_long', 'age',
        
        # Time based
        'hour_of_day', 'month', 'day_of_week', 'is_weekend',
        'txn_count_1h', 'txn_count_6h', 'txn_count_24h',
        'hour_fraud_rate', 'day_fraud_rate', 'month_fraud_rate',
        'is_high_risk_age', 'is_low_risk_age', 'is_medium_risk_age',
        'is_low_risk_hour', 'is_early_morning', 'is_late_night',
        'is_high_risk_day', 'is_high_risk_window',
    
        # Distance based
        'home_to_merchant_dist', 'likely_diff_state',
        'dist_change_prev_txn', 'time_since_prev_txn',

        # Transaction Based
        'amt_z_score', 'amt_zscore_category', 'is_round_1', 'is_round_10',

        # Merchant Based
        'cc_merchant_txn_count', 'account_age_days', 'is_typical_category',
        'card_usage_freq', 'customer_category_entropy',

        'is_high_risk_amt', 'is_very_high_risk_amt', 'is_card_testing_amt',
        'is_low_risk_amt', 'is_micro_amt',

        'is_high_risk_category', 'is_low_risk_category'
    ]             
            
    cat_cols = [
        'merchant', 'category', 'gender', 'city', 'state'
    ]
    
    return num_cols, cat_cols

def get_preprocessor(num_cols, cat_cols):
    logging.info("Building preprocessor pipeline")

    num_pipeline = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
      
    cat_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])
      
    column_transformer = ColumnTransformer(
        transformers=[
            ("num_features", num_pipeline, num_cols),
            ("cat_features", cat_pipeline, cat_cols)
        ], remainder='passthrough'
    )

    preprocessor = Pipeline(steps=[
        ("column_transformer", column_transformer)
    ])
    logging.info("Preprocessor pipeline built successfully.")

    return preprocessor

def run_final_model(x_train, x_test, y_train, y_test, models, params, preprocessor, num_cols, cat_cols):
    results = {}
    
    for name, model in models.items():
        logging.info(f"Starting Hyperparameter Tuning for model: {name}")
        
        randomsearch = RandomizedSearchCV(
            estimator=model,
            param_distributions=params[name],
            verbose=2,
            n_jobs=-1,
            cv=3
        )
        randomsearch.fit(x_train, y_train)
        best_model = randomsearch.best_estimator_
     
        probs = best_model.predict_proba(x_test)[:, 1]

        roc_auc = roc_auc_score(y_test, probs)
        avg_precision = average_precision_score(y_test, probs)
        preds = np.where(probs >= 0.5, 1, 0)
       
        results = x_test[['cc_num']].copy()
        results['prob'] = probs
        results['prediction'] = preds
        results['actual'] = y_test.values
        results['roc_auc'] = roc_auc
        results['avg_precision'] = avg_precision
        
        ohe_feature_names = preprocessor.named_steps['column_transformer'] \
                                        .named_transformers_['cat_features'] \
                                        .get_feature_names_out(cat_cols).tolist()

        all_feature_names = num_cols + ohe_feature_names

        importance = pd.DataFrame({
            'feature': all_feature_names,
            'importance': best_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return name, results, importance, best_model