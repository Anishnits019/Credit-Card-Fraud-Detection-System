
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer
from fraud_detection.components.customclasstrainer import CustomFeatureEngineering
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from fraud_detection.entity.artifact_entity import DataValidationArtifact, FeatureExtractionArtifact 
from fraud_detection.entity.config_entity import FeatureExtractionConfig
from fraud_detection.exception.exception import CustomException
from fraud_detection.logging.logger import logging
import pandas as pd
import numpy as np
import os
import sys

class FeatureExtraction:
    def __init__(self, feature_extraction_config: FeatureExtractionConfig, data_validation_artifact: DataValidationArtifact):
        try:
            logging.info("Initializing Feature Extraction Component.")
            self.data_validation_artifact = data_validation_artifact
            self.feature_extraction_config = feature_extraction_config
        except Exception as e:
            raise CustomException(e, sys)



    @staticmethod
    def add_velocity_features(data: pd.DataFrame) -> tuple:
        try:
            logging.info("Calculating transaction velocity windows (1h, 6h, 24h).")
            data = data.copy()

            data = data.sort_values(['cc_num', 'trans_date_trans_time']).reset_index(drop=True)
            data = data.set_index('trans_date_trans_time')
            
            for window, col in [('1h', 'txn_count_1h'), ('6h', 'txn_count_6h'), ('24h', 'txn_count_24h')]:
                data[col] = (
                    data.groupby('cc_num')['amt']   
                    .rolling(window)             
                    .count()
                    .reset_index(level=0, drop=True)
                )

            data = data.reset_index()
            

            logging.info("Transaction velocity features calculated successfully.")
            return data
        except Exception as e:
            logging.error("Exception occurred while adding velocity features.")
            raise CustomException(e, sys)
    
    @staticmethod
    def add_fraud_rate(baseline_df: pd.DataFrame, data: pd.DataFrame) -> tuple:
        try:
            logging.info("Calculating historical target-encoded fraud rates for time partitions.")
            baseline_df = baseline_df.copy()
            data = data.copy()

            global_fraud_rate = baseline_df['is_fraud'].mean()

            hour_map = baseline_df.set_index('hour_of_day')['hour_fraud_rate']
            day_map = baseline_df.set_index('day_of_week')['day_fraud_rate']
            month_map = baseline_df.set_index('month')['month_fraud_rate']
         
            data['hour_fraud_rate'] = data['hour_of_day'].map(hour_map).fillna(global_fraud_rate)
            data['day_fraud_rate'] = data['day_of_week'].map(day_map ).fillna(global_fraud_rate)
            data['month_fraud_rate'] = data['month'].map(month_map).fillna(global_fraud_rate)
                
            logging.info("Time-based fraud rate maps generated and applied.")
            return  data
        except Exception as e:
            logging.error("Exception occurred while adding time-based fraud rate attributes.")
            raise CustomException(e, sys)
    
    @staticmethod
    def haversine_miles(data: pd.DataFrame) -> np.ndarray:
        try:
            R = 6371
            lat1 = np.radians(data['lat'])
            lon1 = np.radians(data['long'])
            lat2 = np.radians(data['merch_lat'])
            lon2 = np.radians(data['merch_long'])

            dlat = lat2 - lat1
            dlong = lon2 - lon1

            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2)**2
            d = 2 * np.arcsin(np.sqrt(a))
            return R * d
        except Exception as e:
            logging.error("Exception occurred inside backend Haversine computation.")
            raise CustomException(e, sys)

    @classmethod
    def add_geo_features(cls, baseline_df: pd.DataFrame, data: pd.DataFrame) -> tuple:
        try:
            logging.info("Extracting geospatial mapping profiles and sequential distance transitions.")
            data = data.copy()


            data['home_to_merchant_dist'] = cls.haversine_miles(data)
            data['likely_diff_state'] = (data['home_to_merchant_dist'] > 200)

            data = data.sort_values(['cc_num', 'trans_date_trans_time']).reset_index(drop=True)

            data['prev_distance'] = data.groupby('cc_num')['home_to_merchant_dist'].shift(1)
            data['dist_change_prev_txn'] = np.abs(data['home_to_merchant_dist'] - data['prev_distance']).fillna(0)
            data.drop(columns=['prev_distance'], inplace=True)

            data['prev_trans_date_trans_time'] = data.groupby('cc_num')['trans_date_trans_time'].shift(1)
            data['time_since_prev_txn'] = np.abs(
                (data['trans_date_trans_time'] - data['prev_trans_date_trans_time']).dt.total_seconds()
            ).fillna(0)
            data.drop(columns=['prev_trans_date_trans_time'], inplace=True)


            logging.info("Geospatial tracking matrices successfully updated.")
            return data
        except Exception as e:
            logging.error("Exception occurred while building geographic features.")
            raise CustomException(e, sys)
    
    @staticmethod
    def add_transaction_features(baseline_df: pd.DataFrame, data: pd.DataFrame) -> tuple:
        try:
            logging.info("Evaluating scalar transaction characteristics and cardholder Z-Scores.")
            data = data.copy()

            cardholder_stats = baseline_df.groupby('cc_num')['amt'].agg(['mean', 'std']).rename(
                columns={'mean': "amt_mean", 'std': 'amt_std'}
            )
            
            merged = data.join(cardholder_stats, on='cc_num')
            data['amt_z_score'] = ((data['amt'] - merged['amt_mean']) / merged['amt_std']).fillna(0)

            category_stats = baseline_df.groupby('category')['amt'].agg(['mean', 'std']).rename(
                columns={'mean': 'cat_amt_mean', 'std': 'cat_amt_std'}
            )

            
            merged = data.join(category_stats, on='category')
            data['amt_zscore_category'] = (
                    (data['amt'] - merged['cat_amt_mean']) / merged['cat_amt_std']
                ).fillna(0)
            
            data['is_round_1'] = (data['amt'] % 1 == 0).astype(int)
            data['is_round_10'] = (data['amt'] % 10 == 0).astype(int)
                
            logging.info("Transaction value variance tracking complete.")
            return data
        except Exception as e:
            logging.error("Exception occurred while extracting pricing variance signals.")
            raise CustomException(e, sys)
    
    @staticmethod
    def add_merchant_based_features(baseline_df: pd.DataFrame, data: pd.DataFrame) -> tuple:
        try:
            logging.info("Compiling merchant and network risk profile aggregates.")
            data = data.copy()

            global_fraud_rate = baseline_df['is_fraud'].mean()
           
            data['merchant_fraud_rate'] = (
                    data['merchant']
                    .map(baseline_df[['merchant','merchant_fraud_rate']])
                    .fillna(global_fraud_rate)  
            )

        
            data['category_fraud_rate'] = (
                    data['category']
                    .map(baseline_df[['category','category_fraud_rate']])
                    .fillna(global_fraud_rate)  
                ) 
                

            # for data_split in [baseline_df, data]:
            #     data_split['cc_merchant_txn_count'] = (
            #         data_split.set_index(['cc_num', 'merchant'])
            #         .index
            #         .map(cardholder_merchant_counts)
            #         .fillna(0)
            #         .values
            #     )
              
            logging.info("Merchant vulnerability tracking features appended successfully.")
            return  data
        except Exception as e:
            logging.error("Exception occurred while aggregating merchant risk records.")
            raise CustomException(e, sys)
    
    @staticmethod
    def cardholder_behaviour_features(baseline_df: pd.DataFrame, data: pd.DataFrame) -> tuple:
        try:
            logging.info("Assessing account maturity, usage density profiles, and spending entropy.")
            data = data.copy()

            first_txn_date = (
                data.groupby('cc_num')['trans_date_trans_time']
                .min()
                .rename('first_txn_date')
            )

            data['account_age_days'] = (
                    data['trans_date_trans_time']
                    - data['cc_num'].map(first_txn_date)
                ).dt.days.fillna(0)

            top_category = (
                baseline_df.groupby('cc_num')['category']
                .agg(lambda x: x.value_counts().index[0])
                .rename('top_category')
            )

            for df_split in [baseline_df, data]:
                df_split['is_typical_category'] = (
                    df_split['category'] == df_split['cc_num'].map(top_category)
                ).astype(int)

            cardholder_txn_count = baseline_df.groupby('cc_num').size()

            cardholder_date_range = (
                baseline_df.groupby('cc_num')['trans_date_trans_time']
                .agg(lambda x: (x.max() - x.min()).days + 1) 
            )

            card_usage_freq = (
                (cardholder_txn_count / cardholder_date_range)
                .rename('card_usage_freq')
            )

            global_usage_freq = card_usage_freq.mean()

            for df_split in [baseline_df, data]:
                df_split['card_usage_freq'] = (
                    df_split['cc_num']
                    .map(card_usage_freq)
                    .fillna(global_usage_freq)
                )
            
            category_probability = (
                baseline_df.groupby(['cc_num', 'category'])['amt'].count()
                / baseline_df.groupby('cc_num')['category'].count()
            ).reset_index(name='category_probability')

            category_probability['entropy_component'] = (
                category_probability['category_probability']
                * np.log2(category_probability['category_probability'])
            )

            customer_category_entropy = (
                -category_probability.groupby('cc_num')['entropy_component'].sum()
            )
            global_customer_entropy = customer_category_entropy.median()

            for data_split in [baseline_df, data]:
                data_split['customer_category_entropy'] = (
                    data_split['cc_num'].map(customer_category_entropy).fillna(global_customer_entropy)
                )

            logging.info("Cardholder long-term behavior profiles fully established.")
            return baseline_df, data
        except Exception as e:
            logging.error("Exception occurred while building historical cardholder profile traits.")
            raise CustomException(e, sys)
    
    @staticmethod
    def flag_high_risk(baseline_df: pd.DataFrame, data: pd.DataFrame) -> tuple:
        try:
            logging.info("Mapping critical operating hour exceptions (Early Morning, Late Night, Weekends).")
            baseline_df = baseline_df.copy()
            data = data.copy()

            for df in [baseline_df, data]:
                data['is_low_risk_hour'] = data['hour_of_day'].between(5, 21).astype(int)
                data['is_early_morning'] = (data['hour_of_day'] <= 4).astype(int)
                data['is_late_night'] = data['hour_of_day'].between(22, 23).astype(int)
                data['is_high_risk_day'] = data['day_of_week'].isin([2, 3, 4, 5]).astype(int)
                data['is_high_risk_window'] = (
                    (data['is_high_risk_day'] == 1) &
                    ((data['is_early_morning'] == 1) | (data['is_late_night'] == 1))
                ).astype(int)

            logging.info("Temporal operational threat matrix indicators updated.")
            return data
        except Exception as e:
            logging.error("Exception occurred while processing high risk window rules.")
            raise CustomException(e, sys)
    
    def add_high_risk_city_feature(self, smoothing: float, baseline_df: pd.DataFrame, data: pd.DataFrame) -> tuple:
        try:
            logging.info("Applying empirical Laplace smoothing to regional city risk metrics.")
            baseline_df = baseline_df.copy()
            data = data.copy()

            global_mean = baseline_df['is_fraud'].mean()

            city_stats = baseline_df.groupby('city').agg(
                count=('is_fraud', 'count'),
                mean=('is_fraud', 'mean')
            )

            city_stats['smoothed_score'] = (
                (city_stats['count'] * city_stats['mean']) + (smoothing * global_mean)
            ) / (city_stats['count'] + smoothing)
            
            city_risk_score = city_stats['smoothed_score']
            for data_split in [baseline_df, data]:
                data_split['city_fraud_risk_score'] = data_split['city'].map(city_risk_score).fillna(global_mean)

            logging.info("Regional demographic anomaly indices successfully applied.")
            return baseline_df, data
        except Exception as e:
            logging.error("Exception occurred while compiling localized risk vectors.")
            raise CustomException(e, sys)

    @staticmethod
    def add_age_features(data: pd.DataFrame) -> tuple:
        try:
            logging.info("Partitioning user structural demographic age classifications.")
            data = data.copy()
        ata
            data['is_high_risk_age'] = (data['age'] > 62).astype(int)
            data['is_medium_risk_age'] = (data['age'].between(43, 61)).astype(int)
            data['is_low_risk_age'] = (data['age'].between(17, 42)).astype(int)
              
            logging.info("Demographic categorization complete.")
            return data
        except Exception as e:
            logging.error("Exception occurred while splitting biological age risk segments.")
            raise CustomException(e, sys)

    @staticmethod
    def add_amount_risk_features(data: pd.DataFrame) -> tuple:
        try:
            logging.info("Flagging structural transaction ticket amount size intervals.")
            data = data.copy()

            data['is_high_risk_amt'] = (data['amt'] >= 250).astype(int)
            data['is_very_high_risk_amt'] = (data['amt'] >= 500).astype(int)
            data['is_card_testing_amt'] = (data['amt'].between(10, 25)).astype(int)
            data['is_low_risk_amt'] = (data['amt'].between(50, 100)).astype(int)
            data['is_micro_amt'] = (data['amt'] <= 5).astype(int)

            logging.info("Pricing threshold metrics created.")
            return data
        except Exception as e:
            logging.error("Exception occurred while tracking scalar amount brackets.")
            raise CustomException(e, sys)

    @staticmethod
    def add_category_risk_features(data: pd.DataFrame) -> tuple:
        try:
            logging.info("Isolating merchant category risk classification buckets.")
            data = data.copy()

            high_risk_categories = ['shopping_net', 'misc_net', 'grocery_pos']
            low_risk_categories = ['health_fitness', 'home', 'food_dining', 'kids_pets', 'personal_care']

            
            data['is_high_risk_category'] = (data['category'].isin(high_risk_categories)).astype(int)
            data['is_low_risk_category'] = (data['category'].isin(low_risk_categories)).astype(int)

            logging.info("Merchant category classifications mapped successfully.")
            return data
        except Exception as e:
            logging.error("Exception occurred while creating commercial category flags.")
            raise CustomException(e, sys)

    def initiate_feature_extraction(self, baseline_df: pd.DataFrame = None, data: pd.DataFrame = None) -> FeatureExtractionArtifact:
        try:
            logging.info("==================== Starting Feature Extraction Pipeline ====================")
            
            logging.info(f"Loading validated dataset targets from validation artifact paths.")
            baseline_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            data = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            logging.info("Enforcing strict conversion of time indexes to Timestamp types.")
            baseline_df['trans_date_trans_time'] = pd.to_datetime(baseline_df['trans_date_trans_time'])
            data['trans_date_trans_time'] = pd.to_datetime(data['trans_date_trans_time'])

            # Cascade processing functions sequentially
            baseline_df, data = self.add_time_features(baseline_df, data)
            baseline_df, data = self.add_velocity_features(baseline_df, data)
            baseline_df, data = self.add_fraud_rate(baseline_df, data)
            baseline_df, data = self.add_geo_features(baseline_df, data)
            baseline_df, data = self.add_transaction_features(baseline_df, data)
            baseline_df, data = self.add_merchant_based_features(baseline_df, data)
            baseline_df, data = self.cardholder_behaviour_features(baseline_df, data)
            baseline_df, data = self.flag_high_risk(baseline_df, data)
            baseline_df, data = self.add_high_risk_city_feature(100, baseline_df, data)
            baseline_df, data = self.add_age_features(baseline_df, data)
            baseline_df, data = self.add_amount_risk_features(baseline_df, data)
            baseline_df, data = self.add_category_risk_features(baseline_df, data)

            train_file_path = self.feature_extraction_config.training_file_path
            logging.info(f"Persisting processed engineering train matrix split out to: {train_file_path}")
            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
            baseline_df.to_csv(train_file_path, index=False, header=True)

            test_file_path = self.feature_extraction_config.testing_file_path
            logging.info(f"Persisting processed engineering test matrix split out to: {test_file_path}")
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
            data.to_csv(test_file_path, index=False, header=True)

            logging.info("Constructing complete Feature Extraction Pipeline Output Artifact object.")
            featureextractionartifact = FeatureExtractionArtifact(
                trained_file_path=self.feature_extraction_config.training_file_path,
                test_file_path=self.feature_extraction_config.testing_file_path
            )

            logging.info("==================== Feature Extraction Pipeline Success ====================")
            return featureextractionartifact 
        except Exception as e:
            logging.error("Critical failure during complete execution of feature extraction pipeline cascades.")
            raise CustomException(e, sys)
