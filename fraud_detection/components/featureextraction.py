
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
    def add_time_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple: 
        import logging # Note: If you have a custom logger in your project, import that instead at the top of your file.
        
        logging.info("Starting time feature extraction for train and test sets.")
        
        train_df = train_df.copy()
        test_df = test_df.copy()
            
        train_df['_split'] = 'train'
        test_df['_split'] = 'test'

        # Combine for bulk processing
        df = pd.concat([train_df, test_df])
        
        logging.info("Converting 'trans_date_trans_time' to datetime objects.")
        df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
        
        logging.info("Extracting hour, day of week, month, and weekend flags.")
        df['hour_of_day'] = df['trans_date_trans_time'].dt.hour
        df['day_of_week'] = df['trans_date_trans_time'].dt.day_of_week
        df['month'] = df['trans_date_trans_time'].dt.month
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

        # Split back into train and test
        train_df = df[df['_split'] == 'train'].drop(columns=['_split'])
        test_df = df[df['_split'] == 'test'].drop(columns=['_split'])

        logging.info("Time feature extraction completed successfully.")
        
        return train_df, test_df

    @staticmethod
    def add_velocity_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        try:
            logging.info("Calculating transaction velocity windows (1h, 6h, 24h).")
            train_df = train_df.copy()
            test_df = test_df.copy()
            
            train_df['_split'] = 'train'
            test_df['_split'] = 'test'

            df = pd.concat([train_df, test_df])
            df = df.sort_values(['cc_num', 'trans_date_trans_time']).reset_index(drop=True)
            df = df.set_index('trans_date_trans_time')
            
            for window, col in [('1h', 'txn_count_1h'), ('6h', 'txn_count_6h'), ('24h', 'txn_count_24h')]:
                df[col] = (
                    df.groupby('cc_num')['amt']   
                    .rolling(window)             
                    .count()
                    .reset_index(level=0, drop=True)
                )

            df = df.reset_index()
            
            train_df = df[df['_split'] == 'train'].drop(columns='_split')
            test_df = df[df['_split'] == 'test'].drop(columns="_split")

            logging.info("Transaction velocity features calculated successfully.")
            return train_df, test_df
        except Exception as e:
            logging.error("Exception occurred while adding velocity features.")
            raise CustomException(e, sys)
    
    @staticmethod
    def add_fraud_rate(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        try:
            logging.info("Calculating historical target-encoded fraud rates for time partitions.")
            train_df = train_df.copy()
            test_df = test_df.copy()

            global_fraud_rate = train_df['is_fraud'].mean()
            hour_fraud_rate = train_df.groupby('hour_of_day')['is_fraud'].mean()
            day_fraud_rate = train_df.groupby('day_of_week')['is_fraud'].mean()
            month_fraud_rate = train_df.groupby('month')['is_fraud'].mean()

            for data_split in [train_df, test_df]:
                data_split['hour_fraud_rate'] = data_split['hour_of_day'].map(hour_fraud_rate).fillna(global_fraud_rate)
                data_split['day_fraud_rate'] = data_split['day_of_week'].map(day_fraud_rate).fillna(global_fraud_rate)
                data_split['month_fraud_rate'] = data_split['month'].map(month_fraud_rate).fillna(global_fraud_rate)
                
            logging.info("Time-based fraud rate maps generated and applied.")
            return train_df, test_df
        except Exception as e:
            logging.error("Exception occurred while adding time-based fraud rate attributes.")
            raise CustomException(e, sys)
    
    @staticmethod
    def haversine_miles(df: pd.DataFrame) -> np.ndarray:
        try:
            R = 6371
            lat1 = np.radians(df['lat'])
            lon1 = np.radians(df['long'])
            lat2 = np.radians(df['merch_lat'])
            lon2 = np.radians(df['merch_long'])

            dlat = lat2 - lat1
            dlong = lon2 - lon1

            a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong / 2)**2
            d = 2 * np.arcsin(np.sqrt(a))
            return R * d
        except Exception as e:
            logging.error("Exception occurred inside backend Haversine computation.")
            raise CustomException(e, sys)

    @classmethod
    def add_geo_features(cls, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        try:
            logging.info("Extracting geospatial mapping profiles and sequential distance transitions.")
            train_df = train_df.copy()
            test_df = test_df.copy()

            train_df['_split'] = 'train'
            test_df['_split'] = 'test'

            for df in [train_df, test_df]:
             df['home_to_merchant_dist'] = cls.haversine_miles(df)
             df['likely_diff_state'] = (df['home_to_merchant_dist'] > 200)

             df = pd.concat([train_df, test_df])
             df = df.sort_values(['cc_num', 'trans_date_trans_time']).reset_index(drop=True)

             df['prev_distance'] = df.groupby('cc_num')['home_to_merchant_dist'].shift(1)
             df['dist_change_prev_txn'] = np.abs(df['home_to_merchant_dist'] - df['prev_distance']).fillna(0)
             df.drop(columns=['prev_distance'], inplace=True)

             df['prev_trans_date_trans_time'] = df.groupby('cc_num')['trans_date_trans_time'].shift(1)

             df['time_since_prev_txn'] = np.abs(
                (df['trans_date_trans_time'] - df['prev_trans_date_trans_time']).dt.total_seconds()
            ).fillna(0)
             
             df.drop(columns=['prev_trans_date_trans_time'], inplace=True)

            train_df = df[df['_split'] == 'train'].drop(columns=['_split'])
            test_df = df[df['_split'] == 'test'].drop(columns=['_split'])

            logging.info("Geospatial tracking matrices successfully updated.")
            return train_df, test_df
        
        except Exception as e:
            logging.error("Exception occurred while building geographic features.")
            raise CustomException(e, sys)
    
    @staticmethod
    def add_transaction_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        try:
            logging.info("Evaluating scalar transaction characteristics and cardholder Z-Scores.")
            train_df = train_df.copy()
            test_df = test_df.copy()

            cardholder_stats = train_df.groupby('cc_num')['amt'].agg(['mean', 'std']).rename(
                columns={'mean': "amt_mean", 'std': 'amt_std'}
            )
            
            for data_split in [train_df, test_df]:
                merged = data_split.join(cardholder_stats, on='cc_num')
                data_split['amt_z_score'] = ((data_split['amt'] - merged['amt_mean']) / merged['amt_std']).fillna(0)

            category_stats = train_df.groupby('category')['amt'].agg(['mean', 'std']).rename(
                columns={'mean': 'cat_amt_mean', 'std': 'cat_amt_std'}
            )

            for df_split in [train_df, test_df]:
                merged = df_split.join(category_stats, on='category')
                df_split['amt_zscore_category'] = (
                    (merged['amt'] - merged['cat_amt_mean']) / merged['cat_amt_std']
                ).fillna(0)
            
            for df_split in [train_df, test_df]:
                df_split['is_round_1'] = (df_split['amt'] % 1 == 0).astype(int)
                df_split['is_round_10'] = (df_split['amt'] % 10 == 0).astype(int)
                
            logging.info("Transaction value variance tracking complete.")
            return train_df, test_df
        except Exception as e:
            logging.error("Exception occurred while extracting pricing variance signals.")
            raise CustomException(e, sys)
    
    @staticmethod
    def add_merchant_based_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        try:
            logging.info("Compiling merchant and network risk profile aggregates.")
            train_df = train_df.copy()
            test_df = test_df.copy()

            global_fraud_rate = train_df['is_fraud'].mean()
           
            merchant_fraud_rate = (
                train_df.groupby('merchant')['is_fraud']
                .mean()
                .rename('merchant_fraud_rate')
            ) * 100

            for df_split in [train_df, test_df]:
                df_split['merchant_fraud_rate'] = (
                    df_split['merchant']
                    .map(merchant_fraud_rate)
                    .fillna(global_fraud_rate)  
                )

            category_fraud_rate = (
                train_df.groupby('category')['is_fraud']
                .mean()
                .rename('category_fraud_rate')
            ) * 100

            for df_split in [train_df, test_df]:
                df_split['category_fraud_rate'] = (
                    df_split['category']
                    .map(category_fraud_rate)
                    .fillna(global_fraud_rate)  
                )
                
            cardholder_merchant_counts = train_df.groupby(['cc_num', 'merchant']).size().rename('cc_merchant_txn_count')

            for data_split in [train_df, test_df]:
                data_split['cc_merchant_txn_count'] = (
                    data_split.set_index(['cc_num', 'merchant'])
                    .index
                    .map(cardholder_merchant_counts)
                    .fillna(0)
                    .values
                )
              
            logging.info("Merchant vulnerability tracking features appended successfully.")
            return train_df, test_df
        except Exception as e:
            logging.error("Exception occurred while aggregating merchant risk records.")
            raise CustomException(e, sys)
    
    @staticmethod
    def cardholder_behaviour_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        try:
            logging.info("Assessing account maturity, usage density profiles, and spending entropy.")

            train_df = train_df.copy()
            test_df = test_df.copy()

            first_txn_date = (
                train_df.groupby('cc_num')['trans_date_trans_time']
                .min()
                .rename('first_txn_date')
            )

            for df_split in [train_df, test_df]:
                df_split['account_age_days'] = (
                    df_split['trans_date_trans_time']
                    - df_split['cc_num'].map(first_txn_date)
                ).dt.days.fillna(0)

            top_category = (
                train_df.groupby('cc_num')['category']
                .agg(lambda x: x.value_counts().index[0])
                .rename('top_category')
            )

            for df_split in [train_df, test_df]:
                df_split['is_typical_category'] = (
                    df_split['category'] == df_split['cc_num'].map(top_category)
                ).astype(int)

            cardholder_txn_count = train_df.groupby('cc_num').size()

            cardholder_date_range = (
                train_df.groupby('cc_num')['trans_date_trans_time']
                .agg(lambda x: (x.max() - x.min()).days + 1) 
            )

            card_usage_freq = (
                (cardholder_txn_count / cardholder_date_range)
                .rename('card_usage_freq')
            )

            global_usage_freq = card_usage_freq.mean()

            for df_split in [train_df, test_df]:
                df_split['card_usage_freq'] = (
                    df_split['cc_num']
                    .map(card_usage_freq)
                    .fillna(global_usage_freq)
                )
            
            category_probability = (
                train_df.groupby(['cc_num', 'category'])['amt'].count()
                / train_df.groupby('cc_num')['category'].count()
            ).reset_index(name='category_probability')

            category_probability['entropy_component'] = (
                category_probability['category_probability']
                * np.log2(category_probability['category_probability'])
            )

            customer_category_entropy = (
                -category_probability.groupby('cc_num')['entropy_component'].sum()
            )
            global_customer_entropy = customer_category_entropy.median()

            for data_split in [train_df, test_df]:
                data_split['customer_category_entropy'] = (
                    data_split['cc_num'].map(customer_category_entropy).fillna(global_customer_entropy)
                )

            logging.info("Cardholder long-term behavior profiles fully established.")
            return train_df, test_df
        except Exception as e:
            logging.error("Exception occurred while building historical cardholder profile traits.")
            raise CustomException(e, sys)
    
    @staticmethod
    def flag_high_risk(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        try:
            logging.info("Mapping critical operating hour exceptions (Early Morning, Late Night, Weekends).")
            train_df = train_df.copy()
            test_df = test_df.copy()

            for df in [train_df, test_df]:
                df['is_low_risk_hour'] = df['hour_of_day'].between(5, 21).astype(int)
                df['is_early_morning'] = (df['hour_of_day'] <= 4).astype(int)
                df['is_late_night'] = df['hour_of_day'].between(22, 23).astype(int)
                df['is_high_risk_day'] = df['day_of_week'].isin([2, 3, 4, 5]).astype(int)
                df['is_high_risk_window'] = (
                    (df['is_high_risk_day'] == 1) &
                    ((df['is_early_morning'] == 1) | (df['is_late_night'] == 1))
                ).astype(int)

            logging.info("Temporal operational threat matrix indicators updated.")
            return train_df, test_df
        except Exception as e:
            logging.error("Exception occurred while processing high risk window rules.")
            raise CustomException(e, sys)
    
    def add_high_risk_city_feature(self, smoothing: float, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        try:
            logging.info("Applying empirical Laplace smoothing to regional city risk metrics.")
            train_df = train_df.copy()
            test_df = test_df.copy()

            global_mean = train_df['is_fraud'].mean()

            city_stats = train_df.groupby('city').agg(
                count=('is_fraud', 'count'),
                mean=('is_fraud', 'mean')
            )

            city_stats['smoothed_score'] = (
                (city_stats['count'] * city_stats['mean']) + (smoothing * global_mean)
            ) / (city_stats['count'] + smoothing)
            
            city_risk_score = city_stats['smoothed_score']
            for data_split in [train_df, test_df]:
                data_split['city_fraud_risk_score'] = data_split['city'].map(city_risk_score).fillna(global_mean)

            logging.info("Regional demographic anomaly indices successfully applied.")
            return train_df, test_df
        except Exception as e:
            logging.error("Exception occurred while compiling localized risk vectors.")
            raise CustomException(e, sys)

    @staticmethod
    def add_age_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        try:
            logging.info("Partitioning user structural demographic age classifications.")
            train_df = train_df.copy()
            test_df = test_df.copy()
            for data_split in [train_df, test_df]:
                data_split['is_high_risk_age'] = (data_split['age'] > 62).astype(int)
                data_split['is_medium_risk_age'] = (data_split['age'].between(43, 61)).astype(int)
                data_split['is_low_risk_age'] = (data_split['age'].between(17, 42)).astype(int)
              
            logging.info("Demographic categorization complete.")
            return train_df, test_df
        except Exception as e:
            logging.error("Exception occurred while splitting biological age risk segments.")
            raise CustomException(e, sys)

    @staticmethod
    def add_amount_risk_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        try:
            logging.info("Flagging structural transaction ticket amount size intervals.")
            train_df = train_df.copy()
            test_df = test_df.copy()

            for df in [train_df, test_df]:
                df['is_high_risk_amt'] = (df['amt'] >= 250).astype(int)
                df['is_very_high_risk_amt'] = (df['amt'] >= 500).astype(int)
                df['is_card_testing_amt'] = (df['amt'].between(10, 25)).astype(int)
                df['is_low_risk_amt'] = (df['amt'].between(50, 100)).astype(int)
                df['is_micro_amt'] = (df['amt'] <= 5).astype(int)

            logging.info("Pricing threshold metrics created.")
            return train_df, test_df
        except Exception as e:
            logging.error("Exception occurred while tracking scalar amount brackets.")
            raise CustomException(e, sys)

    @staticmethod
    def add_category_risk_features(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple:
        try:
            logging.info("Isolating merchant category risk classification buckets.")
            train_df = train_df.copy()
            test_df = test_df.copy()

            high_risk_categories = ['shopping_net', 'misc_net', 'grocery_pos']
            low_risk_categories = ['health_fitness', 'home', 'food_dining', 'kids_pets', 'personal_care']

            for df in [train_df, test_df]:
                df['is_high_risk_category'] = (df['category'].isin(high_risk_categories)).astype(int)
                df['is_low_risk_category'] = (df['category'].isin(low_risk_categories)).astype(int)

            logging.info("Merchant category classifications mapped successfully.")
            return train_df, test_df
        except Exception as e:
            logging.error("Exception occurred while creating commercial category flags.")
            raise CustomException(e, sys)

    def initiate_feature_extraction(self, train_df: pd.DataFrame = None, test_df: pd.DataFrame = None) -> FeatureExtractionArtifact:
        try:
            logging.info("==================== Starting Feature Extraction Pipeline ====================")
            
            logging.info(f"Loading validated dataset targets from validation artifact paths.")
            train_df = pd.read_csv(self.data_validation_artifact.valid_train_file_path)
            test_df = pd.read_csv(self.data_validation_artifact.valid_test_file_path)

            logging.info("Enforcing strict conversion of time indexes to Timestamp types.")
            train_df['trans_date_trans_time'] = pd.to_datetime(train_df['trans_date_trans_time'])
            test_df['trans_date_trans_time'] = pd.to_datetime(test_df['trans_date_trans_time'])

            # Cascade processing functions sequentially
            train_df, test_df = self.add_time_features(train_df, test_df)
            train_df, test_df = self.add_velocity_features(train_df, test_df)
            train_df, test_df = self.add_fraud_rate(train_df, test_df)
            train_df, test_df = self.add_geo_features(train_df, test_df)
            train_df, test_df = self.add_transaction_features(train_df, test_df)
            train_df, test_df = self.add_merchant_based_features(train_df, test_df)
            train_df, test_df = self.cardholder_behaviour_features(train_df, test_df)
            train_df, test_df = self.flag_high_risk(train_df, test_df)
            train_df, test_df = self.add_high_risk_city_feature(100, train_df, test_df)
            train_df, test_df = self.add_age_features(train_df, test_df)
            train_df, test_df = self.add_amount_risk_features(train_df, test_df)
            train_df, test_df = self.add_category_risk_features(train_df, test_df)

            train_file_path = self.feature_extraction_config.training_file_path
            logging.info(f"Persisting processed engineering train matrix split out to: {train_file_path}")
            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)
            train_df.to_csv(train_file_path, index=False, header=True)

            test_file_path = self.feature_extraction_config.testing_file_path
            logging.info(f"Persisting processed engineering test matrix split out to: {test_file_path}")
            os.makedirs(os.path.dirname(test_file_path), exist_ok=True)
            test_df.to_csv(test_file_path, index=False, header=True)

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
