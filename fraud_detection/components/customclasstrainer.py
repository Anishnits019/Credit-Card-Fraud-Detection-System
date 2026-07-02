import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
class CustomFeatureEngineering(BaseEstimator, TransformerMixin):
         
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        X.drop(columns='id', inplace=True)
        X['age'] = (X['age'] / 365.0).round(2) 

        low_ap_hi = X['ap_hi'].quantile(0.001)
        high_ap_hi = X['ap_hi'].quantile(0.999)
        X = X[(X['ap_hi'] >= low_ap_hi) & (X['ap_hi'] <= high_ap_hi)]

        low_ap_lo = X['ap_lo'].quantile(0.001)
        high_ap_lo = X['ap_lo'].quantile(0.999)
        X = X[(X['ap_lo'] >= low_ap_lo) & (X['ap_lo'] <= high_ap_lo)]

        low_height = X['height'].quantile(0.001)
        high_height = X['height'].quantile(0.999)
        X = X[(X['height'] >= low_height) & (X['height'] <= high_height)]

        low_weight = X['weight'].quantile(0.001)
        high_weight = X['weight'].quantile(0.999)
        X = X[(X['weight'] >= low_weight) & (X['weight'] <= high_weight)]

        X['bmi'] = (X['weight'] / ((X['height'] / 100) ** 2)).round(2)
        X['bmi_category'] = pd.cut(
            X['bmi'], bins=[0, 18.5, 25, 30, 1000],
            labels=['Underweight', 'Normal', 'Overweight', 'Obese']
        )
        X=X.reset_index(drop=True)
        X = X.reindex(columns=['age','gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'bmi_category','cholesterol','gluc','smoke','alco','active','cardio'])
        return X
                