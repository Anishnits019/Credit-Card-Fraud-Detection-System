from sklearn.metrics import f1_score,precision_score,recall_score
from fraud_detection.entity.artifact_entity import ClassificationMetricArtifact
from fraud_detection.exception.exception import CustomException
import pandas as pd
import numpy as np
import sys

def calculate_metrics(metrics,test_desc:str,result:pd.DataFrame,roc_auc:float,average_precision:float)->pd.DataFrame:
        total=len(result)
        true_positive=np.sum((result['prob']==1)&(result['actual']==1))
        false_positive=np.sum((result['prob']==1)&(result['actual']==0))
        false_negative=np.sum((result['prob']==0)&(result['actual']==1))
        true_negative=np.sum((result['prob']==0)&( result['actual']==0))

        precision=true_positive/(true_positive+false_positive)
        recall=true_positive/(true_positive+false_negative)
        f1_score=(2*precision*recall)/(precision+recall)
        f2_score=(5*precision*recall)/(4*precision+recall)
        fraud_caught_pct=(true_positive/(true_positive+false_negative))*100
        fraud_missed=(false_negative/(true_positive+false_negative))*100
        flagged=((true_positive+false_positive)/total)*100

        new_series=pd.Series({
          'ro-auc':round(roc_auc,2),
          'average_precision':round(average_precision,2),
          'precision':    round(precision, 4),
          'recall':       round(recall, 4),
          'f1_score':     round(f1_score, 4),
          'f2_score':     round(f2_score, 4),
          'Fraud Caught %': round(fraud_caught_pct, 2),
          'Fraud Missed %': round(fraud_missed, 2),
          'Flagged':      round(flagged, 2)
             }).rename(test_desc).to_frame()
        
        return pd.concat([metrics,new_series],axis=1)

 
