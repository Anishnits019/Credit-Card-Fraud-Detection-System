import pandas as pd
import os
import pymongo
from dotenv import load_dotenv
from diabetes.entity.config_entity import DataStoreConfig
from diabetes.exception.exception import CustomException
load_dotenv()  
class DataStore():
     def __init__(self,data_store_config:DataStoreConfig):
          self.data_store_config=data_store_config

     def initiate_data_store(self,df:pd.DataFrame):
     
          client=pymongo.MongoClient(os.getenv("MONGO_DB_URL"))

          db_name=self.data_store_config.database_name
          coll_name=self.data_store_config.collection_name
          collection=client[db_name][coll_name]

          records=df.to_dict(orient="records")
          try:
               if(len(records)>0):
                 collection.insert_many(records)
                 print("The diabetes is successfully inserted into mongodb")
               else:
                 print("DataFrame is empty. No records to insert.")

          except Exception as e:
            raise CustomException
     
if __name__=="__main__":
      data_store_config=DataStoreConfig()
      data_store= DataStore(data_store_config)

      df=pd.read_csv('/Users/anish/Documents/Diabetes/Notebooks/diabetes.csv',sep=';')
      data_store.initiate_data_store(df)
      
    
    
