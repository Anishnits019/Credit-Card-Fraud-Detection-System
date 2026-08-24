from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

url = "mongodb://localhost:27017/"
client=MongoClient(url,server_api=ServerApi('1'))

db=client['Credit_Card_Transaction']
collection=db['Fraud_Detection']
try:
    client.admin.command('ping')
    print("Successfully connected to local MongoDB!")
except Exception as e:
    print(f"Connection failed: {e}")

   