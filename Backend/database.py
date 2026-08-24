from fastapi import FastAPI
from pymongo import MongoClient
from pymongo import MongoClient
import os

client = MongoClient(os.getenv("MONGODB_URI"))
db = client["Credit_Card_Transaction"]
collection = db["Fraud_Detection"]