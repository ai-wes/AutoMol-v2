from pymongo import MongoClient
import logging
import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger('pymongo')
logger.setLevel(logging.INFO)


from pymongo.mongo_client import MongoClient

uri = "mongodb+srv://weslagarde:Beaubeau2023!@cluster0.zpowdpt.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri)
db = client['molecule_db']
molecules_collection = db['molecules']

# Send a ping to confirm a successful connection
try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print(e)