import pymongo
import pandas as pd
import json

#Retrieve information from DB
client = pymongo.MongoClient('mongodb+srv://userDB:InC2QuunWeQUFOCm@userdb.opquo83.mongodb.net/')

files = pd.read_csv('MOCK_DATA.csv')
data = files.to_dict(orient = 'records')

db = client['UserBank']
print(db)

print(db.Bank.insert_many(data))