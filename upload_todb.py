import pymongo
import pandas as pd
import json

#Retrieve information from DB
client = pymongo.MongoClient('mongodb+srv://phuchauxd12:Abcd0123@cluster0.lf8sh9p.mongodb.net/')

files = pd.read_csv('MOCK_DATA.csv')
data = files.to_dict(orient = 'records')

db = client['UserBank']
print(db)

print(db.Bank.insert_many(data))