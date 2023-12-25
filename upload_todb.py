import pymongo
import pandas as pd
import json

#Retrieve information from DB
client = pymongo.MongoClient('mongodb+srv://phuchauxd12:Abcd0123@cluster0.lf8sh9p.mongodb.net/')

#Read/Setup the files
files = pd.read_csv('MOCK_DATA.csv')
data = files.to_dict(orient = 'records')


#Create a collection 
db = client['UserBank']
print(db)

#Insert file to database
print(db.Bank.insert_many(data))


#Updating records