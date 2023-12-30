import warnings
from fastapi import APIRouter, FastAPI, HTTPException
from bson import ObjectId, objectid
from pymongo import MongoClient
import json
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.linear_model import LinearRegression
from pymongo import MongoClient
warnings.filterwarnings('ignore')


#Link to client
router = APIRouter(tags=["Transaction Prediction"])
client = MongoClient(host="mongodb+srv://phuchauxd12:Abcd0123@cluster0.lf8sh9p.mongodb.net/").get_database("dev")
db = client.get_collection('TRANSACTION_HISTORY')

#find all data from mongoDB collections
all_data = db.find()
""" This will find the data form the mongoDB
and call the information"""


#Setting up the files column
headers = ['transaction_date', 'Amount']

""" 
When we read the files from the DB, it will also return the ID aswell,
which we don't need it. """

df = pd.DataFrame(all_data)
df = df[headers]
df['Amount_diff'] = df['Amount'].diff()
df = df[['transaction_date','Amount_diff','Amount']].dropna()
print(df)

# #Split x and y
end_value = df.tail(1)
x = df.drop(['transaction_date','Amount'], axis=1)
y = df['Amount']


print(end_value)

def prediction(number):
    lr = LinearRegression()
    lr.fit(x,y)
    predict = lr.predict(number)
    return predict

# print(prediction([[7]]))

def multi_step_prediction():
    pass

@router.get('/')
def expenses_prediction():
    return {'Prediction': prediction([[10]])}

# @router.get('/income_prediction')
# def expenses_prediction():
#     prediction = lr.score(x_train, y_train)
#     drop_date = predict_df.drop('date',axis=1)
#     if prediction > 0.5:
#         return 'High Accuracy Score', 'Accuracy of the Prediction:',prediction,'Predicted Expenses are:', drop_date 
#     else:
#         return 'Low prediction Score','Accuracy of the Prediction:',prediction, 'Predicted Expenses are:', drop_date
    

