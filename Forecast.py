import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from pymongo import MongoClient
import json

#Retrieve information from DB
mongoClient = MongoClient('mongodb+srv://userDB:InC2QuunWeQUFOCm@userdb.opquo83.mongodb.net/') # Insert a differnet mongoDB link

# Access to the mongo database and collection
db = mongoClient['UserBank']
collection = db['Bank'] 

# enter the collection name
data = db.get_collection('Bank')

#find all data from mongoDB collections
all_data = collection.find()
""" This will find the data form the mongoDB
and call the information"""

print(all_data)

#Setting up the files column
headers = ['date', 'total_income', 'total_expenses']
""" 
When we read the files from the DB, it will also return the ID aswell,
which we don't need it. """

df = pd.DataFrame(all_data)
df = df[headers]
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df = df[0:150]

#This will return the first 5 rows, to check if everything is working
print(df.head())

#Convert the data types of date
df['date'] = df['date'].dt.to_period('D')
daily_money = df.groupby('date').sum().reset_index()
daily_money['date'] = daily_money['date'].dt.to_timestamp()
print(daily_money['date'])


# #Split test for training and dataset
x = df.drop(['date','total_income'], axis=1)
y = df['total_income']

"""
This code returns the income difference between the previous day"""

daily_money['income_diff'] = daily_money['total_income'].diff()
daily_money = daily_money.dropna()

 #Sketching the scatter plot of the user overall income within a one year period 
# plt.figure(figsize=(20,7))
# plt.plot(daily_money['date'],daily_money['income_diff'], color = 'red')
# plt.plot(daily_money['date'],daily_money['total_income'])
# plt.xlabel('Date')
# plt.ylabel('Total Income($)')
# plt.legend(['Income','Income difference compared to previous month'])
# plt.title('User total income')
# plt.show()

# #Creating a supervised data 
supervised_data = daily_money.drop(['date','total_income'], axis=1)

#Set up supervised data
"""The range is the 12 months of the year."""
for i in range(1,13):
    col_name = 'month_ ' + str(i) 
    supervised_data[col_name] = supervised_data['income_diff'].shift(i)
supervised_data = supervised_data.dropna().reset_index(drop = True)
# print(supervised_data)

train_data = supervised_data[:-12]
test_data = supervised_data[-12:]

scaler = MinMaxScaler(feature_range=(-1,1))
scaler.fit(train_data)
train_data = scaler.transform(train_data)
test_data = scaler.transform(test_data)

x_train, y_train = train_data[:,1:], train_data[:,0:1]
x_test, y_test = test_data[:,1:], train_data[:,0:1]
y_train = y_train.ravel()
y_test = y_test.ravel()

"""
Even though there are training and testing variables.
We haven't actually train any dataset.
As we didn't use the train_test_split function to do so.
We just named it as that way for easy references"""


#Makijng income prediction
df = daily_money['date'][-12:].reset_index(drop = True)
predict_df = pd.DataFrame(df)

cor_income = daily_money['total_income'][:].to_list()
# print(cor_income)

#Linear Regression
""" Linear Regression
is one of the many other ways that we can use for model prediction"""
lr = LinearRegression()
lr.fit(x_train, y_train)
lr_predict = lr.predict(x_test)
lr_predict

lr_predict = lr_predict.reshape(-1,1)
lr_pre_test_set  = np.concatenate([lr_predict, x_test], axis=1)
lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)

result = []
for index in range(0, len(lr_pre_test_set)):
    result.append(lr_pre_test_set[index][0] + cor_income[index])
lr_pre_series = pd.Series(result, name = 'Linear Prediction')
predict_df = predict_df.merge(lr_pre_series, left_index=True, right_index= True)

""" 
We compare the predicted income with the actual income"""

#Visualisation of prediction plt.figure(figsize=(20,7))
    #Actual Income
# plt.figure(figsize=(20,7))
plt.plot(daily_money['date'],daily_money['total_income'])

    #Predicted Income
plt.plot(predict_df['date'],predict_df['Linear Prediction'])
plt.xlabel('Date')
plt.ylabel('Total Income($)')
plt.title('User predicted Income')
plt.legend(['Actual','Predicted'])
plt.show()
