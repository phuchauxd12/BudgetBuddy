
# train_data = supervised_data[:-12]
# test_data = supervised_data[-12:]

# scaler = MinMaxScaler(feature_range=(-1,1))
# scaler.fit(train_data)
# train_data = scaler.transform(train_data)
# test_data = scaler.transform(test_data)

# x_train, y_train = train_data[:,1:], train_data[:,0:1]
# x_test, y_test = test_data[:,1:], train_data[:,0:1]
# y_train = y_train.ravel()
# y_test = y_test.ravel()

# """
# Even though there are training and testing variables.
# We haven't actually train any dataset.
# As we didn't use the train_test_split function to do so.
# We just named it as that way for easy references"""


# #Makijng income prediction
# df = daily_money['date'][-12:].reset_index(drop = True)
# predict_df = pd.DataFrame(df)

# cor_income = daily_money['total_expenses'][:].to_list()
# # print(cor_income)

# #Linear Regression
# """ Linear Regression
# is one of the many other ways that we can use for model prediction"""
# lr = LinearRegression()
# lr.fit(x_train, y_train)
# lr_predict = lr.predict(x_test)
# lr_predict

# lr_predict = lr_predict.reshape(-1,1)
# lr_pre_test_set  = np.concatenate([lr_predict, x_test], axis=1)
# lr_pre_test_set = scaler.inverse_transform(lr_pre_test_set)

# result = []
# for index in range(0, len(lr_pre_test_set)):
#     result.append(lr_pre_test_set[index][0] + cor_income[index])
# lr_pre_series = pd.Series(result, name = 'Linear Prediction')
# predict_df = predict_df.merge(lr_pre_series, left_index=True, right_index= True)

# """ 
# We compare the predicted income with the actual income"""

# #Visualisation of prediction plt.figure(figsize=(20,7))
#     #Actual Income
# # plt.figure(figsize=(20,7))
# plt.plot(daily_money['date'],daily_money['total_expenses'])

#     #Predicted Income
# plt.plot(predict_df['date'],predict_df['Linear Prediction'])
# plt.xlabel('Date')
# plt.ylabel('Total Income($)')
# plt.title('User predicted Income')
# plt.legend(['Actual','Predicted'])
# plt.show()
