#!/usr/bin/env python
# coding: utf-8

# In[1]:


##Collecting the data
import os
import pandas as pd
import tensorflow as tf


# In[2]:


df = pd.read_csv("C:\\Users\\Shreya\\Downloads\\NFLX.csv")


# In[3]:


df.head()


# In[7]:


df.shape


# In[8]:


df1=df.reset_index()['Close']


# In[9]:


df1.head()


# In[10]:


df1.shape


# In[11]:


import matplotlib.pyplot as plt
plt.plot(df1)


# In[12]:


#LSTM are sensitive to the scale of data


# In[13]:


import numpy as np


# In[14]:


df1


# In[19]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))


# In[20]:


df1


# In[21]:


##Train -test split is done using cross validation or random seed


# In[22]:


training_size=int(len(df1)*0.65)
test_size=len(df1) -training_size
train_data,test_data=df1[0:training_size,:],df1[training_size:len(df1), :1]


# In[23]:


training_size,test_size


# In[24]:


# Preprocessing to create input-output pairs for an LSTM model

import numpy as np
#convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)


# In[25]:


time_step = 70
X_train, y_train = create_dataset(train_data,time_step)
X_test,ytest = create_dataset(test_data,time_step)


# In[26]:


print(X_train.shape), print(y_train.shape)


# In[27]:


print(X_test.shape), print(ytest.shape)


# In[28]:


#3d
# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)


# In[29]:


#create the stacked LSTM model
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense 
from tensorflow.keras.layers import LSTM


# In[30]:


model=Sequential()
model.add(LSTM(50,return_sequences=True,input_shape=(70,1))) 
model.add(LSTM(50,return_sequences=True)) 
model.add(LSTM(50)) 
model.add(Dense(1)) 
model.compile(loss='mean_squared_error',optimizer='adam')


# In[31]:


model.summary()


# In[32]:


model.fit(X_train,y_train,validation_data=(X_test,ytest),epochs=50,batch_size=64,verbose=1)


# In[33]:


tf.__version__


# In[34]:


train_predict=model.predict(X_train)
test_predict=model.predict(X_test)


# In[35]:


#transfroming back to original form
train_predict=scaler.inverse_transform(train_predict)
test_predict=scaler.inverse_transform(test_predict)


# In[36]:


##Calculating RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error
math.sqrt(mean_squared_error(y_train,train_predict))


# In[37]:


##Test data RMSE
math.sqrt(mean_squared_error(ytest,test_predict))


# In[45]:


look_back=70
trainPredictPlot =np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np. nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(df1)-1, :] = test_predict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()


# In[46]:


len(test_data)


# In[47]:


x_input=test_data[18:].reshape(1,-1)
x_input.shape


# In[48]:


temp_input=list(x_input)
temp_input=temp_input[0].tolist()


# In[49]:


from numpy import array

# Initialize an empty list for the first output
first_output = []
n_steps=70
i = 0
while(i<18):
    
    if len(temp_input) > 70:
        #print(temp_input)
        x_input=array(temp_input[1:])  # Remove the first element
        print("{} day input: {}".format(i,x_input))
        x_input=x_input.reshape((1,-1))
        x_input=x_input.reshape((1, n_steps, 1))
        #print(x_input)
        yhat = model.predict(x_input, verbose=0)
        print("{} day output: {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input=temp_input[1:]
        #print(temp_input)
        first_output.extend(yhat.tolist())
        i=i+1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        print(yhat[0])
        temp_input.extend(yhat[0].tolist())
        print(len(temp_input))
        first_output.extend(yhat.tolist())
        i = i + 1
        
        
        
print(first_output)


# In[50]:


day_new=np.arange(1,71)
day_pred=np.arange(71,89)


# In[51]:


import matplotlib.pyplot as plt


# In[52]:


len(df1)-70


# In[53]:


plt.plot(day_new,scaler.inverse_transform(df1[181:]))
plt.plot(day_pred,scaler.inverse_transform(first_output))


# # This way we can predict stock prices using LSTM
