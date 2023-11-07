#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns


# In[ ]:


credit_card_train=pd.read_csv("fraudTrain.csv")
credit_card_test=pd.read_csv("fraudTest.csv")


# In[ ]:


credit_card_train.head()


# In[ ]:


credit_card_test.head()


# In[ ]:


credit_card_train.tail()


# In[ ]:


credit_card_train.info()


# In[ ]:


credit_card_train.shape


# In[ ]:


print("no of rows",credit_card_train.shape[0])
print("no of columns",credit_card_train.shape[1])


# In[ ]:


train_data=credit_card_train.columns
test_data=credit_card_test.columns


# In[ ]:


train_data=credit_card_train.drop(['Unnamed: 0','trans_date_trans_time','cc_num','street','city','zip','dob','trans_num','lat','long','job','merch_lat','merch_long','state'
],axis=1)


# In[ ]:


test_data=credit_card_test.drop(['Unnamed: 0','trans_date_trans_time','cc_num','street','city','zip','dob','trans_num','lat','long','job','merch_lat','merch_long','state'
],axis=1)


# In[ ]:


#coverting the data from categorical from to numerical form
from sklearn.preprocessing import LabelEncoder
data = ['merchant','category','first','last','gender']

# Encode Categorical Columns
le = LabelEncoder()
train_data[data] = train_data[data].apply(le.fit_transform)
test_data[data] = test_data[data].apply(le.fit_transform)


# In[ ]:


train_data.head(10)


# In[ ]:


test_data.head(10)


# In[ ]:


train_data["is_fraud"].value_counts()


# In[ ]:


train_data.duplicated().any()


# # data visulization

# In[ ]:


train_data['is_fraud'].value_counts().plot(kind='bar', color=['green', 'salmon'])
plt.title('credit_card_data (0: Non-Fraud, 1: Fraud)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.figure(figsize=(6, 6))
plt.show()


# In[ ]:


train_data['category'].value_counts().plot(kind='bar')
plt.title('distribution of category')
plt.xlabel('category')
plt.ylabel('Count')
plt.figure(figsize=(6, 6))
plt.show()


# #splitting the data

# In[ ]:


X_train=train_data.drop('is_fraud',axis=1)
y_train=train_data["is_fraud"]


# In[ ]:


X_test=test_data.drop('is_fraud',axis=1)
y_test=test_data["is_fraud"]


# In[ ]:


X_train


# In[ ]:


y_train


# # Model training 

# # Using logistic regression 

# In[ ]:


lr = LogisticRegression()


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


y_pred=lr.predict(X_test)


# In[ ]:


data_accuracy = accuracy_score(y_test, y_pred)


# In[ ]:


print("data_accuracy",data_accuracy)


# In[ ]:





# In[ ]:





# In[ ]:




