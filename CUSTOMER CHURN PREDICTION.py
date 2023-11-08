#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 


# In[2]:


churn=pd.read_csv("Churn_Modelling.csv")


# In[3]:


churn.head()


# In[4]:


churn.tail()


# In[5]:


churn.info()


# In[6]:


churn.columns


# In[7]:


churn.shape


# In[8]:


print("no of rows",churn.shape[0])
print("no of columns",churn.shape[1])


# In[9]:


churn.dtypes


# Checking null&missing values

# In[10]:


churn.isnull().sum()


# Removing irrelevant columns

# In[11]:


churn_data=churn.drop(['RowNumber','CustomerId','Surname'],axis=1)


# In[12]:


churn_data.head(10)


# Converting catogorical data into numerical data

# In[13]:


data = ['Geography','Gender']

# Encode Categorical Columns
le = LabelEncoder()
churn_data[data] = churn_data[data].apply(le.fit_transform)


# In[14]:


churn_data.head(10)


# In[15]:


churn_data.describe()


# In[16]:


churn_data["Exited"].value_counts()


# # Data visualization

# In[17]:


churn_data['Exited'].value_counts().plot(kind='bar')
plt.xlabel('Exit')
plt.ylabel('Count')
plt.figure(figsize=(6, 6))
plt.show()


# In[18]:


churn_data.hist(figsize=(15,12))
plt.title('Class Distribution')
plt.show()


# In[19]:


plt.figure(figsize=(15,5))
sns.heatmap(churn_data.corr(),annot=True)


# Splitting the training model using logistic regression

# In[20]:


X = churn_data.drop(['Exited'],axis=1)
y = churn_data['Exited']


# In[21]:


print(X)


# In[22]:


print(y)


# In[23]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=0)


# In[24]:


X_train


# In[25]:


y_train


# In[26]:


print(X,X_train.shape,X_test.shape,y_train.shape,y_test.shape)


# In[27]:


rfc=RandomForestClassifier(n_estimators = 100) 


# In[28]:


rfc.fit(X_train,y_train)


# In[29]:


y_pred = rfc.predict(X_test)


# In[30]:


accuracy=accuracy_score(y_test,y_pred)


# In[31]:


print("ACCURACY OF THE MODEL:",accuracy)


# In[32]:


from sklearn.metrics import precision_score,recall_score,f1_score


# In[33]:


precision_score(y_test,y_pred)


# In[34]:


recall_score(y_test,y_pred)


# In[35]:


f1_score(y_test,y_pred)


# In[36]:


import math 
math.sqrt(len(y_test))


# In[37]:


from sklearn.neighbors import KNeighborsClassifier
Classifiers=KNeighborsClassifier(n_neighbors=57,p=2,metric="euclidean")


# In[38]:


Classifiers.fit(X_train,y_train)


# In[39]:


y_pred=Classifiers.predict(X_test)


# In[40]:


accuracy=accuracy_score(y_test,y_pred)


# In[41]:


print(accuracy)


# In[ ]:




