#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder 


# In[2]:


spam_data=pd.read_csv("spam.csv",encoding='latin-1')


# In[3]:


spam_data.head(10)


# In[4]:


spam_data.tail(10)


# In[5]:


spam_data.info()


# In[6]:


spam_data.columns


# #Removing irrelevant columns

# In[7]:


spam_data.drop(columns = ['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace = True)
spam_data.head()


# # Convert 'ham' to 0 and 'spam' to 1 directly in the 'v1' column

# In[8]:


spam_data['v1'] = spam_data['v1'].apply(lambda x: 1 if x == 'spam' else 0)


# In[9]:


spam_data.rename(columns = {'v1': 'Category', 'v2': 'Message'}, inplace=True)


# In[10]:


spam_data.head(10)


# In[11]:


spam_data=spam_data.drop_duplicates()
spam_data.duplicated().sum()


# In[12]:


# Check Missing Values
spam_data.isnull().sum()


# In[13]:


spam_data.shape


# In[14]:


spam_data['Category'].value_counts()


# #visualization

# In[15]:


spam_data['Category'].value_counts().plot(kind='bar')
plt.xlabel('Category')
plt.ylabel('Count')
plt.figure(figsize=(10, 10))
plt.show()


# In[16]:


X=spam_data.Message
y=spam_data.Category


# In[17]:


print(X)


# In[18]:


print(y)


# In[19]:


#Splitting the data set.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[20]:


#Converting the text files into numerical feature vectors.
from sklearn.feature_extraction.text import CountVectorizer

count_vect = CountVectorizer()
count = count_vect.fit_transform(X_train.values)


# In[21]:


count#


# In[22]:


#Here we can even reduce the weightage of dataset of more common words using TF-IDF
from sklearn.feature_extraction.text import TfidfTransformer

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(count)
X_train_tfidf.shape


# In[23]:


print(count)


# In[24]:


from sklearn.naive_bayes import MultinomialNB
mnb= MultinomialNB()


# In[25]:


mnb.fit(X_train_tfidf,y_train.values )


# In[26]:


from sklearn.pipeline import Pipeline
   
text_mnb = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('mnb', MultinomialNB()), ])
text_mnb = text_mnb.fit(X_train.values,y_train.values)


# In[27]:


predicted= text_mnb.predict(X_train.values)
np.mean(predicted == y_train.values)


# In[28]:


from sklearn import metrics
metrics.confusion_matrix(y_train.values, predicted)


# In[ ]:





# In[ ]:





# In[ ]:




