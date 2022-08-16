#!/usr/bin/env python
# coding: utf-8

# In[33]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics

df=pd.read_csv(r'C:\Users\danie\Downloads\final_data.csv', header='infer', )
df_first_30 = df.head(30)

y = df_first_30.trending_date
X = df_first_30.drop('trending_date', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=200, stratify=y)
print(X_train, X_test, y_train, y_test)

gnb = GaussianNB()

#Train the model using the training sets
gnb.fit_transform(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
print(y_pred)


# In[ ]:





# In[ ]:




