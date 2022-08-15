#!/usr/bin/env python
# coding: utf-8

# In[46]:


import numpy as np
#Import scikit-learn dataset library
from sklearn import datasets
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB 
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics 

#Load dataset
wine = datasets.load_wine()
print("Features: ", wine.feature_names)
print("Labels: ", wine.target_names)
print(wine.data.shape)

np.set_printoptions(suppress=True)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109) 
# 70% training and 30% test
print(X_train, X_test, y_train, y_test)

#Create a Gaussian Classifier
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
print(y_pred)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))


# In[39]:


import numpy as np
# Import train_test_split function
from sklearn.model_selection import train_test_split

np.set_printoptions(suppress=True)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3,random_state=109, stratify=wine.target) 
# 70% training and 30% test

print(X_train, X_test, y_train, y_test)




# In[40]:


from sklearn.naive_bayes import GaussianNB 
#Import scikit-learn metrics module for accuracy calculation

#Create a Gaussian Classifier
gnb = GaussianNB() 

#Train the model using the training sets
gnb.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = gnb.predict(X_test)
print(y_pred)


# In[41]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred)) 


# In[ ]:




