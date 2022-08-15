#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

df=pd.read_csv(r'C:\Users\danie\Downloads\final_data.csv', header='infer', )

unique_videos = df[['title','video_id']].drop_duplicates()


columns  = ['view_count','comment_count','comments_disabled','ratings_disabled']

X = df.loc[unique_videos.index,columns]

view_count = df.loc[X.index,'view_count']
oh = OneHotEncoder(sparse=False)
view_count = pd.DataFrame(oh.fit_transform(view_count.values.reshape(-1,1)),index=view_count.index,columns=oh.get_feature_names_out ())
#Calculates the mean(μ) and standard deviation(σ) of the feature F at a time it will transform the data points of the feature F.

X = pd.concat([X,view_count],axis=1)

y = df.loc[X.index,['view_count','comment_count']]

# Processing the data between view count and comment count


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)
# Split dataset into training set and test set
#80% train 20% test

pca_kmeans = make_pipeline(StandardScaler(), PCA(n_components=2), KMeans(n_clusters=7, init='k-means++', random_state=0))

pca_kmeans.fit(X_train)
#fit the instantiated k_means to the X_Train
pred = pca_kmeans.predict(X_test)
#predict function provides predictions on which cluster the data in the test set will be associated to

X_test_pca = pca_kmeans[0].transform(X_test)
X_test_pca = pca_kmeans[1].transform(X_test_pca)

X_test_pca.shape
# Principal component analysis (PCA) helps to reduce the number of "features" while preserving the variance
#whereas clustering reduces the number of "data-points" by 
#summarizing several points by their expectations/means (in the case of k-means).
plt.scatter(x=X_test_pca[:,0],y=X_test_pca[:,1],c=pred)

enc = LabelEncoder()
y_test_enc = enc.fit_transform(y_test['view_count'])

plt.scatter(x=X_test_pca[:,0],y=X_test_pca[:,1],c=y_test_enc)


# In[48]:


from sklearn.ensemble import RandomForestClassifier


rf = RandomForestClassifier(n_estimators=500,max_depth=25,max_features='log2',verbose=1,criterion='gini')

rf.fit(X_train,y_train)

rf = RandomForestClassifier(n_estimators=1000, max_depth=None, criterion='entropy',)
rf.fit(X_train,y_train)


# In[ ]:




