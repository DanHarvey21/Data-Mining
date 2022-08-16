#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd

df=pd.read_csv(r'C:\Users\danie\Downloads\final_data.csv',header='infer')

df


# In[69]:


import pandas as pd

df=pd.read_csv(r'C:\Users\danie\Downloads\final_data.csv',header='infer')

df.head(30)


# In[62]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

df=pd.read_csv(r'C:\Users\danie\Downloads\final_data.csv', header='infer', )
df.to_numpy()
df_first_30 = df.head(30)

y = df_first_30.trending_date
X = df_first_30.drop('trending_date', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)
print(X_train, X_test, y_train, y_test)


# In[67]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

df= pd.read_csv(r'C:\Users\danie\Downloads\final_data.csv', header='infer', )
df.to_numpy()
df_first_30 = df.head(30)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100, stratify=y)
print(X_train, X_test, y_train, y_test)


# In[ ]:





# In[ ]:




