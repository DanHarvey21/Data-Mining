#!/usr/bin/env python
# coding: utf-8

# In[29]:


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv(r"C:\Users\danie\Downloads\archive (2)\GroceryStoreDataSet.csv", names = ['products'], sep = ',')
df.head()


# In[30]:


data = list(df["products"].apply(lambda x:x.split(",") ))
data

from mlxtend.preprocessing import TransactionEncoder
a = TransactionEncoder()
a_data = a.fit(data).transform(data)
df = pd.DataFrame(a_data,columns=a.columns_)
df = df.replace(False,0)
df 


# In[31]:


import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules

df = pd.read_csv(r"C:\Users\danie\Downloads\archive (2)\GroceryStoreDataSet.csv", names = ['products'], sep = ',')
df.head()

data = list(df["products"].apply(lambda x:x.split(",") ))
data


from mlxtend.preprocessing import TransactionEncoder
a = TransactionEncoder()
a_data = a.fit(data).transform(data)
df = pd.DataFrame(a_data,columns=a.columns_)
df = df.replace(False,0)
df 

df = apriori(df, min_support = 0.2, use_colnames = True, verbose = 1)
df

df_ar = association_rules(df, metric = "confidence", min_threshold = 0.6)
df_ar 


# In[ ]:




