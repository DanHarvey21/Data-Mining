#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd

data = pd.read_csv(r'C:\Users\danie\Downloads\Lab2_data_Verterbrate.csv',header='infer') 
data


# In[4]:


import pandas as pd

data = pd.read_csv(r'C:\Users\danie\Downloads\Lab2_data_Verterbrate.csv',header='infer') 

data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')
data


# In[9]:


import pandas as pd

data = pd.read_csv(r'C:\Users\danie\Downloads\Lab2_data_Verterbrate.csv',header='infer') 
data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')

pd.crosstab([data['Warm-blooded'],data['Gives Birth']],data['Class']) 


# In[15]:


import pandas as pd
import pydotplus
from sklearn import tree 
from IPython.display import Image

data = pd.read_csv(r'C:\Users\danie\Downloads\Lab2_data_Verterbrate.csv',header='infer') 
data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')

pd.crosstab([data['Warm-blooded'],data['Gives Birth']],data['Class']) 

Y = data['Class']
X = data.drop(['Name','Class'],axis=1)
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3)
clf = clf.fit(X, Y) 

dot_data = tree.export_graphviz(clf, feature_names=X.columns, class_names=['mammals','non-mammals'], filled=True, 
                                out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
Image(graph.create_png())


# In[19]:


import pandas as pd
import pydotplus
from sklearn import tree
from sklearn.metrics import accuracy_score
from IPython.display import Image

data = pd.read_csv(r'C:\Users\danie\Downloads\Lab2_data_Verterbrate.csv',header='infer') 
data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')

pd.crosstab([data['Warm-blooded'],data['Gives Birth']],data['Class']) 

Y = data['Class']
X = data.drop(['Name','Class'],axis=1)
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3)
clf = clf.fit(X, Y) 

dot_data = tree.export_graphviz(clf, feature_names=X.columns, class_names=['mammals','non-mammals'], filled=True, 
                                out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
Image(graph.create_png())

testData = [['gila monster',0,0,0,0,1,1,'non-mammals'],
 ['platypus',1,0,0,0,1,1,'mammals'],
 ['owl',1,0,0,1,1,0,'non-mammals'],
 ['dolphin',1,1,1,0,0,0,'mammals']]
testData = pd.DataFrame(testData, columns=data.columns)
testData 

testY = testData['Class']
testX = testData.drop(['Name','Class'],axis=1)
predY = clf.predict(testX)
predictions = pd.concat([testData['Name'],pd.Series(predY,name='Predicted Class')], axis=1)
predictions 


# In[20]:


import pandas as pd
import pydotplus
from sklearn import tree
from sklearn.metrics import accuracy_score
from IPython.display import Image

data = pd.read_csv(r'C:\Users\danie\Downloads\Lab2_data_Verterbrate.csv',header='infer') 
data['Class'] = data['Class'].replace(['fishes','birds','amphibians','reptiles'],'non-mammals')

pd.crosstab([data['Warm-blooded'],data['Gives Birth']],data['Class']) 

Y = data['Class']
X = data.drop(['Name','Class'],axis=1)
clf = tree.DecisionTreeClassifier(criterion='entropy',max_depth=3)
clf = clf.fit(X, Y) 

dot_data = tree.export_graphviz(clf, feature_names=X.columns, class_names=['mammals','non-mammals'], filled=True, 
                                out_file=None) 
graph = pydotplus.graph_from_dot_data(dot_data) 
Image(graph.create_png())

testData = [['gila monster',0,0,0,0,1,1,'non-mammals'],
 ['platypus',1,0,0,0,1,1,'mammals'],
 ['owl',1,0,0,1,1,0,'non-mammals'],
 ['dolphin',1,1,1,0,0,0,'mammals']]
testData = pd.DataFrame(testData, columns=data.columns)
testData 

testY = testData['Class']
testX = testData.drop(['Name','Class'],axis=1)
predY = clf.predict(testX)
predictions = pd.concat([testData['Name'],pd.Series(predY,name='Predicted Class')], axis=1)
predictions 

print('Accuracy on test data is %.2f' % (accuracy_score(testY, predY)))


# In[ ]:




