#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Author : Amir Shokri
# github link : https://github.com/amirshnll/iris
# dataset link : http://archive.ics.uci.edu/ml/datasets/Iris
# email : amirsh.nll@gmail.com


# In[1]:


import pandas as pd


# In[5]:


data = pd.read_csv('iris.csv', header=None)
data


# In[4]:


data.describe()


# In[6]:


x = data[data.columns[:4]]
Y = data[data.columns[4]]


# In[7]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_x = scaler.fit_transform(x)


# In[9]:


Y.value_counts().plot.pie()


# In[11]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_x, Y, test_size=0.3, random_state=0)


# In[12]:


from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score


# In[14]:


from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='micro'))


# In[15]:


from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='micro'))


# In[16]:


from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='micro'))


# In[17]:


from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(100,))
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='micro'))


# In[19]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
print(f1_score(y_test, y_pred, average='micro'))


# In[ ]:




