#!/usr/bin/env python
# coding: utf-8

# # Predicting Students Score Using Supervised ML LinearRegression Algorithm

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv(r'C:\Users\Parth Gohil\OneDrive\Desktop\TSF\data\students_score.csv')


# In[3]:


#Here we are printing the first 5 entries of data
data.head(5)


# # Plotting The Data
# 

# Here we can see the values of data are continous

# In[4]:


data.plot(x='Hours',y='Scores',style='d',figsize=(10,5))


# # Preparing the Model for Prediction

# We are using the LinearRegression Algorithm because the values of the Data are continous and LinearRegression Algorithm is used to Predict for continous vlaues

# In[5]:


X = data[['Hours']]
y = data[['Scores']]


# In[6]:


X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.5, random_state = 0)


# In[7]:


model = LinearRegression()


# In[8]:


model.fit(X_train1,y_train1)


# In[9]:


y_pred1= model.predict(X_test1)


# In[10]:


print(X_test1)
print(y_pred1)


# In[11]:


plt.plot(X_test1,y_pred1)
plt.xlabel('Hours')
plt.ylabel('Score')


# # Predicting By Splitting Into Different Train Test Values

# In[12]:


X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[13]:


model = LinearRegression()


# In[14]:


model.fit(X_train2,y_train2)


# In[15]:


y_pred2 = model.predict(X_test2)


# In[16]:


print(X_test2)
print(y_pred2)


# In[17]:


plt.plot(X_test2,y_pred2)
plt.xlabel('Hours')
plt.ylabel('Score')


# In[23]:


hours = [[9.25]]  
own_pred = model.predict(hours)  
print("Number of hours Studied = {}".format(hours))  
print("Predicted Score by hours studied = {}".format(own_pred[0]))  


# # Testing Accuracy of Models

# In[19]:


print("Accuracy of Model 1")
print(model.score(X_test1, y_test1))


# In[20]:


print("Accuracy of Model 2")
print(model.score(X_test2, y_test2))


# In[21]:


fig, (ax1,ax2) = plt.subplots(2)
labels = ['Model1','Model2']


# # Comparing Both Models Result

# In[22]:


ax1 = plt.plot(X_test1,y_pred1)
ax2 = plt.plot(X_test2,y_pred2)
plt.legend(labels= labels)


# In[ ]:





# In[ ]:




