#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[16]:


import numpy as np
import pandas as pd
import dill as pickle
from sklearn.metrics import confusion_matrix


# In[43]:


#path = "/home/pedro/repos/ml_web_api/ml-app-model/data/gridCV/"
#with open(path+"gridCV.pk", 'rb') as model_file:
#    model = pickle.load(model_file)
#data = pd.read_csv(path+"train.csv", encoding='utf-8', sep=",")
#prediction = model.predict(data)
#prediction # array([0, 1, 1, 1, 0, 0, 0, 0, 1, 


# In[ ]:





# In[ ]:





# In[33]:


path = "/home/pedro/repos/ml_web_api/ml-app-model/data/gridCV/"


# In[34]:


y_hat = pd.read_csv(path+"prediction_results.csv")
y_hat.shape


# In[35]:


y_test = pd.read_csv(path+"train.csv")
y_test = y_test.reset_index(drop=False)
y_test =y_test[["index","Survived"]].copy()
y_test.shape


# In[36]:


res = y_test.merge(y_hat, left_on="index", right_on="ID", how="inner")


# In[37]:


res = res[["ID","Survived", "y_hat"]]


# In[39]:


res.head(2)


# In[ ]:





# In[40]:


confusion_matrix(y_true=res.Survived, y_pred=res.y_hat)


# In[41]:


res_table = res.groupby(["Survived", "y_hat"]).ID.count().reset_index(drop=False)
res_table["perc"] = np.around(res_table.ID / res_table.ID.sum() * 100,1)
res_table


# In[ ]:


#Survived	y_hat	ID	perc
#0	0	0	499	56.0
#1	0	1	50	5.6
#2	1	0	94	10.5
#3	1	1	248	27.8


# In[26]:


#Survived	y_hat	ID	perc
#0	0	0	538	60.4
#1	0	1	11	1.2
#2	1	0	156	17.5
#3	1	1	186	20.9


# In[27]:


#Survived	y_hat	ID	perc
#0	0	0	490	55.0
#1	0	1	59	6.6
#2	1	0	193	21.7
#3	1	1	149	16.7


# In[ ]:





# In[28]:


data_path="/home/pedro/repos/ml_web_api/How-to-score-0.8134-in-Titanic-Kaggle-Challenge/data/"
train = pd.read_csv(data_path+'train.csv')


# In[30]:


#pd.get_dummies(train['Pclass'], prefix="Pclass")


# In[31]:


#train.isnull().sum()


# In[32]:


#train.Sex.unique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




