#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[8]:


import pandas as pd
from sklearn.metrics import confusion_matrix


# In[75]:


path = "/home/pedro/repos/ml_web_api/ml-app-model/data/gridCV/"


# In[76]:


y_hat = pd.read_csv(path+"prediction_results.csv")
y_hat.shape


# In[77]:


y_test = pd.read_csv(path+"train.csv")
y_test = y_test.reset_index(drop=False)
y_test =y_test[["index","Survived"]].copy()
y_test.shape


# In[78]:


res = y_test.merge(y_hat, left_on="index", right_on="ID", how="inner")


# In[79]:


res = res[["ID","Survived", "y_hat"]]


# In[80]:


res.head(2)


# In[81]:





# In[82]:


confusion_matrix(y_true=res.Survived, y_pred=res.y_hat)


# In[83]:


res_table = res.groupby(["Survived", "y_hat"]).ID.count().reset_index(drop=False)
res_table["perc"] = np.around(res_table.ID / res_table.ID.sum() * 100,1)
res_table


# In[ ]:





# In[84]:


#Survived	y_hat	ID	perc
#0	0	0	538	60.4
#1	0	1	11	1.2
#2	1	0	156	17.5
#3	1	1	186	20.9


# In[48]:


#Survived	y_hat	ID	perc
#0	0	0	490	55.0
#1	0	1	59	6.6
#2	1	0	193	21.7
#3	1	1	149	16.7


# In[ ]:





# In[63]:


data_path="/home/pedro/repos/ml_web_api/How-to-score-0.8134-in-Titanic-Kaggle-Challenge/data/"
train = pd.read_csv(data_path+'train.csv')


# In[64]:


train.isnull().sum()


# In[65]:


train.Sex.unique()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




