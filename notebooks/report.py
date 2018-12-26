#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os 
import json
import numpy as np
import pandas as pd
import dill as pickle
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import make_pipeline

import warnings
warnings.filterwarnings("ignore")

path = "../data/titanic/"

#data_path="https://raw.githubusercontent.com/ahmedbesbes/How-to-score-0.8134-in-Titanic-Kaggle-Challenge/master/data/"
data_path="/home/pedro/repos/ml_web_api/How-to-score-0.8134-in-Titanic-Kaggle-Challenge/data/"


# In[2]:


train = pd.read_csv(data_path+'train.csv')
y = "Survived"
X = [x for x in train.columns if x != y]
X_train, X_test, y_train, y_test = train_test_split(train[X], train[y], test_size=0.25, random_state=42)
print("Shape: ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# In[53]:


X_train.nunique()


# In[62]:


X_train.isnull().sum()


# In[61]:


X_train['Age'] = X_train['Age'].fillna(X_train.groupby(['Sex','Pclass'])['Age'].transform('mean'))


# In[ ]:





# In[ ]:





# In[3]:


X_train.head(2).transpose()


# In[39]:


def status(feature):
    print('Processing', feature, ': ok')

Title_Dictionary = {
        "Capt": "Officer",
        "Col": "Officer",
        "Major": "Officer",
        "Jonkheer": "Royalty",
        "Don": "Royalty",
        "Sir" : "Royalty",
        "Dr": "Officer",
        "Rev": "Officer",
        "the Countess":"Royalty",
        "Mme": "Mrs",
        "Mlle": "Miss",
        "Ms": "Mrs",
        "Mr" : "Mr",
        "Mrs" : "Mrs",
        "Miss" : "Miss",
        "Master" : "Master",
        "Lady" : "Royalty"}


def get_titles(df):
    # we extract the title from each name
    df['Title'] = df['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
    # a map of more aggregated title
    # we map each title
    df['Title'] = df.Title.map(Title_Dictionary)
    status('Title')
    return df
    
def fill_age(row):
    condition = (
        (grouped_median_train['Sex'] == row['Sex']) & 
        (grouped_median_train['Title'] == row['Title']) & 
        (grouped_median_train['Pclass'] == row['Pclass'])) 
    return grouped_median_train[condition]['Age'].values[0]

def process_age(df):
    #global df
    # a function that fills the missing values of the Age variable
    df['Age'] = df.apply(lambda row: fill_age(row) if np.isnan(row['Age']) else row['Age'], axis=1)
    status('age')
    return df

def process_names(df):
    #global df
    # we clean the Name variable
    df.drop('Name', axis=1, inplace=True)
    # encoding in dummy variable
    titles_dummies = pd.get_dummies(df['Title'], prefix='Title')
    df = pd.concat([df, titles_dummies], axis=1)
    # removing the title variable
    df.drop('Title', axis=1, inplace=True)
    status('names')
    return df

def process_fares(df):
    #global df
    # there's one missing fare value - replacing it with the mean.
    df.Fare.fillna(df.iloc[:891].Fare.mean(), inplace=True)
    status('fare')
    return df

def process_embarked(df):
    #global df
    # two missing embarked values - filling them with the most frequent one in the train  set(S)
    df.Embarked.fillna('S', inplace=True)
    # dummy encoding 
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked')
    df = pd.concat([df, embarked_dummies], axis=1)
    df.drop('Embarked', axis=1, inplace=True)
    status('embarked')
    return df

def process_cabin(df):
    #global df    
    # replacing missing cabins with U (for Uknown)
    df.Cabin.fillna('U', inplace=True)
    # mapping each Cabin value with the cabin letter
    df['Cabin'] = df['Cabin'].map(lambda c: c[0])
    # dummy encoding ...
    cabin_dummies = pd.get_dummies(df['Cabin'], prefix='Cabin')    
    df = pd.concat([df, cabin_dummies], axis=1)

    df.drop('Cabin', axis=1, inplace=True)
    status('cabin')
    return df

def process_sex(df):
    #global df
    # mapping string values to numerical one 
    df['Sex'] = df['Sex'].map({'male':1, 'female':0})
    status('Sex')
    return df

def process_pclass(df):
    #global df
    # encoding into 3 categories:
    pclass_dummies = pd.get_dummies(df['Pclass'], prefix="Pclass")
    # adding dummy variable
    df = pd.concat([df, pclass_dummies],axis=1)
    # removing "Pclass"
    df.drop('Pclass',axis=1,inplace=True)
    status('Pclass')
    return df

def process_ticket(df):
    #global df
    # a function that extracts each prefix of the ticket, returns 'XXX' if no prefix (i.e the ticket is a digit)
    def cleanTicket(ticket):
        ticket = ticket.replace('.','')
        ticket = ticket.replace('/','')
        ticket = ticket.split()
        ticket = map(lambda t : t.strip(), ticket)
        ticket = filter(lambda t : not t.isdigit(), ticket)
        if len(ticket) > 0:
            return ticket[0]
        else: 
            return 'XXX'

    # Extracting dummy variables from tickets:
    df['Ticket'] = df['Ticket'].map(cleanTicket)
    tickets_dummies = pd.get_dummies(df['Ticket'], prefix='Ticket')
    df = pd.concat([df, tickets_dummies], axis=1)
    df.drop('Ticket', inplace=True, axis=1)
    status('Ticket')
    return df

def process_family(df):
    #global df
    # introducing a new feature : the size of families (including the passenger)
    df['FamilySize'] = df['Parch'] + df['SibSp'] + 1

    # introducing other features based on the family size
    df['Singleton'] = df['FamilySize'].map(lambda s: 1 if s == 1 else 0)
    df['SmallFamily'] = df['FamilySize'].map(lambda s: 1 if 2 <= s <= 4 else 0)
    df['LargeFamily'] = df['FamilySize'].map(lambda s: 1 if 5 <= s else 0)

    status('family')
    return df


# In[40]:


df = X_train.copy()


# In[41]:


df.head(2)


# In[42]:


df = get_titles(df)
df.head(2)


# In[43]:


grouped_train = df.groupby(['Sex','Pclass','Title'])
grouped_median_train = grouped_train.median()
grouped_median_train = grouped_median_train.reset_index()[['Sex', 'Pclass', 'Title', 'Age']]


# In[44]:


df = process_names(df)
df.head(2)


# In[45]:


#df = process_age(df)
#df.head(2)


# In[46]:


df = process_fares(df)
df.head(2)


# In[47]:


df = process_embarked(df)
df.head(2)


# In[48]:


df = process_cabin(df)
df.head(2)


# In[49]:


df = process_sex(df)
df.head(2)


# In[50]:


df = process_pclass(df)
df.head(2)


# In[55]:


#df = process_ticket(df)
#df.head(2)


# In[52]:


df = process_family(df)
df.head(2)


# In[ ]:


#df = get_titles(df)
#df = process_names(df)
###df = process_age(df)
#df = process_fares(df)
#df = process_embarked(df)
#df = process_cabin(df)
#df = process_sex(df)
#df = process_pclass(df)
####df = process_ticket(df)
#df = process_family(df)
#print(df.info)
#return df.as_matrix()


# In[56]:


pd.DataFrame(df.info())


# In[58]:


df.select_dtypes(include=[np.number]).shape


# In[59]:


df.shape


# In[ ]:





# In[8]:


import pandas as pd


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


from sklearn.metrics import confusion_matrix


# In[82]:


confusion_matrix(y_true=res.Survived, y_pred=res.y_hat)


# In[83]:


res_table = res.groupby(["Survived", "y_hat"]).ID.count().reset_index(drop=False)
res_table["perc"] = np.around(res_table.ID / res_table.ID.sum() * 100,1)
res_table


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




