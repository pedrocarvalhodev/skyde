#!/usr/bin/env python
# coding: utf-8

# In[90]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('/home/pedro/Downloads/train_timeseries.csv')#, nrows = 11856)


# In[4]:


print(df.shape)


# In[5]:


d = df[0:11856:]


# In[6]:


#Creating train and test set 
#Index 10392 marks the end of October 2013 
train=d[0:10392] 
test=d[10392:]


# In[7]:


print(train.shape, test.shape)


# In[8]:


train.tail(2)


# In[9]:


test.head(2)


# In[10]:


#Aggregating the dataset at daily level
df.Timestamp = pd.to_datetime(df.Datetime,format='%d-%m-%Y %H:%M') 
df.index = df.Timestamp 
df = df.resample('D').mean()


# In[11]:


train.Timestamp = pd.to_datetime(train.Datetime,format='%d-%m-%Y %H:%M') 
train.index = train.Timestamp 
train = train.resample('D').mean()


# In[12]:


test.Timestamp = pd.to_datetime(test.Datetime,format='%d-%m-%Y %H:%M') 
test.index = test.Timestamp 
test = test.resample('D').mean()


# In[13]:


y_hat_avg = test.copy()
fit1 = sm.tsa.statespace.SARIMAX(train.Count, order=(2, 1, 4),seasonal_order=(0,1,1,7)).fit()
y_hat_avg['SARIMA'] = fit1.predict(start="2013-11-1", end="2013-12-31", dynamic=True)


# In[14]:


plt.figure(figsize=(16,8))
plt.plot( train['Count'], label='Train')
plt.plot(test['Count'], label='Test')
plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
plt.legend(loc='best')


# In[ ]:





# # Auto Arima

# In[25]:


from pmdarima.arima import auto_arima
import plotly.plotly as ply
import cufflinks as cf
cf.go_offline()


# In[26]:


data = pd.read_csv('/home/pedro/Downloads/IPG2211A2N.csv',index_col=0)
data.head()


# In[27]:


data.index = pd.to_datetime(data.index)


# In[28]:


data.head(2)


# In[29]:


data.columns = ['Energy Production']


# In[30]:


data.iplot(title="Energy Production Jan 1985--Jan 2018")


# In[41]:


init_notebook_mode(connected=True)
#from plotly.plotly import plot_mpl
from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot, plot_mpl
from statsmodels.tsa.seasonal import seasonal_decompose
result = seasonal_decompose(data, model='multiplicative')
fig = result.plot()
plot_mpl(fig)


# In[33]:


data.shape


# In[42]:


data.head(2)


# In[43]:


data.tail(2)


# In[60]:


train = data.loc['2010-01-01':'2015-12-31']
test = data.loc['2016-01-01':]


# In[61]:


print(train.shape, test.shape)


# In[63]:


stepwise_model = auto_arima(train, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

print(stepwise_model.aic())


# In[ ]:


303.47216294278553


# In[47]:


#train = data.loc['1985-01-01':'2016-12-01']
#test = data.loc['2017-01-01':]


# In[64]:


stepwise_model.fit(train)


# In[65]:


future_forecast = stepwise_model.predict(n_periods=test.shape[0])
# This returns an array of predictions:
print(future_forecast)


# In[66]:


future_forecast = pd.DataFrame(future_forecast,
                               index = test.index,columns=['Prediction'])

pd.concat([test,future_forecast],axis=1).plot()


# In[67]:


pd.concat([data.loc['2010-01-01':],future_forecast],axis=1).iplot()


# In[ ]:





# ## WorldBank

# In[69]:


import pandas_datareader.data as web


# In[121]:


data = web.DataReader("DEXBZUS", 'fred', '2017-01-01', '2019-01-12')


# In[129]:





# In[130]:


df=data.copy()
idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
df.index = pd.DatetimeIndex(df.index)
df = df.reindex(idx, fill_value=np.nan)
df = df.interpolate(method='linear')
df.dropna(inplace=True)


# In[139]:





# In[140]:


df.tail()


# In[141]:


train = df.loc['2017-01-02':'2018-11-30']
test = df.loc['2018-12-01':]


# In[142]:


train.tail(2)


# In[143]:


test.head(2)


# In[144]:


print(train.shape, test.shape)


# In[156]:


stepwise_model = auto_arima(y=train, start_p=1, start_q=1,
                           max_p=7, max_q=7, max_d=7, max_order=14,
                           m=7, start_P=0, seasonal=True,
                           d=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

print(stepwise_model.aic())


# In[157]:


#auto_arima?


# In[158]:


stepwise_model.fit(train)


# In[159]:


future_forecast = stepwise_model.predict(n_periods=test.shape[0])
# This returns an array of predictions:
print(future_forecast)


# In[160]:


future_forecast = pd.DataFrame(future_forecast,
                               index = test.index,columns=['Prediction'])

pd.concat([test,future_forecast],axis=1).plot()


# In[162]:


pd.concat([df.loc['2018-10-01':],future_forecast],axis=1).iplot()


# In[ ]:





# In[ ]:





# In[ ]:


## GDP Brazil


# In[163]:


data = web.DataReader("NAEXKP01BRQ661S", 'fred')#, '2017-01-01', '2019-01-12')


# In[181]:


df=data.copy()
idx = pd.date_range(df.index.min(), df.index.max(), freq="Q")
print(len(idx), data.shape[0])


# In[182]:


df.index = pd.DatetimeIndex(df.index)
#df = df.reindex(idx, fill_value=np.nan)
#df = df.interpolate(method='linear')
df.dropna(inplace=True)
train = df.loc[:'2016-07-01']
test = df.loc['2017-01-01':]


# In[183]:


stepwise_model = auto_arima(y=train, start_p=1, start_q=1,
                           max_p=7, max_q=7, max_d=7, max_order=14,
                           m=4, start_P=0, seasonal=True,
                           d=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

stepwise_model.fit(train)
future_forecast = stepwise_model.predict(n_periods=test.shape[0])
future_forecast = pd.DataFrame(future_forecast,
                               index = test.index,columns=['Prediction'])

pd.concat([df,future_forecast],axis=1).iplot()


# In[177]:





# In[180]:





# In[ ]:





# In[ ]:





# In[184]:


#import pandas as pd
import datetime
#import pandas_datareader.data as web
name = 'MSFT'
end = datetime.datetime.today()
start = end-datetime.timedelta(days=365)
df = web.DataReader(name,'iex',start=start,end=end)


# In[185]:


df.tail()


# https://iextrading.com/trading/eligible-symbols/

# In[200]:


name = 'SPY'
end = datetime.datetime.today()
start = end-datetime.timedelta(days=365*3)
data = web.DataReader(name,'iex',start=start,end=end)
data.tail(2)


# In[201]:


print(data.shape, data.index.min(), data.index.max())


# In[207]:


df=data.copy()
df = df["close"].to_frame()
idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
df.index = pd.DatetimeIndex(df.index)
df = df.reindex(idx, fill_value=np.nan)
df = df.interpolate(method='linear')
df.dropna(inplace=True)


# In[208]:


train = df.loc['2018-01-01':'2019-01-02']
test = df.loc['2019-01-03':]
print(train.shape, test.shape)


# In[211]:


stepwise_model = auto_arima(y=train, start_p=1, start_q=1,
                           max_p=7, max_q=7, max_d=7, max_order=14,
                           m=7, start_P=0, seasonal=True,
                           d=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

stepwise_model.fit(train)
future_forecast = stepwise_model.predict(n_periods=test.shape[0])
future_forecast = pd.DataFrame(future_forecast,
                               index = test.index,columns=['Prediction'])


# In[213]:


pd.concat([df.loc['2018-12-01':],future_forecast],axis=1).iplot()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




