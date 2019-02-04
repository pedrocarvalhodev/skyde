#!/usr/bin/env python
# coding: utf-8

# In[23]:


import io
import base64


# In[11]:


import datetime
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from pmdarima.arima import auto_arima


# In[12]:


from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style("dark")


# In[13]:


#from plotly.offline import download_plotlyjs, init_notebook_mode, plot,iplot, plot_mpl
#import plotly.plotly as ply
#import cufflinks as cf
#cf.go_offline()


# https://iextrading.com/trading/eligible-symbols/

# In[14]:


SYMBOLS = {"AAPL":"APPLE INC",
           "PBR" :"PETROLEO BRASILEIRO-SPON ADR",
           "VIV" :"TELEFONICA BRASIL-ADR PREF",
           "BZF" :"WISDOMTREE BRAZILIAN REAL ST",
           "C"   :"CITIGROUP INC",
           "FB":"FACEBOOK INC-CLASS A",
           "AMZN":"AMAZON.COM INC"}


# In[ ]:





# In[15]:


name = 'AMZN'
end = datetime.datetime.today()
start = end-datetime.timedelta(days=365)
data = web.DataReader(name,'iex',start=start,end=end)
data.tail(2)


# In[16]:


df=data.copy()
df.rename(index=str, columns={"close": "Close"}, inplace=True)
df = df["Close"].to_frame()
idx = pd.date_range(df.index.min(), df.index.max(), freq="D")
df.index = pd.DatetimeIndex(df.index)
df = df.reindex(idx, fill_value=np.nan)
df = df.interpolate(method='linear')
df.dropna(inplace=True)


# In[17]:


stepwise_model = auto_arima(y=df, start_p=1, start_q=1,
                           max_p=7, max_q=7, max_d=7, max_order=14,
                           m=7, start_P=0, seasonal=True,
                           d=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)

stepwise_model.fit(df)


# In[18]:


n_periods=30
future_forecast = stepwise_model.predict(n_periods=n_periods)
future_forecast = pd.DataFrame(future_forecast,
                               index = pd.date_range(start=df.index[-1], end=None, periods=n_periods, freq="D"),
                               columns=['Prediction'])


# In[19]:


ds = pd.concat([df.tail(n_periods*4),future_forecast],axis=1)


# In[24]:


ds.plot()
plt.xlabel('Relative Importance')
plt.ylabel('Top Features \n Descending order')
plt.title(f"Dataset features by importance \n Target: SYMBOL \n ML method: MMM")
# 5. Save and render
img = io.BytesIO()
plt.savefig(img, format='png')
img.seek(0)


# In[30]:


#ds.plot()
plt.figure(figsize=(10,6))
plt.plot(list(ds.index), list(ds['Close'].values))
plt.plot(list(ds.index), list(ds['Prediction'].values))
plt.xlabel('Relative Importance')
plt.ylabel('Top Features \n Descending order')
plt.title(f"Dataset features by importance \n Target: SYMBOL \n ML method: MMM")


# In[29]:


ds.head()


# In[ ]:




