#!/usr/bin/env python
# coding: utf-8

# In[2]:


cd downloads


# In[3]:


pwd


# In[4]:


import numpy as np
import pandas as pd
from pathlib import Path
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Return Forcasting: Read Historical Daily Yen Futures Data

# In[5]:


# Futures contract on the Yen-dollar exchange rate:
# This is the continuous chain of the futures contracts that are 1 month to expiration
## Had to double check to make sure I didn't reopen the Regression Starter Code!
yen_futures = pd.read_csv(("yen.csv"), index_col="Date", infer_datetime_format=True, parse_dates=True
)
yen_futures.head()


# In[6]:


# Trim the dataset to begin on January 1st, 1990
yen_futures = yen_futures.loc["1990-01-01":, :]
yen_futures.head()


# ### Return Forecasting: Initial Time-Series Plotting
# 
# Start by plotting the "Settle" price. Do you see any patterns, long-term and/or short?
# 

# In[7]:


# Plot just the "Settle" column from the dataframe:
## I think I like 20,15 the best
yen_futures.Settle.plot(figsize=[20,15],title='Yen Futures Settle Prices',legend=True)


# ### Decomposition Using a Hodrick-Prescott Filter
# 
# Using a Hodrick-Prescott Filter, decompose the Settle price into a trend and noise.
# 

# In[8]:


import statsmodels.api as sm

# Apply the Hodrick-Prescott Filter by decomposing the "Settle" price into two separate series:

ts_noise, ts_trend = sm.tsa.filters.hpfilter(yen_futures['Settle'])

ts_trend.rename({'Settle':'trend'},inplace=True)
ts_trend.head


# In[9]:


ts_noise.rename({'Settle':'noise'},inplace=True)
ts_noise.head()


# In[10]:


# Create a dataframe of just the settle price, and add columns for "noise" and "trend" series from above:


df_yen_futures = yen_futures[['Settle']].copy()
df_yen_futures['trend'] = ts_trend
df_yen_futures['noise'] = ts_noise
df_yen_futures.tail()


# In[11]:


# Plot the Settle Price vs. the Trend for 2015 to the present


df_yen_futures.plot(y=['Settle', 'trend'],figsize=[20,15],title='Settle vs Trend',legend=True)


# In[12]:


# Plot the Settle Noise

df_yen_futures.plot(y='noise',figsize=[20,15],title='Noise',legend=True)


# 
# ### Forecasting Returns using an ARMA Model
# 
# Using futures Settle Returns, estimate an ARMA model
# 
#     ARMA: Create an ARMA model and fit it to the returns data. Note: Set the AR and MA ("p" and "q") parameters to p=2 and q=1: order=(2, 1).
#     Output the ARMA summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
#     Plot the 5-day forecast of the forecasted returns (the results forecast from ARMA model)
# 
# 

# In[13]:


# Create a series using "Settle" price percentage returns, drop any nan"s, and check the results:
# (Make sure to multiply the pct_change() results by 100)
# In this case, you may have to replace inf, -inf values with np.nan"s
returns = (yen_futures[["Settle"]].pct_change() * 100)
returns = returns.replace(-np.inf, np.nan).dropna()
returns.tail()


# In[14]:


import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
# Estimate and ARMA model using statsmodels (use order=(2, 1))

model = ARMA(returns.values, order=(2,1))

# Fit the model and assign it to a variable called results

results = model.fit()


# In[15]:


# Output model summary results:
results.summary()


# In[16]:


# Plot the 5 Day Returns Forecast

pd.DataFrame(results.forecast(steps=5)[0]).plot(title="5 Day Returns Forecast")


# ### Forecasting the Settle Price using an ARIMA Model
# 
#        Using the raw Yen Settle Price, estimate an ARIMA model.
#         Set P=5, D=1, and Q=1 in the model (e.g., ARIMA(df, order=(5,1,1))
#         P= # of Auto-Regressive Lags, D= # of Differences (this is usually =1), Q= # of Moving Average Lags
#     Output the ARIMA summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
#     Construct a 5 day forecast for the Settle Price. What does the model forecast will happen to the Japanese Yen in the near term?
# 
# 

# In[17]:


from statsmodels.tsa.arima_model import ARIMA

# Estimate and ARIMA Model:
# Hint: ARIMA(df, order=(p, d, q))

model = ARIMA(yen_futures['Settle'], order=(5, 1, 1))


# Fit the model
results = model.fit()


# In[18]:


# Output model summary results:
results.summary()


# In[19]:


# Plot the 5 Day Price Forecast
# YOUR CODE HERE!

pd.DataFrame(results.forecast(steps=5)[0]).plot(title="5 Day Returns Forecast")


# 
# ### Volatility Forecasting with GARCH
# 
# Rather than predicting returns, let's forecast near-term volatility of Japanese Yen futures returns. Being able to accurately predict volatility will be extremely useful if we want to trade in derivatives or quantify our maximum loss.
# 
# Using futures Settle Returns, estimate an GARCH model
# 
#     GARCH: Create an GARCH model and fit it to the returns data. Note: Set the parameters to p=2 and q=1: order=(2, 1).
#     Output the GARCH summary table and take note of the p-values of the lags. Based on the p-values, is the model a good fit (p < 0.05)?
#     Plot the 5-day forecast of the volatility.
# 
# 

# In[ ]:


from arch import arch_model


# In[ ]:


# Estimate a GARCH model:

model = arch_model(returns['Settle'], mean="Zero", vol="GARCH", p=2,q=1)

# Fit the model

res = model.fit(disp="off")


# In[ ]:


# Summarize the model results

res.summary()


# In[ ]:


# Find the last day of the dataset
last_day = returns.index.max().strftime('%Y-%m-%d')
last_day


# In[ ]:


# Create a 5 day forecast of volatility
forecast_horizon = 5
# Start the forecast using the last_day calculated above
# YOUR CODE HERE!
forecasts = res.forecast(start=last_day, horizon=forecast_horizon)
forecasts


# In[ ]:


# Annualize the forecast
intermediate = np.sqrt(forecasts.variance.dropna() * 252)
intermediate.head()


# In[ ]:


# Transpose the forecast so that it is easier to plot
final = intermediate.dropna().T
final.head()


# In[ ]:


# Plot the final forecast
# YOUR CODE HERE!

final.plot(title='5 Day Forecast of Volatility', legend=True)


# ### Conclusions

# Based on your time series analysis, would you buy the yen now?
# 
# The volatility and overall risk is increasing so I would't buy into yen
# 
# Is the risk of the yen expected to increase or decrease?
# 
# It seems to be expected to increase
# 
# Based on the model evaluation, would you feel confident in using these models for trading?
# 
# Yes, the models seem to work well as long as you have everything updated and running well.
# 
# 

# In[ ]:




