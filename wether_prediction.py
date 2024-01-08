#!/usr/bin/env python
# coding: utf-8

# In[373]:


# Reading in the initial data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

weather = pd.read_csv("wether_somewhere_in-USA", index_col="DATE")

print(weather.info())
print(weather["TMAX"].head(5))
print(weather["TMIN"].head(5))
print(weather["PRCP"].value_counts(dropna=True))
print(weather["SNOW"].value_counts(dropna=True))
print(weather["SNWD"].value_counts(dropna=True))


# # 1. Prepare Data
# 
# At first we used the dataframe.info() function, to list some data.
# It immediatly showed us that there is quiet a lot of unnecessary columns, like column number 7, that only contains 2 non-null entitys.
# For the sake of this project we only need the following 5:
# - prcp - total precipitation
# - tmax - maximum temperature in Fahrenheit (i think)
# - tmin - minimum temperatire in Fahrenheit (i think)
# - snow - snowfall (mostly empty)
# - snwd - total snow depth on the ground (mostly empty)
# 
# Since °C is the metric Unit for Temperature, it should also be converted from Fahrenheit to Celsius
# 

# In[374]:


# impo_we stands for weather data, and is so that one can type 
# it more quickly. It only contains the 5 columns that are used
we_da = weather[["PRCP","TMAX","TMIN","SNOW","SNWD"]]
we_da = we_da.rename(columns ={"PRCP":"prcp","TMAX":"tmax","TMIN":"tmin","SNOW":"snow","SNWD":"snowed"})


# since snow and snowed contains only 0.0 we drop them too
we_da = we_da[["prcp","tmax","tmin"]]

# convert tmin and tmax in to Celsius
we_da["tmax"] = (we_da["tmax"]-32)*(5/9)
we_da["tmin"] = (we_da["tmin"]-32)*(5/9)
print(we_da["tmax"])


# In[375]:


#check for errors
print(we_da.describe())
# so far no 9999 since it should be the max value in this case.
# Now we clear the nan values. we do it by inerpolating with the rest of the values
we_da = we_da.interpolate()
print(we_da.describe())

# This didnt change the max nor the min values but filled the empty ones.

# Now convert the date into a datetime datatype
we_da.index = pd.to_datetime(we_da.index) 


# In[376]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[377]:


we_da[["tmin","tmax"]].plot()
plt.show()
# Some Years of Data are missing


# # 2. Train the Model
# 
# To Train the Model we need another column in our data that is the prediction target. Since we re trying to predict the temperature, it should be the temperature of the next day. We have information about the max and min temperature, so we should have two predictions.   
# 
# At the end of this chapter there is also the implementation of a function that takes segements of weather data and will make predictions out of it.

# In[378]:


# make prediction columns for minimum and maximum temperature per day.
we_da["target_min"] = we_da["tmin"].shift(-1)
we_da["target_max"] = we_da["tmax"].shift(-1)

#print first rows to check wether the data is correct
print(we_da[["tmin", "target_min","tmax","target_max"]].head(6))

#print last few rows to check for nan values due to the -1 shift
print(we_da[["tmin", "target_min","tmax","target_max"]].iloc[-6:,:])

# since the last row hat a nan value, it should be dropped. Also print last rows to check wether it worked
we_da = we_da.iloc[:-1,:]
print(we_da[["tmin", "target_min","tmax","target_max"]].iloc[-6:,:])


# In[379]:


from sklearn.linear_model import Ridge
reg_tmax = Ridge(alpha=0.1)
reg_tmin = Ridge(alpha=0.1)


# There is 16858 data points. 90% of it should be used as Training Data. 10% of it as test Data.
three_third = int(16858*0.9)
train = we_da.iloc[:three_third,:]
test = we_da.iloc[three_third+1:,:]
print(test.head(5))


# In[380]:


# Train both Regression algorithms
reg_tmax.fit(train[["prcp","tmax","tmin"]],train["target_max"])
reg_tmin.fit(train[["prcp","tmax","tmin"]],train["target_min"])


# In[381]:


# Now use the Test Data to make predictions from the Test Data
predict_tmax = reg_tmax.predict(test[["prcp","tmax","tmin"]])
predict_tmin = reg_tmin.predict(test[["prcp","tmax","tmin"]])


# In[382]:


# A Function is needed that is able to evaluate our prediction. For this it just substracts
# the prediction from the actual data, makes it an absolute value, and returns the mean.

def mean_absolute_error(target, predictions):
    error = abs(target-predictions)
    mean_error = error.mean()
    return mean_error

print("tmax Error: ", mean_absolute_error(test["target_max"],predict_tmax))
print("tmin Error: ", mean_absolute_error(test["target_min"],predict_tmin))


# In[383]:


combined_data_max = pd.concat([test["tmax"], pd.Series(predict_tmax, index = test.index, name = "prediction_tmax")] ,axis=1)
combined_data_min = pd.concat([test["tmin"], pd.Series(predict_tmin, index = test.index, name = "prediction_tmin")] , axis=1)
combined_data = pd.concat([combined_data_max, combined_data_min], axis=1)
print(combined_data)


# In[384]:


print(len(we_da))


# In[385]:


# This Function takes the weather data, the predictors as a list e.g. ["tmax","tmin"], the regression model, and 
# the split percentage. E.g. now it uses 90% as training data, and 10% as testing data
def create_predictions(weather_data, predictors_in_weather, reg_model, split=0.9): 
    splitter = int(len(weather_data)*split)
    train = weather_data.iloc[:splitter,:]
    test = weather_data.iloc[splitter+1:,:]
    try:
        reg_model.fit(train[predictors_in_weather],train["target"])
    except:
        print("Is there a column named 'target' in your training data? Are your Predictors named in the Dataframe columns")
    predictions = reg_model.predict(test[predictors_in_weather])
    error = mean_absolute_error(test["target"], predictions)
    combined = pd.concat([test["target"], pd.Series(predictions,index=test.index)], axis = 1)
    combined.columns = ["actual_Celsius","predictions_Celsius"]
    return combined, error

# Use the function to test it
test_weather = we_da[["prcp","tmax","tmin"]].copy()
test_weather["target"] = we_da["target_max"].copy()
test_predictors = ["prcp","tmax","tmin"]
test_reg = Ridge(alpha=0.1)
test_combined, test_error = create_predictions(test_weather, test_predictors, test_reg)

print("the test function Error is: ", test_error)
print(test_combined.head(5))
print(test_combined.loc["2018-01-01":"2018-01-10","actual_Celsius"])


# # 3 Make better predictions
# 
# To make better predictions we will make another column with the average temperature the last 30 days. Tmax aswell as Tmin should be used. The ridge regression algorithm can use that data to make better predictions.

# In[386]:


# Add the additional columns
we_da["30_day_max_avg"] = we_da["tmax"].rolling(30).mean()
#we_da["30_day_min_avg"] = we_da["tmin"].rolling(30).mean()
we_da["month_day_max"] = we_da["30_day_max_avg"] / we_da["tmax"]
we_da["max_min"] = we_da["tmax"] / we_da["tmin"]
#we_da["month_day_min"] = we_da["30_day_min_avg"] / we_da["tmin"]


# In[387]:


# Since the new columns use the average on the last 30 days, the first 30 rows contain nan values, since there is no avg value.
# This is why they should be dropped
we_da_temp = we_da.iloc[33:,:]
we_da_plus = we_da_temp[["prcp","tmax","tmin","month_day_max","max_min"]].copy()
we_da_plus["target"] = we_da_temp["target_max"].copy()


# make the new predictors
predictors_plus = ["prcp","tmax","tmin","month_day_max","max_min"]
#make the new ridge algorithm
reg_plus = Ridge(alpha=0.1)

print(we_da_plus.isin([np.inf, -np.inf]).sum())
print(we_da_plus.isnull().sum())

# Some values of tmin contain "0". Since we divided by tmin for max_min, we get some values that are infinite.
# We need to drop those values, otherwise we will get a error message.
we_da_plus.loc[we_da_plus[(we_da_plus["max_min"]==np.inf)].index,"max_min"] = 0
print("values without inf.")
print(we_da_plus.isin([np.inf, -np.inf]).sum())


# In[388]:


# Test the new Dataset with the function
combined_plus, error_plus = create_predictions(we_da_plus.dropna(), predictors_plus, reg_plus)

print(error_plus)
combined_plus.plot()


# In[389]:


# to make it more precise, more predictors can be added. For this the monthly average will be added.
we_da_plus["monthly_avg"] = we_da_plus["tmax"].groupby(we_da_plus.index.month).apply(lambda x: x.expanding(1).mean())
we_da_plus["day_of_year_avg"] = we_da_plus["tmax"].groupby(we_da_plus.index.dayofyear).apply(lambda x : x.expanding(1).mean())

# update the list of predictors
predictors_plus.append("monthly_avg")
predictors_plus.append("day_of_year_avg")

# Test the dataframe with the extended predicotrs
combined_plus, error_plus = create_predictions(we_da_plus.dropna(), predictors_plus, reg_plus)

print(error_plus)


# In[391]:


# Correlations
we_da_plus.corr()["target"]
# The closer the number to 1 the greater the influence to the value of "target". 


# In[ ]:




