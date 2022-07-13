#!/usr/bin/env python
# coding: utf-8

# In[72]:


import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import numpy as np


# In[3]:


path_to_file = "G:\Other computers\My Laptop\Gsolar_Rimsha\Rimsha_AdditionalPractice\Projects\EDA & Linear Regression\student_scores.csv"
data = pd.read_csv(path_to_file)


# In[5]:


data.describe().round(decimals=0)


# In[6]:


data.shape


# In[7]:


data.head


# In[22]:


data.plot.scatter(x='Hours', y='Scores', title = 'Scatterplot of hours and scores percentages', color = 'red')


# In[27]:


#Checking if there's any missing values
print(data.isnull().sum())
msno.bar(data)


# In[28]:


data.corr() 
#The correlation between scores & hours is 97% which means that there is a strong correlation between them.


# In[31]:


x = data['Hours'].values.reshape(-1,1)
y = data['Scores'].values.reshape(-1,1)


# In[50]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[51]:


from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)


# In[52]:


regressor.coef_


# In[53]:


regressor.intercept_


# In[56]:


y_pred = regressor.predict(x_test)


# In[61]:


plt.scatter(x_test, y_test, color='red')
plt.plot(x_train, regressor.predict(x_train), color= 'blue')
plt.title("Scatter plot of Hours & Student scores")
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()


# In[66]:


print(regressor.predict([[5]]))


# In[71]:


df_preds = pd.DataFrame({'Actual': y_test.squeeze(), 'Predicted': y_pred.squeeze() })
print(df_preds)


# In[73]:


from sklearn.metrics import mean_absolute_error, mean_squared_error


# In[74]:


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)


# In[82]:


print(f'Mean absolute error: {mae:.2f}')
print(f'Mean squared error: {mse:.2f}')
print(f'Root mean squared error: {rmse:.2f}')


# In[ ]:




