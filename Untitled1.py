#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data = pd.read_csv('dataset.csv')


# In[5]:


data.head()


# In[6]:


data.info()


# In[7]:


data.describe()


# In[63]:


data.isnull()


# In[62]:


data["Vehicle_Name"].value_counts()


# In[20]:


sns.pairplot(data)


# In[60]:


sns.distplot(data['Selling_Price'], bins =20)


# In[61]:


sns.heatmap(data.corr())


# In[15]:


from sklearn import preprocessing
encoder = preprocessing.LabelEncoder()


# In[19]:


data["Owner_Type"] = encoder.fit_transform(data["Owner_Type"].fillna('Nan'))
data["Owner_Type"]


# In[22]:


pd.DataFrame(data)


# In[32]:


x = pd.get_dummies(data,
                         columns = ["Vehicle_Name", "Vehicle_Type", "Fuel_Type", "Owner_Type"],
                         drop_first = True)
y = data['Selling_Price']


# In[27]:


from sklearn.linear_model import LinearRegression


# In[28]:


lm = LinearRegression()


# In[30]:


from sklearn.model_selection import train_test_split


# In[33]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=45)


# In[34]:


lm.fit(X_train,y_train)


# In[52]:


# print the intercept
print(lm.intercept_)


# In[54]:


coeff_df = pd.DataFrame(lm.coef_,x.columns,columns=['Coefficient'])
coeff_df


# In[36]:


p = lm.predict(X_test)


# In[44]:


from sklearn.metrics import r2_score
r2_score(y_test, p)


# In[45]:


plt.scatter(y_test,p)


# In[49]:


sns.distplot((y_test-p),bins=20);


# In[51]:


from sklearn import metrics
print('MAE:', metrics.mean_absolute_error(y_test, p))
print('MSE:', metrics.mean_squared_error(y_test, p))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, p)))

