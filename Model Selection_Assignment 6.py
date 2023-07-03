#!/usr/bin/env python
# coding: utf-8

# In[23]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[24]:


df =pd.read_csv('car data.csv')


# In[25]:


df.head(5)


# In[26]:


df.info()


# In[27]:


df.isnull().sum()


# In[28]:


df.duplicated().sum()


# In[29]:


df.describe()


# In[30]:


corr = df.corr(method='kendall')
plt.figure(figsize=(12,8))
sns.heatmap(corr, annot=True)
df.columns


# In[43]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import scale 

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[44]:


x = df.loc[:, ['Car_Name', 'Present_Price', 'Kms_Driven', 'Fuel_Type','Seller_Type', 'Transmission', 'Owner']]

y = df['Selling_Price']


# In[45]:



df_categorical = x.select_dtypes(include=['object'])

df_dummies = pd.get_dummies(df_categorical, drop_first=True)

x = x.drop(list(df_categorical.columns), axis=1)

x = pd.concat([x, df_dummies], axis=1)


# In[46]:


cols = x.columns
x = pd.DataFrame(scale(x))
x.columns = cols
x.columns


# In[67]:



from sklearn.model_selection import train_test_split
x_train,x_test, y_train, y_test = train_test_split(x, y, 
                                                    train_size=0.7,
                                                    test_size = 0.3, random_state=100)


# In[70]:



lm = LinearRegression()

lm.fit(x_train, y_train)
 
y_pred = lm.predict(X_test)

from sklearn.metrics import r2_score
print(r2_score(y_true=y_test, y_pred=y_pred))


# In[71]:



lm = LinearRegression()
n_features = 6

rfe_n = RFE(lm, n_features)

rfe_n.fit(x_train, y_train)

col_n = x_train.columns[rfe_n.support_]

x_train_rfe_n = x_train[col_n]

x_train_rfe_n = sm.add_constant(x_train_rfe_n)

lm_n = sm.OLS(y_train, x_train_rfe_n).fit()
adjusted_r2.append(lm_n.rsquared_adj)
r2.append(lm_n.rsquared)

x_test_rfe_n = x_test[col_n]
 
x_test_rfe_n = sm.add_constant(x_test_rfe_n, has_constant='add')

y_pred = lm_n.predict(x_test_rfe_n)
test_r2.append(r2_score(y_test, y_pred))

lm_n.summary()


# In[ ]:





# In[ ]:




