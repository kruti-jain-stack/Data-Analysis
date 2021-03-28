#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


google_data = pd.read_csv('C:/Users/genz/Documents/Data Analysis/appPrediction.csv')


# In[6]:


type(google_data)


# In[8]:


google_data.head(10)


# In[9]:


google_data.tail(7)


# In[10]:


google_data.shape


# In[12]:


google_data.describe()


# In[13]:


google_data.boxplot()


# In[14]:


google_data.hist()


# In[17]:


google_data.info()


# In[18]:


google_data.isnull()


# In[19]:


google_data.isnull().sum()


# In[20]:


google_data[google_data.Rating > 5 ]


# In[21]:


google_data.drop([10472],inplace=True)


# In[22]:


google_data[10470:10474]


# In[23]:


google_data.boxplot()


# In[24]:


google_data.hist()


# In[25]:


threshold = len(google_data)* 0.1
threshold


# In[26]:


google_data.dropna(thresh=threshold, axis=1, inplace=True) #axis 0 means row and axis 1 means column


# In[27]:


print(google_data.isnull().sum())


# In[28]:


google_data.shape


# In[29]:


#Define a function 
def compute_median(series):
    return series.fillna(series.median())


# In[30]:


google_data.Rating = google_data['Rating'].transform(compute_median)


# In[31]:


google_data.isnull()


# In[32]:


google_data.isnull().sum()


# In[33]:


# modes of categorical values
print(google_data['Type'].mode())
print(google_data['Current Ver'].mode())
print(google_data['Android Ver'].mode())


# In[34]:


google_data['Type'].fillna(str(google_data['Type'].mode().values[0]), inplace=True)
google_data['Current Ver'].fillna(str(google_data['Current Ver'].mode().values[0]), inplace=True)
google_data['Android Ver'].fillna(str(google_data['Android Ver'].mode().values[0]), inplace=True)


# In[37]:


google_data.isnull().sum()


# In[38]:


google_data['Price'] = google_data['Price'].apply(lambda x: str(x).replace('$', '') if '$' in str(x) else str(x))
google_data['Price'] = google_data['Price'].apply(lambda x: float(x))
google_data['Reviews'] = pd.to_numeric(google_data['Reviews'], errors='coerce')


# In[39]:


google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace('+', '') if '+' in str(x) else str(x))
google_data['Installs'] = google_data['Installs'].apply(lambda x: str(x).replace(',', '') if ',' in str(x) else str(x))
google_data['Installs'] = google_data['Installs'].apply(lambda x: float(x))


# In[40]:


google_data.head(20)


# In[41]:


google_data.describe()


# In[42]:


#grouping of the data
grp = google_data.groupby('Category')
x = grp['Rating'].agg(np.mean)
y = grp['Price'].agg(np.sum)
z = grp['Reviews'].agg(np.mean)
print(x)
print(y)
print(z)


# In[44]:


plt.plot(x)


# In[45]:


plt.plot(x,'ro')


# In[47]:


plt.plot(x, "ro", color='y')
plt.xticks(rotation=90)


# In[43]:


plt.figure(figsize=(12,5))
plt.plot(x, "ro", color='r')
plt.xticks(rotation=90)
plt.show()


# In[48]:


plt.figure(figsize=(15,5))
plt.plot(x,'ro', color='b')
plt.xticks(rotation=90)
plt.title('Category wise Rating')
plt.xlabel('Categories-->')
plt.ylabel('Rating-->')
plt.show()


# In[49]:


plt.figure(figsize=(15,5))
plt.plot(y,'r--', color='y')
plt.xticks(rotation=90)
plt.title('Category wise Pricing')
plt.xlabel('Categories-->')
plt.ylabel('Prices-->')
plt.show()


# In[50]:


plt.figure(figsize=(15,5))
plt.plot(z,'bs', color='g')
plt.xticks(rotation=90)
plt.title('Category wise Reviews')
plt.xlabel('Categories-->')
plt.ylabel('Reviews-->')
plt.show()


# In[ ]:




