#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


import warnings
warnings.filterwarnings('ignore')


plt.style.use("fivethirtyeight")


# In[15]:


df=pd.read_csv("C:/Users/ASUS/OneDrive/Desktop/archive (3)/Iris.csv")
df.head()


# In[16]:


#information about the dataset
df.info() 


# In[17]:


#describing about the dataset
df.describe()


# In[18]:


df.shape


# In[19]:


df.drop('Id',axis=1,inplace=True)


# In[20]:


df.head()


# In[21]:


#count the value
df['Species'].value_counts()


# In[22]:


#finding the null value
df.isnull().sum()


# In[23]:


import missingno as msno
msno.bar(df)


# In[24]:


df.drop_duplicates(inplace=True)


# In[25]:


#1. Relationship between species and sepal length

plt.figure(figsize=(15,8))
sns.boxplot(x='Species',y='SepalLengthCm',data=df.sort_values('SepalLengthCm',ascending=False))


# In[26]:


#2. Relationship between species and sepal width

df.plot(kind='scatter',x='SepalWidthCm',y='SepalLengthCm')


# In[27]:


#3. Relationship between sepal width and sepal length

sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=df, size=5)


# In[28]:


#4.Pairplot

sns.pairplot(df, hue="Species", size=3)


# In[29]:


#5. Boxplot

df.boxplot(by="Species", figsize=(12, 6))


# In[30]:


#6. Andrews_curves
import pandas.plotting
from pandas.plotting import andrews_curves
andrews_curves(df, "Species")


# In[31]:


#7.CategoricalPlot

plt.figure(figsize=(15,15))
sns.catplot(x='Species',y='SepalWidthCm',data=df.sort_values('SepalWidthCm',ascending=False),kind='boxen')


# In[32]:


# 8.Violinplot

plt.figure(figsize=(15,10))
plt.subplot(2,2,1)
sns.violinplot(x='Species',y='PetalLengthCm',data=df)
plt.subplot(2,2,2)
sns.violinplot(x='Species',y='PetalWidthCm',data=df)
plt.subplot(2,2,3)
sns.violinplot(x='Species',y='SepalLengthCm',data=df)
plt.subplot(2,2,4)
sns.violinplot(x='Species',y='SepalWidthCm',data=df)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




