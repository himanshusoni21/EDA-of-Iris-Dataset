#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import datasets


# In[7]:


iris = datasets.load_iris()
dir(iris)


# In[20]:


df = pd.DataFrame(iris.data,columns=iris.feature_names)
df.head()


# In[39]:


df['target'] = iris.target
df.head()


# In[40]:


df['species'] = df.target.apply(lambda x:iris.target_names[x])
df.head()


# In[41]:


del df['target']
df.head()


# In[45]:


df_setosa = df.loc[df['species']=='setosa']
df_virginica = df.loc[df['species']=='virginica']
df_versicolor = df.loc[df['species']=='versicolor']


# In[56]:


plt.plot(df_setosa['sepal length (cm)'],np.zeros_like(df_setosa['sepal length (cm)']),'o',color='green')
plt.plot(df_virginica['sepal length (cm)'],np.zeros_like(df_virginica['sepal length (cm)']),'o',color='blue')
plt.plot(df_versicolor['sepal length (cm)'],np.zeros_like(df_versicolor['sepal length (cm)']),'o',color='red')
plt.xlabel('Sepal Length')


# ## Bivariate Analysis

# In[60]:


sns.FacetGrid(df,hue='species',size=5).map(plt.scatter,'petal length (cm)','sepal length (cm)').add_legend()
plt.show()


# ## Multivariate Analysis

# In[63]:


sns.pairplot(df,hue='species',size=4)

