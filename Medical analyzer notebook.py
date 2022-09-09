#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


df = pd.read_csv('C:/Users/pc/Desktop/New folder/DataAnalysisTerm/medical_examination.csv', index_col='id')


# In[5]:


df


# >>>Obesity Check

# In[6]:


over_wieght = (df['weight'] / (df['height'] / 100)**2).astype(int)
df['overweight'] = (over_wieght > 25)
df


# ...Getting obesity number and percentage.

# In[7]:


obe_num = df['overweight'].value_counts(True)
total = df['overweight'].value_counts()
print(obe_num, total)


# ### The percentage of obesity to the total is 53% and 46% for normal_weighted .

# In[8]:


for value in df.groupby('overweight'):
    if value == True:
        value = 1
    else:
        value=0
    print(df['overweight'])


# In[9]:


df['overweight'] = df['overweight'].replace({True:1, False:0})
df


# ## Next step we will normalize the dataset by setting 0 as always bad and 1 value as always good .

# In[10]:


df[['cholesterol', 'gluc']] = (df[['cholesterol', 'gluc']]>1).astype(int)
df


# In[ ]:


cat_plot = sns.catplot(data=df, x=)


# In[11]:


df_cat = pd.melt(frame=df, id_vars=['cardio'], value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'])


# In[12]:


df_cat


# In[43]:


def draw_cat_plot():
    fig = sns.catplot(data=df_cat, kind='count', x='variable',hue='value' ).set(ylabel='total').fig
    return fig

draw_cat_plot()


# ### From previous figure we can see that alcohol has the worst impact on heart's health and smoking comes after it.

# In[44]:


#Drawing heat map and using the correlation matrix
def draw_heat_map():
    # Clean the data
    df_heat = df[ 
        ( df['ap_lo'] <= df['ap_hi'] ) & 
        ( df['height'] >= df['height'].quantile(0.025) ) & 
        ( df['height'] <= df['height'].quantile(0.975) ) & 
        ( df['weight'] >= df['weight'].quantile(0.025) ) & 
        ( df['weight'] <= df['weight'].quantile(0.975) ) 
    ]

    # Calculate the correlation matrix
    corr = df_heat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(corr)


    # Set up the matplotlib figure
    fig, ax =  plt.subplots()
    
    # Draw the heatmap with 'sns.heatmap()'
    ax = sns.heatmap(corr, mask=mask, annot=True, fmt='0.1f', square=True)


    # Do not modify the next two lines
    fig.savefig('heatmap.png')
    return fig
draw_heat_map()


# In[45]:





# In[ ]:




