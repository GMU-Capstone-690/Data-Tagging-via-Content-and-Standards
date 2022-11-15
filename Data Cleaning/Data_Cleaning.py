#!/usr/bin/env python
# coding: utf-8

# # Cleaning Datasets

# In[9]:


# importing required libraries
import pandas as pd
import urllib
from urllib.request import urlopen, URLError, HTTPError
import os
# setting directory 
os.getcwd()


# ## URL Validity Checking Function

# In[10]:


#UDF for checking the invalidity of the URLs
def validate_url(dataframe):
    url_status = []
    for url in dataframe.url:
        try:
            urlopen(url)
            url_status.append("valid")
        except  (urllib.error.URLError, urllib.error.HTTPError):
            url_status.append("invalid")
    return url_status


# ## Aljazeera News Dataset

# In[11]:


aljazeera=pd.read_csv("aljazeera.csv")


# In[12]:


aljazeera.head(3)


# In[13]:


aljazeera.describe()


# In[66]:


# checking null values for each column
print(aljazeera.isna().sum())
# checking null values for url column
print('null url values= %d'%( aljazeera['url'].isna().sum()))
# checking blank values for each column
print('blank url values= %d' % (aljazeera['url'].values == '').sum())
# checking tap values for each column
print('tap url values= %d' % (aljazeera['url'].values == ' ').sum())
# checking 'na' values for each column
print('"na" url values= %d' % (aljazeera['url'].values == 'na').sum())


# In[18]:


# checking duplicate values for url column
print('url = %d' % (aljazeera['url'].duplicated()).sum())


# In[19]:


# dropping rows with duplicate url values
aljazeera = aljazeera.drop_duplicates(subset=['url'])


# In[20]:


## Checking the URLS invalidity
url_status = validate_url(aljazeera)
aljazeera.insert(0,'url_status', url_status)
aljazeera['url_status'].value_counts()


# In[21]:


# dropping invalid urls
aljazeera.drop(aljazeera.loc[aljazeera['url_status']=="invalid"].index, inplace=True)


# In[22]:


# saving cleaned dataset to CSV file
aljazeera.to_csv('aljazeera_cleaned.csv', index=False)


# ## BBC News Dataset

# In[23]:


bbc = pd.read_csv("bbc.csv")


# In[24]:


bbc.head(3)


# In[25]:


bbc.describe()


# In[67]:


# checking null values for each column
print(bbc.isna().sum())
# checking null values for url column
print('null url values= %d'%(bbc['url'].isna().sum()))
# checking blank values for each column
print('blank url values= %d' % (bbc['url'].values == '').sum())
# checking tap values for each column
print('tap url values= %d' % (bbc['url'].values == ' ').sum())
# checking 'na' values for each column
print('"na" url values= %d' % (bbc['url'].values == 'na').sum())


# In[28]:


# checking duplicate values for url column
print('duplicate urls = %d' % (bbc['url'].duplicated()).sum())


# In[29]:


# dropping rows with duplicate url values
bbc = bbc.drop_duplicates(subset=['url'])


# In[30]:


# checking urls validty
url_status = validate_url(bbc)
bbc.insert(0,'url_status', url_status)
bbc['url_status'].value_counts()


# In[31]:


# dropping invalid urls
bbc.drop(bbc.loc[bbc['url_status']=="invalid"].index, inplace=True)


# In[32]:


# saving cleaned dataset to CSV file
bbc.to_csv('bbc_cleaned.csv', index=False)


# ## CNBC News Dataset

# In[33]:


cnbc=pd.read_csv("cnbc_news_dataset.csv")


# In[34]:


cnbc.head(3)


# In[35]:


cnbc.describe()


# In[70]:


# checking null values for each column
print(cnbc.isna().sum())
# checking null values for url column
print('null url values= %d'%(cnbc['url'].isna().sum()))
# checking blank values for each column
print('blank url values= %d' % (cnbc['url'].values == '').sum())
# checking tap values for each column
print('tap url values= %d' % (cnbc['url'].values == ' ').sum())
# checking 'na' values for each column
print('"na" url values= %d' % (cnbc['url'].values == 'na').sum())


# In[37]:


# checking duplicate values for url column
print('url = %d' % (cnbc['url'].duplicated()).sum())


# In[38]:


# dropping rows with duplicate url values
cnbc = cnbc.drop_duplicates(subset=['url'])


# In[39]:


# Checking the URLS invalidity
url_status = validate_url(cnbc)
cnbc.insert(0,'url_status', url_status)
cnbc['url_status'].value_counts()


# In[40]:


# dropping invalid urls
cnbc.drop(cnbc.loc[cnbc['url_status']=="invalid"].index, inplace=True)


# In[41]:


# saving cleaned dataset to CSV file
cnbc.to_csv('cnbc_cleaned.csv', index=False)


# ## CNN News Dataset

# In[42]:


cnn=pd.read_csv("cnn.csv")


# In[43]:


cnn.head(3)


# In[44]:


cnn.describe()


# In[71]:


# checking null values for each column
print(cnn.isna().sum())
# checking null values for url column
print('null url values= %d'%(cnn['url'].isna().sum()))
# checking blank values for each column
print('blank url values= %d' % (cnn['url'].values == '').sum())
# checking tap values for each column
print('tap url values= %d' % (cnn['url'].values == ' ').sum())
# checking 'na' values for each column
print('"na" url values= %d' % (cnn['url'].values == 'na').sum())


# In[46]:


# checking duplicate values for url column
print('url = %d' % (cnn['url'].duplicated()).sum())


# In[47]:


# dropping rows with duplicate url values
cnn = cnn.drop_duplicates(subset=['url'])


# In[48]:


# Checking the URLS invalidity
url_status = validate_url(cnn)
cnn.insert(0,'url_status', url_status)
cnn['url_status'].value_counts()


# In[49]:


# dropping invalid urls
cnn.drop(cnn.loc[cnn['url_status']=="invalid"].index, inplace=True)


# In[50]:


# saving cleaned dataset to CSV file
cnn.to_csv('cnn_cleaned.csv', index=False)


# ## Japan Times News Dataset

# In[51]:


japan_times=pd.read_csv("japan_times.csv")


# In[52]:


japan_times.head(3)


# In[53]:


japan_times.describe()


# In[72]:


# checking null values for each column
print(japan_times.isna().sum())
# checking null values for url column
# checking null values for each column
print('null url values= %d'%(japan_times['url'].isna().sum()))
# checking blank values for each column
print('blank url values= %d' % (japan_times['url'].values == '').sum())
# checking tap values for each column
print('tap url values= %d' % (japan_times['url'].values == ' ').sum())
# checking 'na' values for each column
print('"na" url values= %d' % (japan_times['url'].values == 'na').sum())


# In[55]:


# checking duplicate values for url column
print('url = %d' % (japan_times['url'].duplicated()).sum())


# In[56]:


# dropping rows with duplicate url values
japan_times = japan_times.drop_duplicates(subset=['url'])


# In[58]:


# Checking the URLS invalidity
url_status = validate_url(japan_times)
japan_times.insert(0,'url_status', url_status)
japan_times['url_status'].value_counts()


# In[59]:


# dropping invalid urls
japan_times.drop(japan_times.loc[japan_times['url_status']=="invalid"].index, inplace=True)


# In[60]:


# saving cleaned dataset to CSV file
japan_times.to_csv('japan_times_cleaned.csv', index=False)


# In[ ]:




