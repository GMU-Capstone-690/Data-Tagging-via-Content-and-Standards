import pandas as pd

### BBC News Dataset

bbc=pd.read_csv("C:/DEAN/Datasets/bbc.csv")
bbc.head()
bbc.describe()
bbc.columns

# checking null values for each column
bbc.columns.isna().sum()

# checking blank values for each column
print('title = %d' % (bbc['title'].values == '').sum())
print('url = %d' % (bbc['url'].values == '').sum())

# checking tap values for each column
print('title = %d' % (bbc['title'].values == ' ').sum())
print('url = %d' % (bbc['url'].values == ' ').sum())

# checking 'na' values for each column
print('title = %d' % (bbc['title'].values == 'na').sum())
print('url = %d' % (bbc['url'].values == 'na').sum())

# checking duplicate values for url column
print('url = %d' % (bbc['url'].duplicated()).sum())

#checking the invalidity of the URLs
from urllib.request import urlopen, URLError, HTTPError

def validate_web_url(dataframe):
    url_status = []
    for url in dataframe.url:
        try:
            urlopen(url)
            url_status.append("valid")
        except  (urllib.error.URLError, urllib.error.HTTPError):
            url_status.append("invalid")
    return url_status

url_status = validate_web_url(bbc)
bbc.insert(0,'url_status', url_status)
bbc['url_status'].value_counts()

#droping rows that have invalid url
bbc.drop(bbc.loc[bbc['url_status']=="invalid"].index, inplace=True)


### ALJAZEERA NEWS DATASETS

# In[ ]:


aljazeera=pd.read_csv("C:/DEAN/Datasets/aljazeera.csv")


# In[ ]:


aljazeera.head()


# In[ ]:


aljazeera.describe()


# In[ ]:


aljazeera.columns


# In[ ]:


# checking null values for each column
aljazeera.isna().sum()


# In[ ]:


# checking blank values for each column
print('title = %d' % (aljazeera['title'].values == '').sum())
print('url = %d' % (aljazeera['url'].values == '').sum())


# In[ ]:


# checking tap values for each column
print('title = %d' % (aljazeera['title'].values == ' ').sum())
print('url = %d' % (aljazeera['url'].values == ' ').sum())


# In[ ]:


# checking 'na' values for each column
print('title = %d' % (aljazeera['title'].values == 'na').sum())
print('url = %d' % (aljazeera['url'].values == 'na').sum())


# In[ ]:


# checking duplicate values for url column
print('url = %d' % (aljazeera['url'].duplicated()).sum())


# In[ ]:


# dropping rows with duplicate url values
aljazeera = aljazeera.drop_duplicates(subset=['url'])


# In[ ]:


## Checking the URLS invalidity
url_status = validate_web_url(aljazeera)
aljazeera.insert(0,'url_status', url_status)
aljazeera['url_status'].value_counts()


# In[ ]:


aljazeera.drop(aljazeera.loc[aljazeera['url_status']=="invalid"].index, inplace=True)


# ## CNBC NEWS DATASETS

# In[ ]:


cnbc=pd.read_csv("C:/DEAN/Datasets/cnbc_news_datase.csv")


# In[ ]:


cnbc.head()


# In[ ]:


cnbc.describe()


# In[ ]:


cnbc.columns


# In[ ]:


# checking null values for each column
cnbc.isna().sum()


# In[ ]:


# checking blank values for each column
print('title = %d' % (cnbc['title'].values == '').sum())
print('url = %d' % (cnbc['url'].values == '').sum())


# In[ ]:


# checking tap values for each column
print('title = %d' % (cnbc['title'].values == ' ').sum())
print('url = %d' % (cnbc['url'].values == ' ').sum())


# In[ ]:


# checking 'na' values for each column
print('title = %d' % (cnbc['title'].values == 'na').sum())
print('url = %d' % (cnbc['url'].values == 'na').sum())


# In[ ]:


# checking duplicate values for url column
print('url = %d' % (cnbc['url'].duplicated()).sum())


# In[ ]:


## Checking the URLS invalidity
url_status = validate_web_url(cnbc)
cnbc.insert(0,'url_status', url_status)
cnbc['url_status'].value_counts()


# In[ ]:


cnbc.drop(cnbc.loc[cnbc['url_status']=="invalid"].index, inplace=True)


# ## CNN NEWS DATASET

# In[ ]:


cnn=pd.read_csv("C:/DEAN/Datasets/cnn.csv")


# In[ ]:


cnn.head()


# In[ ]:


cnn.describe()


# In[ ]:


cnn.columns


# In[ ]:


# checking null values for each column
cnn.isna().sum()


# In[ ]:


# checking blank values for each column
print('title = %d' % (cnn['title'].values == '').sum())
print('url = %d' % (cnn['url'].values == '').sum())


# In[ ]:


# checking tap values for each column
print('title = %d' % (cnn['title'].values == ' ').sum())
print('url = %d' % (cnn['url'].values == ' ').sum())


# In[ ]:


# checking 'na' values for each column
print('title = %d' % (cnn['title'].values == 'na').sum())
print('url = %d' % (cnn['url'].values == 'na').sum())


# In[ ]:


# checking duplicate values for url column
print('url = %d' % (cnn['url'].duplicated()).sum())


# In[ ]:


# dropping rows with duplicate url values
cnn = cnn.drop_duplicates(subset=['url'])


# In[ ]:


## Checking the URLS invalidity
url_status = validate_web_url(cnn)
cnn.insert(0,'url_status', url_status)
cnn['url_status'].value_counts()


# In[ ]:


cnn.drop(cnn.loc[cnn['url_status']=="invalid"].index, inplace=True)


# ## JAPAN TIMES NEWS DATASET

# In[ ]:


japan_times=pd.read_csv("C:/DEAN/Datasets/japan_times.csv")


# In[ ]:


japan_times.head()


# In[ ]:


japan_times.describe()


# In[ ]:


japan_times.columns


# In[ ]:


# checking null values for each column
japan_times.isna().sum()


# In[ ]:


# checking blank values for each column
print('title = %d' % (japan_times['headline'].values == '').sum())
print('url = %d' % (japan_times['url'].values == '').sum())


# In[ ]:


# checking tap values for each column
print('title = %d' % (japan_times['headline'].values == ' ').sum())
print('url = %d' % (japan_times['url'].values == ' ').sum())


# In[ ]:


# checking 'na' values for each column
print('title = %d' % (japan_times['headline'].values == 'na').sum())
print('url = %d' % (japan_times['url'].values == 'na').sum())


# In[ ]:


# checking duplicate values for url column
print('url = %d' % (japan_times['url'].duplicated()).sum())


# In[ ]:


## Checking the URLS invalidity
url_status = validate_web_url(cnn)
japan_times.insert(0,'url_status', url_status)
japan_times['url_status'].value_counts()


# In[ ]:


japan_times.drop(japan_times.loc[japan_times['url_status']=="invalid"].index, inplace=True)


# In[ ]:




