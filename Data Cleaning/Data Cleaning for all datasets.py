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


#dropping rows that have invalid url
bbc.drop(bbc.loc[bbc['url_status']=="invalid"].index, inplace=True)


### ALJAZEERA NEWS DATASETS


aljazeera=pd.read_csv("C:/DEAN/Datasets/aljazeera.csv")
aljazeera.head()
aljazeera.describe()
aljazeera.columns


# checking null values for each column
aljazeera.isna().sum()


# checking blank values for each column
print('title = %d' % (aljazeera['title'].values == '').sum())
print('url = %d' % (aljazeera['url'].values == '').sum())


# checking tap values for each column
print('title = %d' % (aljazeera['title'].values == ' ').sum())
print('url = %d' % (aljazeera['url'].values == ' ').sum())


# checking 'na' values for each column
print('title = %d' % (aljazeera['title'].values == 'na').sum())
print('url = %d' % (aljazeera['url'].values == 'na').sum())


# checking duplicate values for url column
print('url = %d' % (aljazeera['url'].duplicated()).sum())


# dropping rows with duplicate url values
aljazeera = aljazeera.drop_duplicates(subset=['url'])


## Checking the URLS invalidity
url_status = validate_web_url(aljazeera)
aljazeera.insert(0,'url_status', url_status)
aljazeera['url_status'].value_counts()


#dropping rows that have invalid url
aljazeera.drop(aljazeera.loc[aljazeera['url_status']=="invalid"].index, inplace=True)



### CNBC NEWS DATASETS


cnbc=pd.read_csv("C:/DEAN/Datasets/cnbc_news_datase.csv")
cnbc.head()
cnbc.describe()
cnbc.columns


# checking null values for each column
cnbc.isna().sum()


# checking blank values for each column
print('title = %d' % (cnbc['title'].values == '').sum())
print('url = %d' % (cnbc['url'].values == '').sum())


# checking tap values for each column
print('title = %d' % (cnbc['title'].values == ' ').sum())
print('url = %d' % (cnbc['url'].values == ' ').sum())


# checking 'na' values for each column
print('title = %d' % (cnbc['title'].values == 'na').sum())
print('url = %d' % (cnbc['url'].values == 'na').sum())


# checking duplicate values for url column
print('url = %d' % (cnbc['url'].duplicated()).sum())


## Checking the URLS invalidity
url_status = validate_web_url(cnbc)
cnbc.insert(0,'url_status', url_status)
cnbc['url_status'].value_counts()


#dropping rows that have invalid url
cnbc.drop(cnbc.loc[cnbc['url_status']=="invalid"].index, inplace=True)



### CNN NEWS DATASET


cnn=pd.read_csv("C:/DEAN/Datasets/cnn.csv")
cnn.head()
cnn.describe()
cnn.columns


# checking null values for each column
cnn.isna().sum()


# checking blank values for each column
print('title = %d' % (cnn['title'].values == '').sum())
print('url = %d' % (cnn['url'].values == '').sum())


# checking tap values for each column
print('title = %d' % (cnn['title'].values == ' ').sum())
print('url = %d' % (cnn['url'].values == ' ').sum())


# checking 'na' values for each column
print('title = %d' % (cnn['title'].values == 'na').sum())
print('url = %d' % (cnn['url'].values == 'na').sum())


# checking duplicate values for url column
print('url = %d' % (cnn['url'].duplicated()).sum())


# dropping rows with duplicate url values
cnn = cnn.drop_duplicates(subset=['url'])


## Checking the URLS invalidity
url_status = validate_web_url(cnn)
cnn.insert(0,'url_status', url_status)
cnn['url_status'].value_counts()


#dropping rows that have invalid url 
cnn.drop(cnn.loc[cnn['url_status']=="invalid"].index, inplace=True)



### JAPAN TIMES NEWS DATASET


japan_times=pd.read_csv("C:/DEAN/Datasets/japan_times.csv")
japan_times.head()
japan_times.describe()
japan_times.columns


# checking null values for each column
japan_times.isna().sum()


# checking blank values for each column
print('title = %d' % (japan_times['headline'].values == '').sum())
print('url = %d' % (japan_times['url'].values == '').sum())


# checking tap values for each column
print('title = %d' % (japan_times['headline'].values == ' ').sum())
print('url = %d' % (japan_times['url'].values == ' ').sum())


# checking 'na' values for each column
print('title = %d' % (japan_times['headline'].values == 'na').sum())
print('url = %d' % (japan_times['url'].values == 'na').sum())


# checking duplicate values for url column
print('url = %d' % (japan_times['url'].duplicated()).sum())


## Checking the URLS invalidity
url_status = validate_web_url(cnn)
japan_times.insert(0,'url_status', url_status)
japan_times['url_status'].value_counts()


#dropping rows that have invalid url
japan_times.drop(japan_times.loc[japan_times['url_status']=="invalid"].index, inplace=True)
