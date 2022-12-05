
# # Cleaning Datasets



# importing required libraries
import pandas as pd
import urllib
from urllib.request import urlopen, URLError, HTTPError
import os
# setting directory 
os.getcwd()


# ## URL Validity Checking Function



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



aljazeera=pd.read_csv("aljazeera.csv")

aljazeera.head(3)

aljazeera.describe()





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





# checking duplicate values for url column
print('url = %d' % (aljazeera['url'].duplicated()).sum())





# dropping rows with duplicate url values
aljazeera = aljazeera.drop_duplicates(subset=['url'])




## Checking the URLS invalidity
url_status = validate_url(aljazeera)
aljazeera.insert(0,'url_status', url_status)
aljazeera['url_status'].value_counts()




# dropping invalid urls
aljazeera.drop(aljazeera.loc[aljazeera['url_status']=="invalid"].index, inplace=True)




# saving cleaned dataset to CSV file
aljazeera.to_csv('aljazeera_cleaned.csv', index=False)


# ## BBC News Dataset



bbc = pd.read_csv("bbc.csv")

bbc.head(3)

bbc.describe()





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





# checking duplicate values for url column
print('duplicate urls = %d' % (bbc['url'].duplicated()).sum())





# dropping rows with duplicate url values
bbc = bbc.drop_duplicates(subset=['url'])





# checking urls validty
url_status = validate_url(bbc)
bbc.insert(0,'url_status', url_status)
bbc['url_status'].value_counts()





# dropping invalid urls
bbc.drop(bbc.loc[bbc['url_status']=="invalid"].index, inplace=True)





# saving cleaned dataset to CSV file
bbc.to_csv('bbc_cleaned.csv', index=False)


# ## CNBC News Dataset




cnbc=pd.read_csv("cnbc_news_dataset.csv")

cnbc.head(3)

cnbc.describe()





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





# checking duplicate values for url column
print('url = %d' % (cnbc['url'].duplicated()).sum())





# dropping rows with duplicate url values
cnbc = cnbc.drop_duplicates(subset=['url'])





# Checking the URLS invalidity
url_status = validate_url(cnbc)
cnbc.insert(0,'url_status', url_status)
cnbc['url_status'].value_counts()





# dropping invalid urls
cnbc.drop(cnbc.loc[cnbc['url_status']=="invalid"].index, inplace=True)





# saving cleaned dataset to CSV file
cnbc.to_csv('cnbc_cleaned.csv', index=False)


# ## CNN News Dataset




cnn=pd.read_csv("cnn.csv")

cnn.head(3)

cnn.describe()





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





# checking duplicate values for url column
print('url = %d' % (cnn['url'].duplicated()).sum())




# dropping rows with duplicate url values
cnn = cnn.drop_duplicates(subset=['url'])





# Checking the URLS invalidity
url_status = validate_url(cnn)
cnn.insert(0,'url_status', url_status)
cnn['url_status'].value_counts()





# dropping invalid urls
cnn.drop(cnn.loc[cnn['url_status']=="invalid"].index, inplace=True)




# saving cleaned dataset to CSV file
cnn.to_csv('cnn_cleaned.csv', index=False)


# ## Japan Times News Dataset




japan_times=pd.read_csv("japan_times.csv")

japan_times.head(3)

japan_times.describe()





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





# checking duplicate values for url column
print('url = %d' % (japan_times['url'].duplicated()).sum())





# dropping rows with duplicate url values
japan_times = japan_times.drop_duplicates(subset=['url'])





# Checking the URLS invalidity
url_status = validate_url(japan_times)
japan_times.insert(0,'url_status', url_status)
japan_times['url_status'].value_counts()





# dropping invalid urls
japan_times.drop(japan_times.loc[japan_times['url_status']=="invalid"].index, inplace=True)





# saving cleaned dataset to CSV file
japan_times.to_csv('japan_times_cleaned.csv', index=False)







