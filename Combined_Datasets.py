# Importing Libraries 
import pandas as pd
import numpy as np


# Reading cnbc dataset
cnbc = pd.read_csv(".../cnbc_extracted.csv", keep_default_na=False)


# Reading huffpost dataset
huffpost = pd.read_csv(".../huffpost_extracted.csv" ,keep_default_na=False)


# Reading aljazeera dataset
aljazeera = pd.read_csv(".../aljazeera_extracted.csv", keep_default_na=False)


# Reading bbc dataset
bbc = pd.read_csv(".../bbc_extracted.csv", keep_default_na=False)


# Reading japan_times dataset
japan_times = pd.read_csv(".../japan_times_extracted.csv", keep_default_na=False)


# Reading cnn dataset
cnn = pd.read_csv(".../cnn_extracted.csv", keep_default_na=False)


#Just selecting seven columns as 'title', 'url', 'published_at', 'publisher', 'tags', 'html_content', 'clean_html_text'
#So have to rename columns of all datasets for consistent combining 


#Renaming columns for cnbc dataset
cnbc_subset = cnbc[['title', 'url', 'published_at', 'publisher', 'keywords', 'html_content', 'clean_html_text']]
cnbc_subset.rename(columns={ "keywords":"tags"}, inplace=True)
cnbc_subset.head(1)


#Renaming columns for huffpost dataset
huffpost_subset = huffpost[['headline', 'url', 'published_at', 'source_title', 'tags', 'html_content', 'clean_html_text']]
huffpost_subset.rename(columns={ "headline":"title","source_title":"publisher"}, inplace=True)
huffpost_subset.head(1)


#Renaming columns for aljazeera dataset
aljazeera_subset = aljazeera[['title', 'url', 'epoch_time', 'website', 'sub_category', 'html_content', 'clean_html_text']]
aljazeera_subset.rename(columns={ "website":"publisher","epoch_time":"published_at","sub_category":"tags"}, inplace=True)
aljazeera_subset.head(1)


#Renaming columns for bbc dataset
bbc_subset = bbc[['title', 'url', 'news_post_date', 'category', 'tags', 'html_content', 'clean_html_text']]
bbc_subset.rename(columns={ "news_post_date":"published_at","category":"publisher"}, inplace=True)
bbc_subset.head(1)


#Renaming columns for japan_times dataset
japan_times_subset = japan_times[['headline','url', 'datePublished', 'siteName', 'keywords', 'html_content', 'clean_html_text']]
japan_times_subset.rename(columns={ "headline":"title","datePublished":"published_at","siteName":"publisher","keywords":"tags"}, inplace=True)
japan_times_subset.head(1)


#Renaming columns for cnn dataset
cnn_subset = cnn[['title', 'url', 'published_at', 'source', 'html_content', 'clean_html_text']]
cnn_subset.rename(columns={ "source":"publisher"}, inplace=True)
cnn_subset.insert(4, "tags"," ",True)
cnn_subset.head(1)


#Concatenating all datasets 
all_news = pd.concat([cnbc_subset, huffpost_subset, aljazeera_subset, bbc_subset, japan_times_subset, cnn_subset])


# Writing csv file locally 
all_news.to_csv('/Users/.../all_news.csv', index=False)
