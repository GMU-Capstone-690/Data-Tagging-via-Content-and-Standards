#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np


# In[49]:


cnbc = pd.read_csv(".../cnbc_extracted.csv", keep_default_na=False)


# In[70]:


huffpost = pd.read_csv(".../huffpost_extracted.csv" ,keep_default_na=False)


# In[10]:


aljazeera = pd.read_csv(".../aljazeera_extracted.csv", keep_default_na=False)


# In[11]:


bbc = pd.read_csv(".../bbc_extracted.csv", keep_default_na=False)


# In[12]:


japan_times = pd.read_csv(".../japan_times_extracted.csv", keep_default_na=False)


# In[88]:


cnn = pd.read_csv(".../cnn_extracted.csv", keep_default_na=False)


# In[50]:


cnbc_subset = cnbc[['title', 'url', 'published_at', 'publisher', 'keywords', 'html_content', 'clean_html_text']]
cnbc_subset.rename(columns={ "keywords":"tags"}, inplace=True)
cnbc_subset.head(1)


# In[57]:


crypto_subset = crypto[['title', 'url', 'published_at', 'publisher', 'tags', 'html_content', 'clean_html_text']]
crypto_subset.head(1)


# In[73]:


huffpost_subset = huffpost[['headline', 'url', 'published_at', 'source_title', 'tags', 'html_content', 'clean_html_text']]
huffpost_subset.rename(columns={ "headline":"title","source_title":"publisher"}, inplace=True)
huffpost_subset.head(1)


# In[75]:


aljazeera_subset = aljazeera[['title', 'url', 'epoch_time', 'website', 'sub_category', 'html_content', 'clean_html_text']]
aljazeera_subset.rename(columns={ "website":"publisher","epoch_time":"published_at","sub_category":"tags"}, inplace=True)
aljazeera_subset.head(1)


# In[77]:


bbc_subset = bbc[['title', 'url', 'news_post_date', 'category', 'tags', 'html_content', 'clean_html_text']]
bbc_subset.rename(columns={ "news_post_date":"published_at","category":"publisher"}, inplace=True)
bbc_subset.head(1)


# In[80]:


japan_times_subset = japan_times[['headline','url', 'datePublished', 'siteName', 'keywords', 'html_content', 'clean_html_text']]
japan_times_subset.rename(columns={ "headline":"title","datePublished":"published_at","siteName":"publisher","keywords":"tags"}, inplace=True)
japan_times_subset.head(1)


# In[92]:


cnn_subset = cnn[['title', 'url', 'published_at', 'source', 'html_content', 'clean_html_text']]
cnn_subset.rename(columns={ "source":"publisher"}, inplace=True)
cnn_subset.insert(4, "tags"," ",True)
cnn_subset.head(1)


# In[93]:


all_news = pd.concat([cnbc_subset, huffpost_subset, aljazeera_subset, bbc_subset, japan_times_subset, cnn_subset])


# In[94]:


all_news


# In[95]:


all_news.to_csv('/Users/lama/Downloads/DAEN 690 Fall 2022/Project/all_news.csv', index=False)


# In[ ]:




