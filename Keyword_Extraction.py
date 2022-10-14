#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from bs4 import BeautifulSoup
import trafilatura
from htmldate import find_date
from urllib.request import urlopen


# In[7]:


all_news = pd.read_csv('D:/DAEN/DAEN 690/Datasets/Combined dataset/all_news.csv',sep=',',nrows=5)
all_news


# In[8]:


all_news = all_news[['url', 'title']]


# In[9]:


extracted_clean_text = []
extracted_published_date = []
extracted_title = []
for x in all_news.url:
            file = urlopen(x)
            parser = BeautifulSoup(file, 'html.parser')
            downloaded = trafilatura.fetch_url(x) 
            extracted_clean_text.append(trafilatura.extract(downloaded, include_comments=False, include_tables=False, no_fallback=True))
            extracted_published_date.append(find_date(x))
            extracted_title.append('|'.join(parser.title.string.split('|')[0:1]))           
            
all_news.insert(0,'extracted_clean_text', extracted_clean_text)          
all_news.insert(1,'extracted_title', extracted_title)
all_news.insert(2,'extracted_published_date', extracted_published_date)
all_news


# In[6]:


import spacy
extractedkeyw_per = []
extractedkeyw_org = []
extractedkeyw_pla = []
for i in all_news.extracted_clean_text:
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(i)
    ent_per = [ent for ent in doc.ents if (ent.label_ == "PERSON") ]
    extractedkeyw_per.append(ent_per)
    ent_org = [ent for ent in doc.ents if (ent.label_ == "ORG") ]
    extractedkeyw_org.append(ent_org)
    ent_pla = [ent for ent in doc.ents if (ent.label_ == "GPE") ]
    extractedkeyw_pla.append(ent_pla)
all_news.insert(3,'extractedkeyw_per', extractedkeyw_per)          
all_news.insert(4,'extractedkeyw_org', extractedkeyw_org)
all_news.insert(5,'extractedkeyw_pla', extractedkeyw_pla)
all_news
    


# In[5]:


import spacy
nlp = spacy.load("en_core_web_lg")


# In[ ]:


doc = nlp(all_news.extracted_clean_text[0])
for ent in doc.ents:
    print(ent.text+' -- '+ent.label_+' -- '+spacy.explain(ent.label_))
    


# In[ ]:


all_news


# In[ ]:


doc = nlp(all_news.extracted_clean_text[4])
print(doc.ents)


# In[ ]:


import spacy
import pytextrank
# example text
text = all_news.extracted_clean_text[4]
# load a spaCy model, depending on language, scale, etc.
nlp = spacy.load("en_core_web_sm")
# add PyTextRank to the spaCy pipeline
nlp.add_pipe("textrank")
doc = nlp(text)
# examine the top-ranked phrases in the document
for phrase in doc._.phrases[:15]:
     print(phrase.text) 
        
    
    


# In[ ]:


from collections import Counter

words = extractedkeyw_pla
counter = Counter(words[3])
for i in words[5:]: 
    counter.update(i)

counter.most_common()


# In[ ]:


import numpy as np

pd.value_counts(np.array(extractedkeyw_pla))


# In[ ]:


import collections
Output = collections.defaultdict(int)
 
# List initialization
Input = extractedkeyw_pla[0]
 
# Using iteration
for elem in Input:
      Output[elem[0]] += 1
     
# Printing output
print(Output)


# In[ ]:


pd.value_counts(np.hstack(extractedkeyw_pla))


# In[ ]:


pip install summa


# In[ ]:


from summa import keywords
TR_keywords = keywords.keywords(all_news.extracted_clean_text[4], scores=True)
print(TR_keywords[0:15])


# In[ ]:




