#!/usr/bin/env python
# coding: utf-8

# ## Tag Organizer Function

# In[766]:


# UDF to clean aand prapre 'extractedkeyw_per', 'extractedkeyw_org','extractedkeyw_pla' columns
pd.options.mode.chained_assignment = None
def tags_organizer(dataframe):
    for i in dataframe[['extractedkeyw_per', 'extractedkeyw_org','extractedkeyw_pla']]:
        dataframe[i] = dataframe[i].astype("string")
        dataframe[i] = dataframe[i].str.replace('[', '', regex=True)
        dataframe[i] = dataframe[i].str.replace(']', '', regex=True)
        dataframe[i] = dataframe[i].str.replace(' ', '', regex=True)
        dataframe[i+'1'] = dataframe[i].copy()
        dataframe[i+'1'] = dataframe[i].str.replace(',', '', regex=True)
        dataframe[i] = dataframe[i].str.split(",")
        dataframe.drop([i+'1'], axis=1, inplace=True)
    return dataframe


# ## Tag Counter Function

# In[795]:


pd.options.mode.chained_assignment = None
def tags_counter(dataframe):    
    Top_N = []
    for i in dataframe[['extractedkeyw_per', 'extractedkeyw_org','extractedkeyw_pla']]:
        a = dataframe[i].explode().unique()
        b = dataframe[i].explode().value_counts()[a].idxmax()
        Top_N.append(b)
    return Top_N


# # Preparing Text for Aljazeera

# ## Loading Aljazeera dataset with extracted candidate tags

# In[813]:


import pandas as pd
aljazeera = pd.read_csv('D:/DAEN/DAEN 690/Datasets/New Extraction_Checkpoints/aljazeera_extracted.csv')
aljazeera


# ### Import/Install Required Libraries and Packages

# In[534]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer


# ### EDA - Wordcloud from titles for possible descriptor tags

# In[535]:


#Creating the text variable

text2 = " ".join(title for title in df1.title)

# Creating word_cloud with text as argument in .generate() method

word_cloud2 = WordCloud(collocations = False, background_color = 'white').generate(text2)

# Display the generated Word Cloud

plt.imshow(word_cloud2, interpolation='bilinear')

plt.axis("off")

plt.show()


# ### Text Preprocessing,Cleaning and Tokenization

# In[536]:


# Convert to list
data1 = aljazeera.extracted_clean_text.values.tolist()
# Remove Emails
data1 = [re.sub('\S*@\S*\s?', '', sent) for sent in data1]

# Remove new line characters
data1 = [re.sub('\s+', ' ', sent) for sent in data1]

# Remove distracting single quotes
data1 = [re.sub("\'", "", sent) for sent in data1]

pprint(data1[1001:1002])


# In[537]:


#Function for tokenizing
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data1))

print(data_words[1001:1002])


# ### Prepare NLTK Stop words

# In[538]:


from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS

stop_words = stopwords.words('english')
stop_words.extend(list(STOP_WORDS))
stop_words.extend(['said','say'])


# In[539]:


print(stop_words)


# ### Visualizing Unigrams in text before and after removing stopwords

# In[540]:


import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[541]:


#The distribution of top unigrams before removing stop words

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(aljazeera['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df2.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 unigrams in text before removing stop words')


# In[542]:


#The distribution of top unigrams after removing stop words

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = frozenset(stop_words)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(aljazeera['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df3 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df3.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 unigrams in text after removing stop words')


# ### Identifying bigrams/trigrams and Lemmatization

# In[543]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[1001]]])


# In[544]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[545]:


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized1 = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized1[1001:1002])


# ### Visualizing Bigrams in text before and after removing stopwords

# In[546]:


#The distribution of top bigrams before removing stop words

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(aljazeera['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df4 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df4.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review before removing stop words')


# In[547]:


#The distribution of top bigrams after removing stop words

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(aljazeera['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df5 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df5.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review after removing stop words')


# ### Visualizing Trigrams in text before and after removing stopwords

# In[548]:


#The distribution of Top trigrams before removing stop words

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(aljazeera['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df6 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df6.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review before removing stop words')


# In[549]:


#The distribution of Top trigrams after removing stop words

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(aljazeera['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df7 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df7.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review after removing stop words')


# # Preparing Text of BBC

# ## Loading BBC dataset with extracted candidate tags

# In[945]:


import pandas as pd
bbc = pd.read_csv('D:/DAEN/DAEN 690/Datasets/New Extraction_Checkpoints/bbc_extracted.csv')
bbc


# ### Import/Install Required Libraries and Packages

# In[ ]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer


# ### EDA - Wordcloud from titles for possible descriptor tags

# In[ ]:


#Creating the text variable

text2 = " ".join(title for title in df1.title)

# Creating word_cloud with text as argument in .generate() method

word_cloud2 = WordCloud(collocations = False, background_color = 'white').generate(text2)

# Display the generated Word Cloud

plt.imshow(word_cloud2, interpolation='bilinear')

plt.axis("off")

plt.show()


# ### Text Preprocessing,Cleaning and Tokenization

# In[902]:


# Convert to list
data2 = bbc.extracted_clean_text.values.tolist()
# Remove Emails
data2 = [re.sub('\S*@\S*\s?', '', sent) for sent in data2]

# Remove new line characters
data2 = [re.sub('\s+', ' ', sent) for sent in data2]

# Remove distracting single quotes
data2 = [re.sub("\'", "", sent) for sent in data2]

pprint(data2[1001:1002])


# In[903]:


#Function for tokenizing
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data2))

print(data_words[1001:1002])


# ### Prepare NLTK Stop words

# In[904]:


from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS

stop_words = stopwords.words('english')
stop_words.extend(list(STOP_WORDS))
stop_words.extend(['said','say'])


# In[905]:


print(stop_words)


# ### Visualizing Unigrams in text before and after removing stopwords

# In[906]:


import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[907]:


#The distribution of top unigrams before removing stop words

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(bbc['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df2.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 unigrams in text before removing stop words')


# In[908]:


#The distribution of top unigrams after removing stop words

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = frozenset(stop_words)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(bbc['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df3 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df3.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 unigrams in text after removing stop words')


# ### Identifying bigrams/trigrams & lemmatization

# In[909]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[1001]]])


# In[910]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[911]:


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized2 = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized2[1001:1002])


# ### Visualizing Bigrams in text before and after removing stopwords

# In[ ]:


#The distribution of top bigrams before removing stop words

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(bbc['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df4 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df4.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review before removing stop words')


# In[ ]:


#The distribution of top bigrams after removing stop words

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(bbc['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df5 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df5.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review after removing stop words')


# ### Visualizing Trigrams in text before and after removing stopwords

# In[ ]:


#The distribution of Top trigrams before removing stop words

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(bbc['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df6 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df6.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review before removing stop words')


# In[ ]:


#The distribution of Top trigrams after removing stop words

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(bbc['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df7 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df7.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review after removing stop words')


# # Preparing Text of CNN

# ## Loading CNN dataset with extracted candidate tags

# In[861]:


import pandas as pd
cnn = pd.read_csv('D:/DAEN/DAEN 690/Datasets/New Extraction_Checkpoints/cnn_extracted.csv')
cnn


# ### Import/Install Required Libraries and Packages

# In[44]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer


# ### EDA - Wordcloud from titles for possible descriptor tags

# In[45]:


#Creating the text variable

text2 = " ".join(title for title in df1.title)

# Creating word_cloud with text as argument in .generate() method

word_cloud2 = WordCloud(collocations = False, background_color = 'white').generate(text2)

# Display the generated Word Cloud

plt.imshow(word_cloud2, interpolation='bilinear')

plt.axis("off")

plt.show()


# ### Text Preprocessing,Cleaning and Tokenization

# In[122]:


# Convert to list
data3 = cnn.extracted_clean_text.values.tolist()
# Remove Emails
data3 = [re.sub('\S*@\S*\s?', '', sent) for sent in data3]

# Remove new line characters
data3 = [re.sub('\s+', ' ', sent) for sent in data3]

# Remove distracting single quotes
data3 = [re.sub("\'", "", sent) for sent in data3]

pprint(data3[1001:1002])


# In[47]:


#Function for tokenizing
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data))

print(data_words[1001:1002])


# ### Prepare NLTK Stop words

# In[48]:


from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS

stop_words = stopwords.words('english')
stop_words.extend(list(STOP_WORDS))
stop_words.extend(['said','say'])


# In[49]:


print(stop_words)


# ### Visualizing Unigrams in text before and after removing stopwords

# In[50]:


import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[51]:


#The distribution of top unigrams before removing stop words

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(cnn['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df2.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 unigrams in text before removing stop words')


# In[52]:


#The distribution of top unigrams after removing stop words

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = frozenset(stop_words)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(cnn['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df3 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df3.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 unigrams in text after removing stop words')


# ### Identifying bigrams/trigrams

# In[53]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[1001]]])


# In[54]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[56]:


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized3 = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized3[1001:1002])


# ### Visualizing Bigrams in text before and after removing stopwords

# In[57]:


#The distribution of top bigrams before removing stop words

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(cnn['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df4 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df4.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review before removing stop words')


# In[58]:


#The distribution of top bigrams after removing stop words

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(cnn['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df5 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df5.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review after removing stop words')


# ### Visualizing Trigrams in text before and after removing stopwords

# In[59]:


#The distribution of Top trigrams before removing stop words

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(cnn['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df6 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df6.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review before removing stop words')


# In[60]:


#The distribution of Top trigrams after removing stop words

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(cnn['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df7 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df7.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review after removing stop words')


# In[ ]:





# # Create Master Dictionary and corpus for all 3 candidate datasets

# In[515]:


# Create Dictionary
id2word1 = corpora.Dictionary(data_lemmatized1)
print("\nThe dictionary now has: " + str(len(id2word1)) + " tokens")
id2word1.add_documents(data_lemmatized2)
print("\nThe dictionary now has: " + str(len(id2word1)) + " tokens")
id2word1.add_documents(data_lemmatized3)
print("\nThe dictionary now has: " + str(len(id2word1)) + " tokens")

corpus1 = [id2word1.doc2bow(text) for text in data_lemmatized1]
corpus2 = [id2word1.doc2bow(text) for text in data_lemmatized2]
corpus3 = [id2word1.doc2bow(text) for text in data_lemmatized3]


# In[912]:


corpus2 = [id2word1.doc2bow(text) for text in data_lemmatized2]


# # Topic Modeling for Aljazeera using Master Dictionary

# ### Build LDA model

# In[516]:



lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus1,
                                           id2word=id2word1,
                                           num_topics=8, 
                                           random_state=100,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='auto',
                                           per_word_topics=True)


# ### Measure Metrics of Model - Coherence/Perplexity

# In[517]:


# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus1))  # a measure of how good the model is. lower the better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized1, dictionary=id2word1, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)


# ### Visualization of Topics identified by Model

# In[518]:


# Visualize the topics
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus1, id2word1)
vis


# ### Building LDA-Mallet model for comparision

# In[519]:


import os

os.environ['MALLET_HOME'] = 'C:/Users/Madesh/mallet-2.0.8/mallet-2.0.8'

from gensim.models.wrappers import LdaMallet

mallet_path = 'C:/Users/Madesh/mallet-2.0.8/mallet-2.0.8/bin/mallet.bat'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus1, num_topics=8, id2word=id2word1)


# In[520]:


# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized1, dictionary=id2word1, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


# ### Find Optimal number of topics to feed the model to identify

# In[554]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus1, num_topics=num_topics, id2word=id2word1,random_seed = 96)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[555]:


model_list, coherence_values = compute_coherence_values(dictionary=id2word1, corpus=corpus1, texts=data_lemmatized1, start=2, limit=14, step=2)


# In[556]:


# Show graph
limit=14; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[563]:


# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[564]:


cv_values = []
for i,(m, cv) in enumerate(zip(x, coherence_values)):
    if m <=10:
        cv_values.append(cv)
index = cv_values.index(max(cv_values))
index
       


# #### selecting the most optimal and efficient number of topics <10 to make sure the articles are not overclassified

# In[565]:


# Select the model and print the topics
optimal_model = model_list[index]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# ### Convert Mallet model to LDA model to visualize the topics

# In[566]:


def convertldaGenToldaMallet(mallet_model):
    model_gensim = gensim.models.ldamodel.LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, eta=0,
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim

optimal_model = convertldaGenToldaMallet(optimal_model)


# In[567]:


#Visualizing Opitmal Model

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(optimal_model, corpus1, id2word1)
vis


# ### Dominant topic in each sentence

# In[826]:


def format_topics_sentences(ldamodel=optimal_model, corpus=corpus1, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus1, texts=data1)

# Format
df_dominant_topic_aljazeera = df_topic_sents_keywords.reset_index()
df_dominant_topic_aljazeera.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic_aljazeera.head(10)


# In[574]:


df_dominant_topic_aljazeera[(df_dominant_topic_aljazeera.Dominant_Topic == 1) & (df_dominant_topic_aljazeera.Topic_Perc_Contrib > 0.5)]


# ### Most representative document for each topic

# In[569]:


# Group top article under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet


# ### Visualizing Word Count and Importance of Topic Keywords

# In[570]:


import matplotlib.colors as mcolors
from collections import Counter
topics = optimal_model.show_topics(formatted=False)
data_flat = [w for w_list in data_lemmatized1 for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 5, figsize=(20,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()


# ### Topic labelling

# In[827]:


topic_labels = ["Economy","Accidents_Disaster","Israeli_Palestinian_Conflict","Politics","Crime","Education","War","Science","Legal","Government"]   


# In[828]:


from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = optimal_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 5, figsize=(20,20), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title(topic_labels[i], fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=5, hspace=5)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# ### Save LDA model
# 

# In[880]:


from gensim.test.utils import datapath
temp_file = datapath("model_1")
optimal_model.save(temp_file)


# ### Grouping Dataset as collections based on identified Topics

# In[829]:


df_dominant_topic_aljazeera['Dominant_Topic'] = df_dominant_topic_aljazeera['Dominant_Topic'].replace([0,1,2,3,4,5,6,7,8,9], topic_labels)


# In[830]:


topic_extracted = df_dominant_topic_aljazeera[['Dominant_Topic', 'Keywords']]


# In[814]:


tags_organizer(aljazeera)


# In[831]:


al_tagged = pd.concat([aljazeera, topic_extracted], axis=1)


# In[832]:


Aljazeera_Collections=[]
for i, x in al_tagged.groupby('Dominant_Topic'):
    globals()['Aljazeera_tagged' + "_" + str(i)] = x
    Aljazeera_Collections.append('Aljazeera_tagged_' + str(i))
Aljazeera_Collections


# In[858]:


Aljazeera_tagged_Economy


# In[847]:


print(tags_counter(Aljazeera_tagged_Accidents_Disaster))
print(Aljazeera_tagged_Accidents_Disaster.extracted_published_date.min())
print(Aljazeera_tagged_Accidents_Disaster.extracted_published_date.max())


# In[850]:


print(tags_counter(Aljazeera_tagged_Crime))
print(Aljazeera_tagged_Crime.extracted_published_date.min())
print(Aljazeera_tagged_Crime.extracted_published_date.max())


# In[851]:


print(tags_counter(Aljazeera_tagged_Economy))
print(Aljazeera_tagged_Economy.extracted_published_date.min())
print(Aljazeera_tagged_Economy.extracted_published_date.max())


# In[852]:


print(tags_counter(Aljazeera_tagged_Education))
print(Aljazeera_tagged_Education.extracted_published_date.min())
print(Aljazeera_tagged_Education.extracted_published_date.max())


# In[853]:


print(tags_counter(Aljazeera_tagged_Israeli_Palestinian_Conflict))
print(Aljazeera_tagged_Israeli_Palestinian_Conflict.extracted_published_date.min())
print(Aljazeera_tagged_Israeli_Palestinian_Conflict.extracted_published_date.max())


# In[854]:


print(tags_counter(Aljazeera_tagged_Legal))
print(Aljazeera_tagged_Legal.extracted_published_date.min())
print(Aljazeera_tagged_Legal.extracted_published_date.max())


# In[855]:


print(tags_counter(Aljazeera_tagged_Politics))
print(Aljazeera_tagged_Politics.extracted_published_date.min())
print(Aljazeera_tagged_Politics.extracted_published_date.max())


# In[856]:


print(tags_counter(Aljazeera_tagged_Science))
print(Aljazeera_tagged_Science.extracted_published_date.min())
print(Aljazeera_tagged_Science.extracted_published_date.max())


# In[857]:


print(tags_counter(Aljazeera_tagged_War))
print(Aljazeera_tagged_War.extracted_published_date.min())
print(Aljazeera_tagged_War.extracted_published_date.max())


# # Topic Modeling for BBC using Master Dictionary

# ### Load the Pre-Trained LDA Topic Model

# In[881]:


# Load LDA model
from gensim.test.utils import datapath
temp_file = datapath("model_1")
lda_model = gensim.models.ldamodel.LdaModel.load(temp_file)


# ### Load Master Dictionary

# In[883]:


from gensim.test.utils import datapath
temp_file = datapath("model_1.id2word")
dictionary = corpora.Dictionary.load(temp_file)


# ### Update Model's corpus

# In[913]:


lda_model.update(corpus2)


# ### Model Tuning using LDA Mallet

# In[914]:


import os

os.environ['MALLET_HOME'] = 'C:/Users/Madesh/mallet-2.0.8/mallet-2.0.8'

from gensim.models.wrappers import LdaMallet

mallet_path = 'C:/Users/Madesh/mallet-2.0.8/mallet-2.0.8/bin/mallet.bat'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus2, num_topics=index*2+2, id2word=dictionary)


# In[915]:


# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized2, dictionary=dictionary, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


# ### Find Optimal number of topics to feed the model to identify

# In[922]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus2, num_topics=num_topics, id2word=dictionary,random_seed=96)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[923]:


model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus2, texts=data_lemmatized2, start=2, limit=14, step=2)


# In[924]:


# Show graph
limit=14; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[925]:


# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[926]:


cv_values = []
for i,(m, cv) in enumerate(zip(x, coherence_values)):
    if m <=10:
        cv_values.append(cv)
index = cv_values.index(max(cv_values))
index
       


# #### selecting the most optimal and efficient number of topics <10 to make sure the articles are not overclassified

# In[927]:


# Select the model and print the topics
optimal_model = model_list[index]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# ### Convert Mallet model to LDA model to visualize the topics

# In[928]:


def convertldaGenToldaMallet(mallet_model):
    model_gensim = gensim.models.ldamodel.LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, eta=0,
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim

optimal_model = convertldaGenToldaMallet(optimal_model)


# In[929]:


#Visualizing Opitmal Model

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(optimal_model, corpus2, dictionary)
vis


# ### Dominant topic in each sentence

# In[932]:


def format_topics_sentences(ldamodel=optimal_model, corpus=corpus2, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus2, texts=data2)

# Format
df_dominant_topic_bbc = df_topic_sents_keywords.reset_index()
df_dominant_topic_bbc.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic_bbc.head(10)


# ### Most representative document for each topic

# In[934]:


# Group top article under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet


# In[938]:


df_dominant_topic_bbc[(df_dominant_topic_bbc.Dominant_Topic == 5) & (df_dominant_topic_bbc.Topic_Perc_Contrib > 0.40)]


# ### Visualizing Word Count and Importance of Topic Keywords

# In[931]:


import matplotlib.colors as mcolors
from collections import Counter
topics = optimal_model.show_topics(formatted=False)
data_flat = [w for w_list in data_lemmatized1 for w in w_list]
counter = Counter(data_flat)

out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 4, figsize=(20,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()


# ### Topic labelling

# In[939]:


topic_labels = ["Health","Economy","Legal","Government","Entertainment","World","Violence","Accident"]   


# In[940]:


from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = optimal_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 4, figsize=(20,15), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title(topic_labels[i], fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# ### save LDA model

# In[941]:


from gensim.test.utils import datapath
temp_file = datapath("model_1")
optimal_model.save(temp_file)


# In[ ]:





# ### Grouping Dataset as collections based on identified Topics

# In[942]:


df_dominant_topic_bbc['Dominant_Topic'] = df_dominant_topic_bbc['Dominant_Topic'].replace([0,1,2,3,4,5,6,7], topic_labels)


# In[943]:


topic_extracted = df_dominant_topic_bbc[['Dominant_Topic', 'Keywords']]


# In[946]:


tags_organizer(bbc)


# In[947]:


bbc_tagged = pd.concat([bbc, topic_extracted], axis=1)


# In[948]:


BBC_Collections=[]
for i, x in bbc_tagged.groupby('Dominant_Topic'):
    globals()['BBC_tagged' + "_" + str(i)] = x
    BBC_Collections.append('BBC_tagged_' + str(i))
BBC_Collections


# In[949]:


BBC_tagged_Economy


# In[950]:


print(tags_counter(BBC_tagged_Accident))
print(BBC_tagged_Accident.extracted_published_date.min())
print(BBC_tagged_Accident.extracted_published_date.max())


# In[951]:


print(tags_counter(BBC_tagged_Economy))
print(BBC_tagged_Economy.extracted_published_date.min())
print(BBC_tagged_Economy.extracted_published_date.max())


# In[952]:


print(tags_counter(BBC_tagged_Entertainment))
print(BBC_tagged_Entertainment.extracted_published_date.min())
print(BBC_tagged_Entertainment.extracted_published_date.max())


# In[953]:


print(tags_counter(BBC_tagged_Health))
print(BBC_tagged_Health.extracted_published_date.min())
print(BBC_tagged_Health.extracted_published_date.max())


# In[954]:


print(tags_counter(BBC_tagged_Legal))
print(BBC_tagged_Legal.extracted_published_date.min())
print(BBC_tagged_Legal.extracted_published_date.max())


# In[955]:


print(tags_counter(BBC_tagged_Violence))
print(BBC_tagged_Violence.extracted_published_date.min())
print(BBC_tagged_Violence.extracted_published_date.max())


# In[956]:


print(tags_counter(BBC_tagged_World))
print(BBC_tagged_World.extracted_published_date.min())
print(BBC_tagged_World.extracted_published_date.max())


# # Topic Modeling for CNN using Master Dictionary

# ### Load the Pre-Trained LDA Topic Model

# In[957]:


# Load LDA model
from gensim.test.utils import datapath
temp_file = datapath("model_1")
lda_model = gensim.models.ldamodel.LdaModel.load(temp_file)


# ### Load Master Dictionary

# In[958]:


from gensim.test.utils import datapath
temp_file = datapath("model_1.id2word")
dictionary = corpora.Dictionary.load(temp_file)


# ### Update Model's corpus

# In[959]:


lda_model.update(corpus3)


# ### Model Tuning using LDA Mallet

# In[960]:


import os

os.environ['MALLET_HOME'] = 'C:/Users/Madesh/mallet-2.0.8/mallet-2.0.8'

from gensim.models.wrappers import LdaMallet

mallet_path = 'C:/Users/Madesh/mallet-2.0.8/mallet-2.0.8/bin/mallet.bat'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus3, num_topics=index*2+2, id2word=dictionary)


# In[961]:


# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized3, dictionary=dictionary, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


# ### Find Optimal number of topics to feed the model to identify

# In[983]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus3, num_topics=num_topics, id2word=dictionary,random_seed=100)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[985]:


model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus3, texts=data_lemmatized3, start=2, limit=14, step=2)


# In[986]:


# Show graph
limit=14; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[987]:


# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[988]:


cv_values = []
for i,(m, cv) in enumerate(zip(x, coherence_values)):
    if m <=10:
        cv_values.append(cv)
index = cv_values.index(max(cv_values))
index
       


# #### selecting the most optimal and efficient number of topics <10 to make sure the articles are not overclassified

# In[989]:


# Select the model and print the topics
optimal_model = model_list[index]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# ### Convert Mallet model to LDA model to visualize the topics

# In[990]:


def convertldaGenToldaMallet(mallet_model):
    model_gensim = gensim.models.ldamodel.LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, eta=0,
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim

optimal_model = convertldaGenToldaMallet(optimal_model)


# In[991]:


#Visualizing Opitmal Model

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(optimal_model, corpus3, dictionary)
vis


# ### Dominant topic in each sentence

# In[1020]:


def format_topics_sentences(ldamodel=optimal_model, corpus=corpus3, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus3, texts=data3)

# Format
df_dominant_topic_cnn = df_topic_sents_keywords.reset_index()
df_dominant_topic_cnn.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic_cnn.head(10)


# ### Most representative document for each topic

# In[993]:


# Group top article under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet


# In[1000]:


df_dominant_topic_cnn[(df_dominant_topic_cnn.Dominant_Topic == 1) & (df_dominant_topic.Topic_Perc_Contrib > 0.4)]


# ### Visualizing Word Count and Importance of Topic Keywords

# In[994]:


import matplotlib.colors as mcolors
from collections import Counter
topics = optimal_model.show_topics(formatted=False)


out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 5, figsize=(20,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()


# ### Topic Labelling

# In[1021]:


topic_labels = ["Politics","Local","Sports_Racing","Legal","Government","Crime","Sports_Tennis","Sports_Football","Sports_Soccer","Sports_Golf"]   


# In[1003]:


from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = optimal_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 5, figsize=(20,15), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title(topic_labels[i], fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# ### save LDA model

# In[1004]:


from gensim.test.utils import datapath
temp_file = datapath("model_1")
optimal_model.save(temp_file)


# ### Grouping Dataset as collections based on identified Topics

# In[1023]:


df_dominant_topic_cnn


# In[1022]:


df_dominant_topic_cnn['Dominant_Topic'] = df_dominant_topic_cnn['Dominant_Topic'].replace([0,1,2,3,4,5,6,7,8,9], topic_labels)


# In[1024]:


topic_extracted = df_dominant_topic_cnn[['Dominant_Topic', 'Keywords']]


# In[1015]:


tags_organizer(cnn)


# In[1025]:


cnn_tagged = pd.concat([cnn, topic_extracted], axis=1)


# In[1026]:


CNN_Collections=[]
for i, x in cnn_tagged.groupby('Dominant_Topic'):
    globals()['CNN_tagged_' + str(i)] = x
    CNN_Collections.append('CNN_tagged_' + str(i))
CNN_Collections


# In[1028]:


CNN_tagged_Sports_Racing


# In[1029]:


print(tags_counter(CNN_tagged_Crime))
print(CNN_tagged_Crime.extracted_published_date.min())
print(CNN_tagged_Crime.extracted_published_date.max())


# In[1030]:


print(tags_counter(CNN_tagged_Government))
print(CNN_tagged_Government.extracted_published_date.min())
print(CNN_tagged_Government.extracted_published_date.max())


# In[1031]:


print(tags_counter(CNN_tagged_Legal))
print(CNN_tagged_Legal.extracted_published_date.min())
print(CNN_tagged_Legal.extracted_published_date.max())


# In[1032]:


print(tags_counter(CNN_tagged_Local))
print(CNN_tagged_Local.extracted_published_date.min())
print(CNN_tagged_Local.extracted_published_date.max())


# In[1033]:


print(tags_counter(CNN_tagged_Politics))
print(CNN_tagged_Politics.extracted_published_date.min())
print(CNN_tagged_Politics.extracted_published_date.max())


# In[1034]:


print(tags_counter(CNN_tagged_Sports_Football))
print(CNN_tagged_Sports_Football.extracted_published_date.min())
print(CNN_tagged_Sports_Football.extracted_published_date.max())


# In[1035]:


print(tags_counter(CNN_tagged_Sports_Golf))
print(CNN_tagged_Sports_Golf.extracted_published_date.min())
print(CNN_tagged_Sports_Golf.extracted_published_date.max())


# In[1036]:


print(tags_counter(CNN_tagged_Sports_Racing))
print(CNN_tagged_Sports_Racing.extracted_published_date.min())
print(CNN_tagged_Sports_Racing.extracted_published_date.max())


# In[1037]:


print(tags_counter(CNN_tagged_Sports_Soccer))
print(CNN_tagged_Sports_Soccer.extracted_published_date.min())
print(CNN_tagged_Sports_Soccer.extracted_published_date.max())


# In[1038]:


print(tags_counter(CNN_tagged_Sports_Tennis))
print(CNN_tagged_Sports_Tennis.extracted_published_date.min())
print(CNN_tagged_Sports_Tennis.extracted_published_date.max())


# # Preparation of corpus and Topic Modeling for Japan Times

# ## Loading Japan Times dataset with extracted candidate tags

# In[1100]:


import pandas as pd
jt = pd.read_csv('D:/DAEN/DAEN 690/Datasets/New Extraction_Checkpoints/japan_times_extracted.csv')
jt


# ### Import/Install Required Libraries and Packages

# In[44]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer


# ### EDA - Wordcloud from titles for possible descriptor tags

# In[399]:


#Creating the text variable

text2 = " ".join(title for title in jt.extracted_title)

# Creating word_cloud with text as argument in .generate() method

word_cloud2 = WordCloud(collocations = False, background_color = 'white').generate(text2)

# Display the generated Word Cloud

plt.imshow(word_cloud2, interpolation='bilinear')

plt.axis("off")

plt.show()


# ### Text Preprocessing,Cleaning and Tokenization

# In[400]:


# Convert to list
data4 = jt.extracted_clean_text.values.tolist()
# Remove Emails
data4 = [re.sub('\S*@\S*\s?', '', sent) for sent in data4]

# Remove new line characters
data4 = [re.sub('\s+', ' ', sent) for sent in data4]

# Remove distracting single quotes
data4 = [re.sub("\'", "", sent) for sent in data4]

pprint(data4[0:1])


# In[401]:


#Function for tokenizing
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data4))

print(data_words[0:1])


# ### Prepare NLTK Stop words

# In[402]:


from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS

stop_words = stopwords.words('english')
stop_words.extend(list(STOP_WORDS))
stop_words.extend(['said','say'])


# In[403]:


print(stop_words)


# ### Visualizing Unigrams in text before and after removing stopwords

# In[50]:


import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[404]:


#The distribution of top unigrams before removing stop words

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(jt['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df2.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 unigrams in text before removing stop words')


# In[405]:


#The distribution of top unigrams after removing stop words

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = frozenset(stop_words)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(jt['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df3 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df3.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 unigrams in text after removing stop words')


# ### Identifying bigrams/trigrams

# In[407]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])


# In[408]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[410]:


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized4 = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized4[0:1])


# ### Visualizing Bigrams in text before and after removing stopwords

# In[411]:


#The distribution of top bigrams before removing stop words

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(jt['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df4 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df4.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review before removing stop words')


# In[412]:


#The distribution of top bigrams after removing stop words

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(jt['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df5 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df5.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review after removing stop words')


# ### Visualizing Trigrams in text before and after removing stopwords

# In[413]:


#The distribution of Top trigrams before removing stop words

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(jt['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df6 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df6.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review before removing stop words')


# In[414]:


#The distribution of Top trigrams after removing stop words

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(jt['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df7 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df7.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review after removing stop words')


# ### Load Dictionary and create corpus for CNN text

# In[1039]:


from gensim.test.utils import datapath
temp_file = datapath("model_1.id2word")
dictionary = corpora.Dictionary.load(temp_file)


# In[1040]:


corpus4 = [dictionary.doc2bow(text) for text in data_lemmatized4]


# ### Load the Pre-Trained LDA Topic Model

# In[1041]:


# Build LDA model
from gensim.test.utils import datapath
temp_file = datapath("model_1")
lda_model = gensim.models.ldamodel.LdaModel.load(temp_file)


# ### Update Model's corpus

# In[1042]:


lda_model.update(corpus4)


# ### Model Tuning using LDA Mallet

# In[1043]:


import os

os.environ['MALLET_HOME'] = 'C:/Users/Madesh/mallet-2.0.8/mallet-2.0.8'

from gensim.models.wrappers import LdaMallet

mallet_path = 'C:/Users/Madesh/mallet-2.0.8/mallet-2.0.8/bin/mallet.bat'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus4, num_topics=index*2+2, id2word=dictionary)


# In[1044]:


# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized4, dictionary=dictionary, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


# ### Find Optimal number of topics to feed the model to identify

# In[1068]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus4, num_topics=num_topics, id2word=dictionary,random_seed=96)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[1069]:


model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus4, texts=data_lemmatized4, start=2, limit=14, step=2)


# In[1070]:


# Show graph
limit=14; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[1071]:


# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[1072]:


cv_values = []
for i,(m, cv) in enumerate(zip(x, coherence_values)):
    if m <=10:
        cv_values.append(cv)
index = cv_values.index(max(cv_values))
index
       


# #### selecting the most optimal and efficient number of topics <10 to make sure the articles are not overclassified

# In[1073]:


# Select the model and print the topics
optimal_model = model_list[index]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# ### Convert Mallet model to LDA model to visualize the topics

# In[1074]:


def convertldaGenToldaMallet(mallet_model):
    model_gensim = gensim.models.ldamodel.LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, eta=0,
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim

optimal_model = convertldaGenToldaMallet(optimal_model)


# In[1075]:


#Visualizing Opitmal Model

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(optimal_model, corpus4, dictionary)
vis


# ### Dominant topic in each sentence

# In[1076]:


def format_topics_sentences(ldamodel=optimal_model, corpus=corpus4, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus4, texts=data4)

# Format
df_dominant_topic_jt = df_topic_sents_keywords.reset_index()
df_dominant_topic_jt.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic_jt.head(10)


# ### Most representative document for each topic

# In[1077]:


# Group top article under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet


# In[1085]:


df_dominant_topic_jt[(df_dominant_topic_jt.Dominant_Topic == 0) & (df_dominant_topic_jt.Topic_Perc_Contrib > 0.60)]


# ### Visualizing Word Count and Importance of Topic Keywords

# In[1135]:


import matplotlib.colors as mcolors
from collections import Counter
topics = optimal_model.show_topics(formatted=False)


out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(1, 2, figsize=(10,5), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()


# ### Topic Labelling

# In[1086]:


topic_labels = ["Government","Business"]   


# In[1087]:


from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = optimal_model.show_topics(formatted=False)

fig, axes = plt.subplots(1, 2, figsize=(20,15), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title(topic_labels[i], fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# ### Save LDA model

# In[1089]:


from gensim.test.utils import datapath
temp_file = datapath("model_1")
optimal_model.save(temp_file)


# ### Grouping Dataset as collections based on identified Topics

# In[1090]:


df_dominant_topic_jt


# In[1091]:


df_dominant_topic_jt['Dominant_Topic'] = df_dominant_topic_jt['Dominant_Topic'].replace([0,1], topic_labels)


# In[1092]:


topic_extracted = df_dominant_topic_jt[['Dominant_Topic', 'Keywords']]


# In[1101]:


tags_organizer(jt)


# In[1102]:


jt_tagged = pd.concat([jt, topic_extracted], axis=1)


# In[1104]:


JapanTimes_Collections=[]
for i, x in jt_tagged.groupby('Dominant_Topic'):
    globals()['JT_tagged_' + str(i)] = x
    JapanTimes_Collections.append('JT_tagged_' + str(i))
JapanTimes_Collections


# In[1105]:


JT_tagged_Business


# In[1106]:


print(tags_counter(JT_tagged_Business))
print(JT_tagged_Business.extracted_published_date.min())
print(JT_tagged_Business.extracted_published_date.max())


# In[1107]:


print(tags_counter(JT_tagged_Government))
print(JT_tagged_Government.extracted_published_date.min())
print(JT_tagged_Government.extracted_published_date.max())


# # Preparation of corpus and Topic Modeling for CNBC

# ## Loading CNBC dataset with extracted candidate tags

# In[480]:


import pandas as pd
cnbc = pd.read_csv('D:/DAEN/DAEN 690/Datasets/New Extraction_Checkpoints/cnbc_extracted.csv')
cnbc


# ### Import/Install Required Libraries and Packages

# In[481]:


import re
import numpy as np
import pandas as pd
from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel

# spacy for lemmatization
import spacy

# Plotting tools
import pyLDAvis
import gensim  # don't skip this
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer


# ### EDA - Wordcloud from titles for possible descriptor tags

# In[482]:


#Creating the text variable

text2 = " ".join(title for title in cnbc.extracted_title)

# Creating word_cloud with text as argument in .generate() method

word_cloud2 = WordCloud(collocations = False, background_color = 'white').generate(text2)

# Display the generated Word Cloud

plt.imshow(word_cloud2, interpolation='bilinear')

plt.axis("off")

plt.show()


# ### Text Preprocessing,Cleaning and Tokenization

# In[484]:


# Convert to list
data5 = cnbc.extracted_clean_text.values.tolist()
# Remove Emails
data5 = [re.sub('\S*@\S*\s?', '', sent) for sent in data5]

# Remove new line characters
data5 = [re.sub('\s+', ' ', sent) for sent in data5]

# Remove distracting single quotes
data5 = [re.sub("\'", "", sent) for sent in data5]

pprint(data5[0:1])


# In[485]:


#Function for tokenizing
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations

data_words = list(sent_to_words(data5))

print(data_words[0:1])


# ### Prepare NLTK Stop words

# In[486]:


from nltk.corpus import stopwords
from spacy.lang.en.stop_words import STOP_WORDS

stop_words = stopwords.words('english')
stop_words.extend(list(STOP_WORDS))
stop_words.extend(['said','say'])


# In[487]:


print(stop_words)


# ### Visualizing Unigrams in text before and after removing stopwords

# In[488]:


import cufflinks as cf
cf.go_offline()
cf.set_config_file(offline=False, world_readable=True)


# In[489]:


#The distribution of top unigrams before removing stop words

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer().fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(cnbc['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df2 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df2.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 unigrams in text before removing stop words')


# In[490]:


#The distribution of top unigrams after removing stop words

def get_top_n_words(corpus, n=None):
    vec = CountVectorizer(stop_words = frozenset(stop_words)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_words(cnbc['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df3 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df3.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 unigrams in text after removing stop words')


# ### Identifying bigrams/trigrams & Lemmatization

# In[491]:


# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)  

# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# See trigram example
print(trigram_mod[bigram_mod[data_words[0]]])


# In[492]:


# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]

def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]

def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]

def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out


# In[503]:


# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)

# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)

# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
# python3 -m spacy download en
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized5 = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

print(data_lemmatized5[0:1])


# ### Visualizing Bigrams in text before and after removing stopwords

# In[494]:


#The distribution of top bigrams before removing stop words

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(cnbc['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df4 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df4.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review before removing stop words')


# In[495]:


#The distribution of top bigrams after removing stop words

def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_bigram(cnbc['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df5 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df5.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 bigrams in review after removing stop words')


# ### Visualizing Trigrams in text before and after removing stopwords

# In[496]:


#The distribution of Top trigrams before removing stop words

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(cnbc['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df6 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df6.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review before removing stop words')


# In[497]:


#The distribution of Top trigrams after removing stop words

def get_top_n_trigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(3, 3), stop_words=stop_words).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
common_words = get_top_n_trigram(cnbc['extracted_clean_text'], 20)
for word, freq in common_words:
    print(word, freq)
df7 = pd.DataFrame(common_words, columns = ['extracted_clean_text' , 'count'])
df7.groupby('extracted_clean_text').sum()['count'].sort_values(ascending=False).iplot(
    kind='bar', yTitle='Count', linecolor='black', title='Top 20 trigrams in review after removing stop words')


# ### Load Dictionary and create corpus for CNBC text

# In[498]:


from gensim.test.utils import datapath
temp_file = datapath("model_1.id2word")
dictionary = corpora.Dictionary.load(temp_file)


# In[506]:


corpus5 = [dictionary.doc2bow(text) for text in data_lemmatized5]


# ### Load the Pre-Trained LDA Topic Model

# In[500]:


# Build LDA model
from gensim.test.utils import datapath
temp_file = datapath("model_1")
lda_model = gensim.models.ldamodel.LdaModel.load(temp_file)


# ### Update Model's corpus

# In[501]:


lda_model.update(corpus5)


# In[418]:





# ### Model Tuning using LDA Mallet

# In[1108]:


import os

os.environ['MALLET_HOME'] = 'C:/Users/Madesh/mallet-2.0.8/mallet-2.0.8'

from gensim.models.wrappers import LdaMallet

mallet_path = 'C:/Users/Madesh/mallet-2.0.8/mallet-2.0.8/bin/mallet.bat'
ldamallet = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus5, num_topics=index*2+2, id2word=dictionary)


# In[1109]:


# Show Topics
pprint(ldamallet.show_topics(formatted=False))

# Compute Coherence Score
coherence_model_ldamallet = CoherenceModel(model=ldamallet, texts=data_lemmatized5, dictionary=dictionary, coherence='c_v')
coherence_ldamallet = coherence_model_ldamallet.get_coherence()
print('\nCoherence Score: ', coherence_ldamallet)


# ### Find Optimal number of topics to feed the model to identify

# In[1110]:


def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=2):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_values : Coherence values corresponding to the LDA model with respective number of topics
    """
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = gensim.models.wrappers.LdaMallet(mallet_path, corpus=corpus5, num_topics=num_topics, id2word=dictionary,random_seed=0)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())

    return model_list, coherence_values


# In[1111]:


model_list, coherence_values = compute_coherence_values(dictionary=dictionary, corpus=corpus5, texts=data_lemmatized5, start=2, limit=14, step=2)


# In[1112]:


# Show graph
limit=14; start=2; step=2;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()


# In[1113]:


# Print the coherence scores
for m, cv in zip(x, coherence_values):
    print("Num Topics =", m, " has Coherence Value of", round(cv, 4))


# In[1114]:


cv_values = []
for i,(m, cv) in enumerate(zip(x, coherence_values)):
    if m <=10:
        cv_values.append(cv)
index = cv_values.index(max(cv_values))
index
       


# #### selecting the most optimal and efficient number of topics <10 to make sure the articles are not overclassified

# In[1115]:


# Select the model and print the topics
optimal_model = model_list[index]
model_topics = optimal_model.show_topics(formatted=False)
pprint(optimal_model.print_topics(num_words=10))


# ### Convert Mallet model to LDA model to visualize the topics

# In[1116]:


def convertldaGenToldaMallet(mallet_model):
    model_gensim = gensim.models.ldamodel.LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, eta=0,
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim

optimal_model = convertldaGenToldaMallet(optimal_model)


# In[1117]:


#Visualizing Opitmal Model

pyLDAvis.enable_notebook()
vis = pyLDAvis.gensim_models.prepare(optimal_model, corpus5, dictionary)
vis


# ### Dominant topic in each sentence

# In[1118]:


def format_topics_sentences(ldamodel=optimal_model, corpus=corpus5, texts=data):
    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return(sent_topics_df)


df_topic_sents_keywords = format_topics_sentences(ldamodel=optimal_model, corpus=corpus5, texts=data5)

# Format
df_dominant_topic_cnbc = df_topic_sents_keywords.reset_index()
df_dominant_topic_cnbc.columns = ['Document_No', 'Dominant_Topic', 'Topic_Perc_Contrib', 'Keywords', 'Text']

# Show
df_dominant_topic_cnbc.head(10)


# ### Most representative document for each topic

# In[1119]:


# Group top article under each topic
sent_topics_sorteddf_mallet = pd.DataFrame()

sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')

for i, grp in sent_topics_outdf_grpd:
    sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet, 
                                             grp.sort_values(['Perc_Contribution'], ascending=[0]).head(1)], 
                                            axis=0)

# Reset Index    
sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

# Format
sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

# Show
sent_topics_sorteddf_mallet


# In[1132]:


df_dominant_topic_cnbc[(df_dominant_topic_cnbc.Dominant_Topic == 7) & (df_dominant_topic_cnbc.Topic_Perc_Contrib > 0.30)]


# ### Visualizing Word Count and Importance of Topic Keywords

# In[1122]:


import matplotlib.colors as mcolors
from collections import Counter
topics = optimal_model.show_topics(formatted=False)


out = []
for i, topic in topics:
    for word, weight in topic:
        out.append([word, i , weight, counter[word]])

df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

# Plot Word Count and Weights of Topic Keywords
fig, axes = plt.subplots(2, 5, figsize=(20,10), sharey=True, dpi=160)
cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
for i, ax in enumerate(axes.flatten()):
    ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
    ax_twin = ax.twinx()
    ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
    ax.set_ylabel('Word Count', color=cols[i])
    ax_twin.set_ylim(0, 0.030); ax.set_ylim(0, 3500)
    ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
    ax.tick_params(axis='y', left=False)
    ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
    ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

fig.tight_layout(w_pad=2)    
fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
plt.show()


# ### Topic Labelling for CNBC dataset

# In[1133]:


topic_labels = ["Market_Trade","Market_Business","Market_Jobs","Market_Stocks_Shares","World","Government","Market_Investments","Sports_Entertainment","Market_Economy","Politics"]   


# In[1134]:


from matplotlib import pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import matplotlib.colors as mcolors

cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]  # more colors: 'mcolors.XKCD_COLORS'

cloud = WordCloud(stopwords=stop_words,
                  background_color='white',
                  width=2500,
                  height=1800,
                  max_words=10,
                  colormap='tab10',
                  color_func=lambda *args, **kwargs: cols[i],
                  prefer_horizontal=1.0)

topics = optimal_model.show_topics(formatted=False)

fig, axes = plt.subplots(2, 5, figsize=(20,15), sharex=True, sharey=True)

for i, ax in enumerate(axes.flatten()):
    fig.add_subplot(ax)
    topic_words = dict(topics[i][1])
    cloud.generate_from_frequencies(topic_words, max_font_size=300)
    plt.gca().imshow(cloud)
    plt.gca().set_title(topic_labels[i], fontdict=dict(size=16))
    plt.gca().axis('off')


plt.subplots_adjust(wspace=0, hspace=0)
plt.axis('off')
plt.margins(x=0, y=0)
plt.tight_layout()
plt.show()


# In[1136]:


#save LDA model
from gensim.test.utils import datapath
temp_file = datapath("model_1")
optimal_model.save(temp_file)


# ### Grouping Dataset as collections based on identified Topics

# In[1138]:


df_dominant_topic_cnbc


# In[1139]:


df_dominant_topic_cnbc['Dominant_Topic'] = df_dominant_topic_cnbc['Dominant_Topic'].replace([0,1,2,3,4,5,6,7,8,9], topic_labels)


# In[1140]:


topic_extracted = df_dominant_topic_cnbc[['Dominant_Topic', 'Keywords']]


# In[1141]:


tags_organizer(cnbc)


# In[1142]:


CNBC_tagged = pd.concat([cnbc, topic_extracted], axis=1)


# In[1144]:


CNBC_Collections=[]
for i, x in CNBC_tagged.groupby('Dominant_Topic'):
    globals()['CNBC_tagged_' + str(i)] = x
    CNBC_Collections.append('CNBC_tagged_' + str(i))
CNBC_Collections


# In[1145]:


CNBC_tagged_Market_Business


# In[1146]:


print(tags_counter(CNBC_tagged_Government))
print(CNBC_tagged_Government.extracted_published_date.min())
print(CNBC_tagged_Government.extracted_published_date.max())


# In[1147]:


print(tags_counter(CNBC_tagged_Market_Business))
print(CNBC_tagged_Market_Business.extracted_published_date.min())
print(CNBC_tagged_Market_Business.extracted_published_date.max())


# In[1148]:


print(tags_counter(CNBC_tagged_Market_Economy))
print(CNBC_tagged_Market_Economy.extracted_published_date.min())
print(CNBC_tagged_Market_Economy.extracted_published_date.max())


# In[1149]:


print(tags_counter(CNBC_tagged_Market_Investments))
print(CNBC_tagged_Market_Investments.extracted_published_date.min())
print(CNBC_tagged_Market_Investments.extracted_published_date.max())


# In[1150]:


print(tags_counter(CNBC_tagged_Market_Jobs))
print(CNBC_tagged_Market_Jobs.extracted_published_date.min())
print(CNBC_tagged_Market_Jobs.extracted_published_date.max())


# In[1151]:


print(tags_counter(CNBC_tagged_Market_Stocks_Shares))
print(CNBC_tagged_Market_Stocks_Shares.extracted_published_date.min())
print(CNBC_tagged_Market_Stocks_Shares.extracted_published_date.max())


# In[1152]:


print(tags_counter(CNBC_tagged_Market_Trade))
print(CNBC_tagged_Market_Trade.extracted_published_date.min())
print(CNBC_tagged_Market_Trade.extracted_published_date.max())


# In[1153]:


print(tags_counter(CNBC_tagged_Politics))
print(CNBC_tagged_Politics.extracted_published_date.min())
print(CNBC_tagged_Politics.extracted_published_date.max())


# In[1154]:


print(tags_counter(CNBC_tagged_Sports_Entertainment))
print(CNBC_tagged_Sports_Entertainment.extracted_published_date.min())
print(CNBC_tagged_Sports_Entertainment.extracted_published_date.max())


# In[1155]:


print(tags_counter(CNBC_tagged_World))
print(CNBC_tagged_World.extracted_published_date.min())
print(CNBC_tagged_World.extracted_published_date.max())


# In[ ]:




