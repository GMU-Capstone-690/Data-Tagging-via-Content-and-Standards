

# # Text and Keywords Extraction from HTML Pages



# importing required libraries
import pandas as pd
from bs4 import BeautifulSoup
import trafilatura
from htmldate import find_date
import urllib
from urllib.request import urlopen
import spacy
import os
# setting directory 
os.getcwd()


# ## Text Extracting Function



# UDF to extract clean text, published date and title from HTML script using URLs
def text_extract(dataframe):
    extracted_clean_text = []
    extracted_published_date = []
    extracted_title = []
    for url in dataframe.url:
            file = urlopen(url)
            parser = BeautifulSoup(file, 'html.parser')
            downloaded = trafilatura.fetch_url(url) 
            extracted_clean_text.append(trafilatura.extract(downloaded, include_comments=False, include_tables=False, no_fallback=True))
            extracted_published_date.append(find_date(url))
            extracted_title.append('|'.join(parser.title.string.split('|')[0:1]))
    return extracted_clean_text, extracted_published_date, extracted_title


# ## Tags Extracting Function



# UDF to extract person names, organization names and places names from HTML script using URLs
def tags_extract(dataframe):
    extractedkeyw_per = []
    extractedkeyw_org = []
    extractedkeyw_pla = []
    for i in dataframe.extracted_clean_text:
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(i)
            ent_per = [ent for ent in doc.ents if (ent.label_ == "PERSON") ]
            extractedkeyw_per.append(ent_per)
            ent_org = [ent for ent in doc.ents if (ent.label_ == "ORG") ]
            extractedkeyw_org.append(ent_org)
            ent_pla = [ent for ent in doc.ents if (ent.label_ == "GPE") ]
            extractedkeyw_pla.append(ent_pla)
    return extractedkeyw_per, extractedkeyw_org, extractedkeyw_pla
            


# ## Tags Counter Function



# UDF to count number of repeats for person names, organization names and places
pd.options.mode.chained_assignment = None
def tags_count(dataframe):
    for i in dataframe[['extractedkeyw_per', 'extractedkeyw_org','extractedkeyw_pla']]:
        dataframe[i] = dataframe[i].astype("string")
        dataframe[i] = dataframe[i].str.replace('[', '', regex=True)
        dataframe[i] = dataframe[i].str.replace(']', '', regex=True)
        dataframe[i] = dataframe[i].str.replace(' ', '', regex=True)
        dataframe[i+'1'] = dataframe[i].copy()
        dataframe[i+'1'] = dataframe[i].str.replace(',', '', regex=True)
        dataframe[i] = dataframe[i].str.split(",")
        dataframe[i] = dataframe[[i, i+'1']].apply(lambda x: {word:x[i+'1'].count(word)
                                          for word in x[i]}, axis=1) 
    return dataframe.drop([i+'1'], axis=1, inplace=True)
      

### Processing for datasets using cleaned data
# ## Aljazeera News dataset




# reading aljazeera news CSV file from a local location
aljazeera = pd.read_csv("aljazeera_cleaned.csv") 
aljazeera.head(3)





# extracting  clean text, published date and title from HTML script using aljazeera articles URLs  
extracted_clean_text, extracted_published_date, extracted_title = text_extract(aljazeera) 
# inserting extracted clean text, published date and title into aljazeera dataset as new columns 
aljazeera.insert(0,'extracted_clean_text', extracted_clean_text)          
aljazeera.insert(1,'extracted_title', extracted_title)
aljazeera.insert(2,'extracted_published_date', extracted_published_date)
aljazeera.head(3)





# extracting  person names, organization names and places names from HTML script using aljazeera articles URLs
extractedkeyw_per, extractedkeyw_org, extractedkeyw_pla = tags_extract(aljazeera)
# inserting extracted person names, organization names and places names into aljazeera dataset as new columns 
aljazeera.insert(3,'extractedkeyw_per', extractedkeyw_per)          
aljazeera.insert(4,'extractedkeyw_org', extractedkeyw_org)
aljazeera.insert(5,'extractedkeyw_pla', extractedkeyw_pla)
aljazeera.head(3)





# counting number of repeats for person names, organization names and places for aljazeera dataset
tags_count(aljazeera)
aljazeera.head(3)





# saving the dataset to CSV file
aljazeera.to_csv('aljazeera_extracted.csv', index=False)


# ## BBC News dataset  




# reading BBC news CSV file from a local location
bbc = pd.read_csv("bbc_cleaned.csv") 
bbc.head(3)





# extracting  clean text, published date and title from HTML script using BBC articles URLs
extracted_clean_text, extracted_published_date, extracted_title = text_extract(bbc)   
# inserting extracted clean text, published date and title into BBC dataset as new columns 
bbc.insert(0,'extracted_clean_text', extracted_clean_text)          
bbc.insert(1,'extracted_title', extracted_title)
bbc.insert(2,'extracted_published_date', extracted_published_date)
bbc.head(3)





# extracting  person names, organization names and places names from HTML script using BBC articles URLs
extractedkeyw_per, extractedkeyw_org, extractedkeyw_pla = tags_extract(bbc)
# inserting extracted person names, organization names and places names into BBC dataset as new columns 
bbc.insert(3,'extractedkeyw_per', extractedkeyw_per)          
bbc.insert(4,'extractedkeyw_org', extractedkeyw_org)
bbc.insert(5,'extractedkeyw_pla', extractedkeyw_pla)
bbc.head(3)





# counting number of repeats for person names, organization names and places for BBC dataset
tags_count(bbc)
bbc.head(3)





# saving the dataset to CSV file
bbc.to_csv('bbc_extracted.csv', index=False)


# ## CNBC News dataset




# reading CNBC news CSV file from a local location
cnbc = pd.read_csv("cnbc_cleaned.csv") 
cnbc.head(3)





# extracting  clean text, published date and title from HTML script using CNBC articles URLs  
extracted_clean_text, extracted_published_date, extracted_title = text_extract(cnbc) 
# inserting extracted clean text, published date and title into cnbc dataset as new columns 
cnbc.insert(0,'extracted_clean_text', extracted_clean_text)          
cnbc.insert(1,'extracted_title', extracted_title)
cnbc.insert(2,'extracted_published_date', extracted_published_date)
cnbc.head(3)





# extracting  person names, organization names and places names from HTML script using CNBC articles URLs
extractedkeyw_per, extractedkeyw_org, extractedkeyw_pla = tags_extract(cnbc)
# inserting extracted person names, organization names and places names into CNBC dataset as new columns 
cnbc.insert(3,'extractedkeyw_per', extractedkeyw_per)          
cnbc.insert(4,'extractedkeyw_org', extractedkeyw_org)
cnbc.insert(5,'extractedkeyw_pla', extractedkeyw_pla)
cnbc.head(3)





# counting number of repeats for person names, organization names and places for CNBC dataset
tags_count(cnbc)
cnbc.head(3)





# saving the dataset to CSV file
cnbc.to_csv('cnbc_extracted.csv', index=False)


# ## CNN News dataset




# reading CNN news CSV file from a local location
cnn = pd.read_csv("cnn_cleaned.csv") 
cnn.head(3)





# extracting  clean text, published date and title from HTML script using CNN articles URLs  
extracted_clean_text, extracted_published_date, extracted_title = text_extract(cnn) 
# inserting extracted clean text, published date and title into CNN dataset as new columns 
cnn.insert(0,'extracted_clean_text', extracted_clean_text)          
cnn.insert(1,'extracted_title', extracted_title)
cnn.insert(2,'extracted_published_date', extracted_published_date)
cnn.head(3)





# extracting  person names, organization names and places names from HTML script using CNN articles URLs
extractedkeyw_per, extractedkeyw_org, extractedkeyw_pla = tags_extract(cnn)
# inserting extracted person names, organization names and places names into CNN dataset as new columns 
cnn.insert(3,'extractedkeyw_per', extractedkeyw_per)          
cnn.insert(4,'extractedkeyw_org', extractedkeyw_org)
cnn.insert(5,'extractedkeyw_pla', extractedkeyw_pla)
cnn.head(3)





# counting number of repeats for person names, organization names and places for CNN dataset
tags_count(cnn)
cnn.head(3)





# saving the dataset to CSV file
cnn1.to_csv('cnn_extracted.csv', index=False)


# ## japan_times News dataset




# reading Japan Times news CSV file from a local location
japan_times = pd.read_csv("japan_times_cleaned.csv") 
japan_times.head(3)





# extracting  clean text, published date and title from HTML script using Japan Times articles URLs  
extracted_clean_text, extracted_published_date, extracted_title = text_extract(japan_times) 
# inserting extracted clean text, published date and title into Japan Times dataset as new columns 
japan_times.insert(0,'extracted_clean_text', extracted_clean_text)          
japan_times.insert(1,'extracted_title', extracted_title)
japan_times.insert(2,'extracted_published_date', extracted_published_date)
japan_times.head(3)





# extracting  person names, organization names and places names from HTML script using Japan Times articles URLs
extractedkeyw_per, extractedkeyw_org, extractedkeyw_pla = tags_extract(japan_times)
# inserting extracted person names, organization names and places names into Japan Times dataset as new columns 
japan_times.insert(3,'extractedkeyw_per', extractedkeyw_per)          
japan_times.insert(4,'extractedkeyw_org', extractedkeyw_org)
japan_times.insert(5,'extractedkeyw_pla', extractedkeyw_pla)
japan_times.head(3)





# counting number of repeats for person names, organization names and places for Japan Times dataset
tags_count(japan_times)
japan_times.head(3)





# saving the dataset to CSV file
japan_times.to_csv('japan_times_extracted.csv', index=False)

