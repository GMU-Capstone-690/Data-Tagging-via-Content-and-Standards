# Data Tagging via Content & Standards
Data tagging is used for classification, arrangement and organization of data by assigning admin and descriptor tags. These metadata tags makes discovery easy in data catalogs. The goal of project is to extract clear, segregated and meaningful tags from text that allow the organization to automate the process of organizing their data inventory while maintaining DCAT standards. 

## Problem Statement & Solution Space
Manual text data tagging is time consuming and is neither effective nor efficient which makes data discovery and standardization an arduous process. The solution is to create an ML/AI model that can identify, categorize, and tag data based on content, while focusing on standardization of the generated tags. So, topic modeling algorithm LDA is used to find topics, thus automating metadata tagging process. 

## Project Pipeline
- [Data Collection](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/tree/main/Data%20Cleaning) Data is collected from data.world website.
- [Data Cleaning](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/tree/main/Data%20Cleaning) Data is cleaned by removing nulls, duplicates and expired links.
- [Data Extraction](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/tree/main/Data%20Extraction) Clean text and admin tags are extracted from html content. 
- [Data Modeling](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/tree/main/Data%20Modeling) Training and tuning of LDA model is done. 

## Installation
We used jupyter notebook to run our project on local system and then convert them to .py files for Github. In order to run the model on local machine, use the compatible version of python3 and run python3 [cleaning.py](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/blob/main/Data%20Cleaning/Data_Cleaning_of_all_datasets.py), [extraction.py](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/blob/main/Data%20Extraction/Data_Extracting.py) and [modeling.py](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/blob/main/Data%20Modeling/Modeling.py) in the command prompt or convert them to jupyter notebooks. 

Mallet implementation by gensim is required for finding optimal number of topics. This installation is required in [modeling.py](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/blob/main/Data%20Modeling/Modeling.py). You need to [download](http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip) the zipfile, unzip it and provide the path to mallet in the unzipped directory to gensim.models.wrappers.LdaMallet.

```
mallet_path = 'path/to/mallet-2.0.8/bin/mallet'
```
*(update mallet_path in [modeling](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/blob/main/Data%20Modeling/Modeling.py) file in line 1065)* 

## Model Implementation
- Download all five datasets (BBC, CNBC, CNN, Aljazeera, Japan times) from [Data Collection](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/tree/main/Data%20Collection) folder.
- All datasets are kept separate for processing. Run [cleaning.py](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/blob/main/Data%20Cleaning/Data_Cleaning_of_all_datasets.py) file for pre-processing data which will remove spaces, N/A, blank and null values. Also, if expired or invalid URL found, these records are dropped and result is saved to fresh csv file.  
- Load clean data from step 2 to [extraction.py](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/blob/main/Data%20Extraction/Data_Extracting.py) which will extract clean text, title, and published date from html content of URLs. Admin tags like person, organization and places with their counts are also extracted.
- Load extracted data from step 3 to [modeling.py](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/blob/main/Data%20Modeling/Modeling.py) which will [pre-process](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/tree/main/Data%20Modeling) data to create dictionary(id2word) and the corpus which are two main inputs to the LDA topic model. Optimal topics are generated and mapped to tags. 

## Use Case
For an organization that maintains its dataset inventory based on URLs, this solution will enable them to tag data and upload the collections into their catalog as records. The tags will be useful in building a search engine for the catalog that will allow users to pull datasets based on keywords that match the tags. For example, if a user wants to find a data collection related to sports and racing, he can enter the same in the search box and the collections with tags that match this keyword in the data catalog will be retrieved by the search engine.

## Credits
George Mason Data Analytics Engineering Program: DAEN 690
<br /> Fall 2022 Team Code- Data Bees:
- Shagufta Hassan (https://www.linkedin.com/in/shagufta-hassan-08/)
- Durafshan Jawad (https://www.linkedin.com/in/durafshan-jawad-5b07b0133/)
- Lama Alznaidi (https://www.linkedin.com/in/lama-a-a51420152/)
- Prajna Shetty (https://www.linkedin.com/in/prajna-shetty-517ab0244/)
- Madesh Chinnathevar Ramesh (https://www.linkedin.com/in/madeshcr/)

