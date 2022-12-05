The data in 'extracted_clean_text' column is used for analysis & modeling.
Following pre-processing steps are done to prepare data for LDA model:
- Conversion of text to list
- Remove emails, new line characters, unwanted quotes & stopwords
- Tokenization (Convert sentences to bag of words)
- Creating unigram, bigram & trigrams
- Lemmatization while keeping only nouns, adjectives, verbs & adverbs

Then master dictionary and corpus are generated which are input of LDA model. Model evaluation matrix like perplexity and coherence along with optimal number of topics are calculated in this [script](https://github.com/GMU-Capstone-690/Data-Tagging-via-Content-and-Standards/blob/main/Data%20Modeling/Modeling.py). 


