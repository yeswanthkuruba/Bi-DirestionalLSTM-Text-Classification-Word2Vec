# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 17:50:10 2018

@author: yeswanth.kuruba
"""
import pandas as pd
import re
import nltk
import nltk.data
from nltk.corpus import stopwords
from gensim.models import word2vec

class NatLangProcessing(object):
    """NatLangProcessing is class for processing raw text into segments for further learning"""
    def __init__(self,tokenizer):
        self.tokenizer=tokenizer
    
    def Review_to_Words( self, review, remove_stopwords=False ):
        #
        review_text = re.sub(r"[^a-zA-Z]+", ' ', review)
        # Convert words to lower case and split them
        words = review_text.lower().split()
        #  Optionally remove stop words (false by default)
        if remove_stopwords:
            stops = set(stopwords.words("english"))
            stops = stops - set(['what','who','where', 'when', 'is', 'in', 'whom', 'which', 'whom', 'why', 'how'])
            words = [w for w in words if not w in stops]
        #  Return a list of words
        return(words)

    # Define a function to split a review into parsed sentences
    def Review_to_Sentence( self, review, remove_stopwords=False ):
        # Function to split a review into parsed sentences. Returns a
        # list of sentences, where each sentence is a list of words
        #  Use the NLTK tokenizer to split the paragraph into sentences
        review = re.sub(r"[^a-zA-Z]+"," ", review)
        raw_sentences = self.tokenizer.tokenize(review.strip())
        #  Loop over each sentence
        sentences = []
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call review_to_wordlist to get a list of words
                sentences.append( self.Review_to_Words( raw_sentence ))
        # Return the list of sentences (each sentence is a list of words,
        # so this returns a list of lists
        return sentences
        
def execute(data, num_features=32, min_word_count=1, context=4):
    
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')    
    NatLangProcessor = NatLangProcessing(tokenizer)
    sent=[]
    
    for review in data["text"]:
        sent += NatLangProcessor.Review_to_Sentence(review)
            
    num_workers = 4       # Number of threads to run in parallel
    downsampling = 1e-3   # Downsample setting for frequent words
#    bigram_transformer = gensim.models.Phrases(sent)
    # Initialize and train the model (this will take some time)
    model = word2vec.Word2Vec(sent, workers=num_workers, size=num_features, min_count = min_word_count, \
                window = context, sample = downsampling, seed=1)

    # If you don't plan to train the model any further, calling
    # init_sims will make the model much more memory-efficient.
    model.init_sims(replace=True)

    # save the model for later use. You can load it later using Word2Vec.load()
    model_name = "Fulldata_W2vec_"+str(num_features)+"dim_"+str(min_word_count)+"mc"
    model.save(model_name)
    return model_name

