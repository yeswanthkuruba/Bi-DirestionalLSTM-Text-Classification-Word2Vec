# -*- coding: utf-8 -*-
"""
Created on Sun Jun 10 16:20:01 2018

@author: yeswanth.kuruba
"""

import Word2Vec_ModelBuilding
import nltk.data
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Bidirectional
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import initializers
from keras.layers.recurrent import LSTM
from keras.layers.normalization import BatchNormalization
from keras import utils

data = pd.read_csv("Train_data.csv")          
data_features = data['text']
data["target"] = data["target"].astype('category')
data["target"] = data["target"].cat.codes
data_target = data['target']
y_target = utils.to_categorical(data_target, 5)

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
NatLangProcessor = Word2Vec_ModelBuilding.NatLangProcessing(tokenizer)
num_features = 64
model_name = Word2Vec_ModelBuilding.execute(data, num_features=num_features, min_word_count=1, context=4)

model_summary = Word2Vec.load(model_name)
train_reviews = []
for review in data_features:
    train_reviews.append(NatLangProcessor.Review_to_Words(review))

index2word_set = set(model_summary.wv.vocab)
reviews_ = []
for review in train_reviews:
    featureVec_ = []
    for word in review:
        if word in index2word_set:
            featureVec_.append(model_summary[word])
    reviews_.append(featureVec_)

final = []
for i in range(len(reviews_)):
    for j in range(29):
        try:
            d = reviews_[i][j]
            final.append(d)
        except:
            final.append(np.zeros((64),dtype="float32"))
        
final_data = np.array(final).reshape(len(data),29,num_features)    
    
X_train, X_test, y_train, y_test = train_test_split(final_data, y_target, test_size=0.15, random_state=42)

def BidirectionalRNN(input_shape):
    model = Sequential()
    model.add(Bidirectional(LSTM(32, kernel_initializer=initializers.lecun_uniform(35), return_sequences=True), input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    
    model.add(Bidirectional(LSTM(64, kernel_initializer=initializers.lecun_uniform(55), return_sequences=True)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Bidirectional(LSTM(32, kernel_initializer=initializers.lecun_uniform(35), return_sequences=False)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
            
    model.add(Dense(5, activation="softmax"))
    return model

nb_epoch = 500
early_stopping_epochs = 50   
batch_size = 200 

earlyStopping = EarlyStopping(monitor='val_loss', patience=early_stopping_epochs, verbose=1)

model_filepath = "model_classification.hdf5"
checkpointer  = ModelCheckpoint(filepath= model_filepath, monitor='val_loss', verbose=1, save_best_only = True)

input_shape = (X_train.shape[1], X_train.shape[2])    

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0.0005)
model = BidirectionalRNN(input_shape)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test, y_test), callbacks=[earlyStopping, reduce_lr, checkpointer] )    

min_loss = min(history.history['val_loss'])

scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))
