import pandas as pd
import collections
import matplotlib.pyplot as plt
import numpy as np
import string
import nltk
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.preprocessing import sequence
from keras.models import load_model
import re
import random
import sys
from keras.utils.data_utils import get_file
from keras.optimizers import RMSprop
import io
from pickle import dump
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from nltk.corpus import gutenberg
from random import randint
from pickle import load
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences


# Gutenberg data corpus 
tokens_raw1 = gutenberg.raw('austen-emma.txt').lower()

# spliting into training and test
n1 = len(tokens_raw1)
tokens_train1  = tokens_raw1[0:int(0.8*n1)];
tokens_test1 = tokens_raw1[int(0.8*n1):n1]; 

tokens_raw2 = gutenberg.raw('austen-sense.txt').lower()
# spliting into training and test
n2 = len(tokens_raw2)
tokens_train2  = tokens_raw2[0:int(0.8*n2)];
tokens_test2 = tokens_raw2[int(0.8*n2):n2]; 

# combining all splits
tokens_raw = tokens_train1 + tokens_train2
tokens_test = tokens_test1 + tokens_test2

# distinct character
char_set = sorted(list(set(tokens_raw)))

# dictionary from character to index and index to character
char_index_dict={}
index_char_dict={}
for i in range(len(char_set)):
    char_index_dict[char_set[i]]=i
    index_char_dict[i]=char_set[i]

tokens_raw=list(tokens_raw)

char_tr=[]
for t in tokens_raw:
    if(t!='\n'):
        char_tr.append(t)
    else:
        char_tr.append(" ")

length = 41 
seq_length = length -1 # LSTM 
sequences =[]
for i in range(length, len(char_tr),2):
    seq = char_tr[i-length:i]
    sequences.append(seq)

lines = sequences;

# tokenizer to conver character to number index
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences)
sequences = tokenizer.texts_to_sequences(sequences)

# 40 character will predict 41st character
X=[]
y=[]
for i in range(len(sequences)):
    X.append(sequences[i][:-1])
    y.append(sequences[i][-1])
X=np.array(X)
y=np.array(y).reshape(-1,1)

# define model in Keras, 
# INPUT -> embedding -> LSTM ->LSTM -> Dense -> softmax -> out probability
vocab_size = len(tokenizer.word_index)+1
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=40))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

# comopile and train the model with epoch 100 and batch_size = 256
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X,y, batch_size=256, epochs=100)

# save the model to file for future use , so, no compiling later
model.save('char_model.h5')
# save the tokenizer to file for future use
dump(tokenizer, open('char_tokenizer', 'wb'))

# Sentence generation definition
def generate_seq(model, tokenizer, seq_length, seed_text, n_words):
    result = list()
    in_text = seed_text
    # generate a fixed number of words
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        # truncate sequences to a fixed length
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        # predict probabilities for each word
        yhat = model.predict_classes(encoded, verbose=0)
        #yhat1 = model.predict(encoded, verbose=0)
        # map predicted word index to word
        out_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                out_word = word
                break
        # append to input
        in_text.append(out_word)
        result.append(out_word)
    return ''.join(result)
    
# select a seed text for starting histrory or context
seed_text = lines[randint(0,len(lines))]
print(''.join(seed_text) + '')
generated = generate_seq(model, tokenizer, length-1, seed_text, 80)
print(generated)

#perplexity measure
seq_length = length-1
test_tokens = tokens_test[1:10000]

in_text = [test_tokens[0]]
logp = 0
c=0
for word in test_tokens:
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    #print encoded
    yhat1 = model.predict(encoded,verbose=0)[0]
    next_word = word
    if next_word in tokenizer.word_index:
        p = yhat1[tokenizer.word_index[word]]
        if  p !=0:
            c+=1
            logp = logp + np.log(p)

    else:
        p = 1
        
        c+=1
        logp = logp + np.log(p)
    
    in_text.append(word)
    #print(in_text)
    
pplxty = np.exp(-1*logp/c)
print c
