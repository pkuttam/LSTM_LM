
from random import randint
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
import numpy as np
import string
import re
from nltk.corpus import gutenberg
from random import randint
from pickle import load
from keras.models import load_model

# cleaning the text file 
def clean_doc(doc):
    doc = doc.replace('--', ' ')
    tokens = doc.split() # split in words
    tokens = [re.sub(r'[^\w\s]','',w) for w in tokens] 
    tokens = [word for word in tokens if word.isalpha()]# removing words which are not alphabetic
    tokens = [word.lower() for word in tokens]
    return tokens

# gutenberg corpus
tokens_raw1 = gutenberg.words('austen-emma.txt')

# splitting in test and train
tokens_train1  = tokens_raw1[0:int(0.8*n1)];
tokens_test1 = tokens_raw1[int(0.8*n1):n1]; 

tokens_raw2 = gutenberg.words('austen-sense.txt')
n2 = len(tokens_raw2)
tokens_train2  = tokens_raw2[0:int(0.8*n2)];
tokens_test2 = tokens_raw2[int(0.8*n2):n2]; 

tokens_raw = tokens_train1 + tokens_train2
tokens_test = tokens_test1 + tokens_test2

text_raw = ' '.join(tokens_raw)
tokens = clean_doc(text_raw) # cleaning the text data


length = 50 + 1
seq_length = length -1 # LSTM cells
step =2
sequences = list()
for i in range(0,len(tokens)-length, step):
    # select sequence of tokens
    seq = tokens[i:i+length]
    # convert into a line
    line = ' '.join(seq)
    # store
    sequences.append(line)

lines = sequences

# tokenizer to encode
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
sequences = tokenizer.texts_to_sequences(lines)

# vocabulary size (V)
vocab_size = len(tokenizer.word_index) + 1

# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
y = to_categorical(y, num_classes=vocab_size)
seq_length = X.shape[1]

# define model
model = Sequential()
model.add(Embedding(vocab_size, 300, input_length=seq_length))
model.add(LSTM(100, return_sequences=True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(vocab_size, activation='softmax'))

# compile model and traing
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit model
model.fit(X, y, batch_size=128, epochs=80)

# save the model to file for future use
model.save('model_a03_wl_300_100_80.h5')
# save the tokenizer to file for future use
dump(tokenizer, open('tokenizer_.pkl', 'wb'))

# In[]
from keras.preprocessing.sequence import pad_sequences


# generate a sequence from a language model
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
        in_text += ' ' + out_word
        result.append(out_word)
    return ' '.join(result)

# select a seed sentence 
seed_text = lines[randint(0,len(lines))]
print(seed_text + '\n')
generated = generate_seq(model, tokenizer, seq_length, seed_text, 15)
print(generated)

# perplexity measure
test_tokens = tokens_test[1:-1]
in_text = test_tokens[0]
logp = 0
c=0
for word in test_tokens:
    encoded = tokenizer.texts_to_sequences([in_text])[0]
    encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
    yhat1 = model.predict(encoded,verbose=0)[0]
    next_word = word
    if next_word in tokenizer.word_index:
        p = yhat1[tokenizer.word_index[next_word]]
        if  p >=0.00001:
            c+=1
            logp = logp + np.log(p)

    else:
        p = 1
        c+=1
        logp = logp + np.log(p)
    
    in_text +=' ' + next_word
pplxty = np.exp(-1*logp/c)
print(pplxty)
