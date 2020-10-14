"""
   Path: D:/Github Repos/2020-AI-Hackathon-Prototype

   Required libraries:
      Tensorflow
      Numpy
      
"""

# disable tensorflow logging
import os
from os import truncate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from time import time
start_time = 0

import sys, json

import tensorflow as tf

# preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer            # converts text into lists of ints and a lookup list
from tensorflow.keras.preprocessing.sequence import pad_sequences    # normalizes the lengths of the lists of ints

# ML objects
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Embedding, Concatenate

import numpy as np

print("aaa \xe2\x80\x93 aaa \n aaa")

""" Create input preprocessor (resume text) -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- """
start_time = time()

# words that will be considered in preprocessing.  Any words not included will be replaced with the oov_token (currently <OOV>)
sample_text = [
   "The stranger, who the reader soon learns is Victor Frankenstein, begins his narration. He starts with his family background, birth, and early childhood, telling Walton about his father, Alphonse, and his mother, Caroline. Alphonse became Caroline’s protector when her father, Alphonse’s longtime friend Beaufort, died in poverty. They married two years later, and Victor was born soon after.",

   "Frankenstein then describes how his childhood companion, Elizabeth Lavenza, entered his family. At this point in the narrative, the original (1818) and revised (1831) versions of Frankenstein diverge. In the original version, Elizabeth is Victor’s cousin, the daughter of Alphonse’s sister; when Victor is four years old, Elizabeth’s mother dies and Elizabeth is adopted into the Frankenstein family. In the revised version, Elizabeth is discovered by Caroline, on a trip to Italy, when Victor is about five years old. While visiting a poor Italian family,",

   "Caroline notices a beautiful blonde girl among the dark-haired Italian children; upon discovering that Elizabeth is the orphaned daughter of a Milanese nobleman and a German woman and that the Italian family can barely afford to feed her, Caroline adopts Elizabeth and brings her back to Geneva. Victor’s mother decides at the moment of the adoption that Elizabeth and Victor should someday marry."
]

# num_words is the max amount of encoded words, oov_token is the character used if the tokenizer is looking for a previously unseen word
tokenizer = Tokenizer(num_words = 1000, oov_token="<OOV>")
tokenizer.fit_on_texts(sample_text)

sequences = tokenizer.texts_to_sequences(sample_text)                                   # converts the text inputs into lists of ints
padded_sequences = pad_sequences(sequences, padding="post", truncating="post")            # normalizes the lengths of the int lists
padded_sequences = np.asarray(padded_sequences)                                         # convert the list into a np array (easier to work with)

print(f"Processed Tokenizer in {time() - start_time}s")

# print(padded_sequences.shape)

""" Create the NLP model -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- """
start_time = time()

# hyperparams
input_shape = padded_sequences.shape[1]
vocab_size = 1000
embeding_dim = 32

def buildEncoder():
   i = Input(shape=(input_shape))

   x = Embedding(vocab_size, embeding_dim)(i)

   x = Flatten()(x) # embeding_dim * input_shape
   
   x = Dense(2048, activation="relu")(x)
   x = Dense(1024, activation="relu")(x)

   return i, x

# build the two encoding portions of the model
i1, x1 = buildEncoder()
i2, x2 = buildEncoder()

# combine them and build the output portion
x = Concatenate()([x1, x2])

x = Dense(1024, activation="relu")(x)
x = Dense(256,  activation="relu")(x)
x = Dense(64,   activation="relu")(x)
x = Dense(2,    activation="sigmoid")(x)

model = Model([i1, i2], x)
# print(model.summary())

print(f"Created model in {time() - start_time}s")

"""  Training process """
# xs and ys are passed from the front end
EPOCHS = 10
BATCH_SIZE = 32

Xs = []
ys = []

new_xs = []
new_ys = []

for e in range(EPOCHS):
   