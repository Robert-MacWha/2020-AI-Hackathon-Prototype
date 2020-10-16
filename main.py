"""
   Path: D:/Github Repos/2020-AI-Hackathon-Prototype

   Required libraries:
      Tensorflow
      Numpy
      
"""

# disable tensorflow logging
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from time import time
start_time = 0

import sys, requests, random, math

import tensorflow as tf

# preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer            # converts text into lists of ints and a lookup list
from tensorflow.keras.preprocessing.sequence import pad_sequences    # normalizes the lengths of the lists of ints

# ML objects
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Embedding, Concatenate

import numpy as np

""" Server Initialization """
dataset_url       = "https://reqres.in/api/userz?delay=3"
new_data_url      = "https://reqres.in/api/userz?delay=3"
return_sorted_url = "https://reqres.in/api/userz?delay=3"

# get the dataset from the website
dataset = None
i = 0
max_attempts = 10

# try [max_attempts] times
while dataset == None:
   try:
      # make a request, if it is valid set the dataset var to the given dataset
      r = requests.get(dataset_url, timeout=10)
      dataset = r.json()
   except:
      # increment the counter and print an warning message
      i += 1
      print(f"Request {i}/{max_attempts} failed, no responce / invalid responce")

   # if the server has not responded for [max_attempts] tries, just quit the program
   if i >= max_attempts:
      print("Failed to connect to server, load dataset onto website before running this script")
      sys.exit()

# convert the dataset into a list of resumes
resumes_dataset = []

""" Functions --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------"""
def buildEncoder():
   i = Input(shape=(input_shape))

   x = Embedding(output_dim=embeding_dim)(i)

   x = Flatten()(x) # embeding_dim * input_shape
   
   x = Dense(2048, activation="relu")(x)
   x = Dense(1024, activation="relu")(x)

   return i, x

""" Create input preprocessor (resume text) -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- """
start_time = time()

# words that will be considered in preprocessing.  Any words not included will be replaced with the oov_token (currently <OOV>)
tokenizer_text = random.choice(resumes_dataset, math.floor(len(resumes_dataset) / 10))

# num_words is the max amount of encoded words, oov_token is the character used if the tokenizer is looking for a previously unseen word
tokenizer = Tokenizer(num_words = 1000, oov_token="<OOV>")
tokenizer.fit_on_texts(tokenizer_text)

sequences       = tokenizer.texts_to_sequences(tokenizer_text)                             # converts the text inputs into lists of ints
resumes_dataset = pad_sequences(sequences, padding="post", truncating="post")              # normalizes the lengths of the int lists
resumes_dataset = np.asarray(resumes_dataset)                                              # convert the list into a np array (easier to work with)


print(f"Processed Tokenizer in {time() - start_time}s")

# print(padded_sequences.shape)

""" Create the NLP model -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- """
start_time = time()

# hyperparams
input_shape = padded_sequences.shape[1]
vocab_size = 1000
embeding_dim = 32

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
model.compile(
   loss="categorical_crossentropy",
   optimizer="adam"
)
# print(model.summary())

print(f"Created model in {time() - start_time}s")

""" Training process -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- """
# xs and ys are passed from the front end
LOSS_GOAL = 0.01
BATCH_SIZE = 32

model_loss = LOSS_GOAL + 1 # starts of higher than the goal

Xs_r0 = []
Xs_r1 = []
ys = []

while model_loss > LOSS_GOAL and not len(Xs_r0) < BATCH_SIZE * 2:
   # try to train the model
   if len(Xs_r0) > BATCH_SIZE:
      history = model.fit([Xs_r0, Xs_r1], ys, epochs = 1, batch_size = BATCH_SIZE)
      model_loss = history.history["loss"]

   # see if the website has any new data for me
   try:
      # create the post info
      resumes = random.sample(range(len(Xs)), 2)
      payload = {"0": resumes[0], "1": resumes[1]}

      # call the post
      r = requests.post(new_data_url, json=payload)

      # get the result and parse the info
      result = r.json()
      r0 = result["0"]
      r1 = result["1"]
      preferred_resume = result["preferred"]

      # append the new inputs to the training data
      Xs_r0.append(resumes_dataset[r0])
      Xs_r1.append(resumes_dataset[r1])

      ys.append([1 - preferred_resume, preferred_resume])

   except:
      print("Failed to connect to website, unable to get new training data")
      sys.exit()

# training is done, make a final post to the website with the ordered list of the resumes
def compareResumes(r1, r2):
   model_prediction = model.predict([r1, r2])

   if model_prediction[0] > model_prediction[1]:
      return -1

   return 1

# sort the list
sorted_resumes = resumes_dataset
sorted_resumes.sort(key=compareResumes)

# post to the website
try:
   r = requests.post(return_sorted_url, json=sorted_resumes)
except:
   print("Unable to post sorted list to website")