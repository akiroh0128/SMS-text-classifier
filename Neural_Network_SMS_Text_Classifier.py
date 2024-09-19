#!/usr/bin/env python
# coding: utf-8

# In[1]:


try:
  get_ipython().system('pip install tf-nightly')
except Exception:
  pass
import tensorflow as tf
import pandas as pd
from tensorflow import keras
get_ipython().system('pip install tensorflow-datasets')
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing import sequence
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
print(tf.__version__)


# In[ ]:


# getting data files from project data of freecodecamp.org
get_ipython().system('wget https://cdn.freecodecamp.org/project-data/sms/train-data.tsv')
get_ipython().system('wget https://cdn.freecodecamp.org/project-data/sms/valid-data.tsv')

train_file_path = "train-data.tsv"
test_file_path = "valid-data.tsv"


# In[ ]:


#Convert the data into pandas dataframes
train_dataset=pd.read_csv(train_file_path,sep='\t',names=['type','message'])
test_dataset=pd.read_csv(test_file_path,sep='\t',names=['type','message'])

#Remove rows with missing data
train_dataset.dropna()
test_dataset.dropna()

batch_size=32


# In[ ]:


#Label the categorical variables into numercial data
train_dataset["type"] = pd.factorize(train_dataset["type"])[0]
test_dataset["type"] = pd.factorize(test_dataset["type"])[0]


# In[ ]:


#COnvert the dataframes into tensorflow dataset
train_labels =  train_dataset["type"].values
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_dataset["message"].values, train_labels)
)
test_labels =  test_dataset["type"].values
test_dataset = tf.data.Dataset.from_tensor_slices(
    (test_dataset["message"].values, test_labels)
)


# In[ ]:


#Batching and prefetching the dataset to reshape it
train_dataset = train_dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)#shuffling the dataset improves the efficiency of training

test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


# In[ ]:


#Create a TextVectorization layer for our model
vec = TextVectorization(
    output_mode='int',
    max_tokens=1000,
    output_sequence_length=1000,
)

vec.adapt(train_dataset.map(lambda text, label: text))


# In[ ]:


#Creating the model using keras.Sequential
model = tf.keras.Sequential([
    vec,
    tf.keras.layers.Embedding(
        len(vec.get_vocabulary()),
        64,
        mask_zero=True,
    ),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(1)
])


# In[ ]:


#training the model with our datasets
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])

history = model.fit(train_dataset,validation_data=test_dataset, epochs=10,validation_steps=30)


# In[ ]:


# function to predict messages based on model
def predict_message(pred_text):
    ps = model.predict([pred_text])
    print(ps)
    p = ps[0][0]
    return [p, "ham" if p <0.5 else "spam"]

pred_text = "how are you doing today?"

prediction = predict_message(pred_text)
print(prediction)


# In[ ]:


# To test our model with some sample text messages
def test_predictions():
  test_messages = ["how are you doing today",
                   "sale today! to stop texts call 98912460324",
                   "i dont want to go. can we try it a different day? available sat",
                   "our new mobile video service is live. just install on your phone to start watching.",
                   "you have won £1000 cash! call to claim your prize.",
                   "i'll bring it tomorrow. don't forget the milk.",
                   "wow, is your arm alright. that happened to me one time too"
                  ]

  test_answers = ["ham", "spam", "ham", "spam", "spam", "ham", "ham"]
  passed = True

  for msg, ans in zip(test_messages, test_answers):
    prediction = predict_message(msg)
    if prediction[1] != ans:
      passed = False

test_predictions()

