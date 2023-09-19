# -*- coding: utf-8 -*-

# A dependency of the preprocessing for BERT inputs

import os
import pandas as pd
import numpy as np
#import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
from official.nlp import optimization  # to create AdamW optimizer

import matplotlib.pyplot as plt

tf.get_logger().setLevel('ERROR')
BATCH_SIZE = 32
epochs = 5
init_lr = 3e-5

# loading the train, test and val dataset
import pandas as pd
train_df = pd.read_excel(os.getcwd() + r'/data/imperative_dataset_train.xlsx')
test_df = pd.read_excel(os.getcwd() + r'/data/imperative_dataset_test.xlsx')
val_df = pd.read_excel(os.getcwd() + r'/data//imperative_dataset_val.xlsx')

train_df['target'] = np.where(train_df['Final class'] == 'Imperative', 1,0)
test_df['target'] = np.where(test_df['Final class'] == 'Imperative', 1,0)
val_df['target'] = np.where(val_df['Final class'] == 'Imperative', 1,0)

with tf.device('/cpu:0'):
  train_data = tf.data.Dataset.from_tensor_slices((train_df['Final_text'].values, train_df['target'].values))
  val_data = tf.data.Dataset.from_tensor_slices((val_df['Final_text'].values, val_df['target'].values))
  test_data = tf.data.Dataset.from_tensor_slices((test_df['Final_text'].values, test_df['target'].values))

for text, label in train_data.take(1):
  print(text.numpy())
  print(label.numpy())

with tf.device('/cpu:0'):
  train_ds = train_data.shuffle(100).batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
  val_ds = val_data.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
  test_ds = test_data.batch(BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

for text_batch, label_batch in train_ds.take(1):
  for i in range(3):
    print(f'Sentence: {text_batch.numpy()[i]}')
    label = label_batch.numpy()[i]
    print(f'Label : {label}')

# Loading BERT model and pre processing model

tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4'
tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3'

# Define model - here regularisation layers would be fine-tuned basis the accuracy scores
def build_classifier_model():
  text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
  preprocessing_layer = hub.KerasLayer(tfhub_handle_preprocess, name='preprocessing')
  encoder_inputs = preprocessing_layer(text_input)
  encoder = hub.KerasLayer(tfhub_handle_encoder, trainable=True, name='BERT_encoder')
  outputs = encoder(encoder_inputs)
  net = outputs['pooled_output']
  net = tf.keras.layers.Dropout(0.1)(net)
  net = tf.keras.layers.Dense(1, activation=None, name='classifier')(net)
  return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()


# Defining Loss
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
metrics = tf.metrics.BinaryAccuracy()

steps_per_epoch = tf.data.experimental.cardinality(train_ds).numpy()
num_train_steps = steps_per_epoch * epochs
num_warmup_steps = int(0.1*num_train_steps)

optimizer = optimization.create_optimizer(init_lr=init_lr,
                                          num_train_steps=num_train_steps,
                                          num_warmup_steps=num_warmup_steps,
                                          optimizer_type='adamw')

classifier_model.compile(optimizer=optimizer,
                         loss=loss,
                         metrics=metrics)

history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=epochs)

# Evaluation on test data
loss, accuracy = classifier_model.evaluate(test_ds)

print(f'Loss: {loss}')
print(f'Accuracy: {accuracy}')

## Saving the model
saved_model_path = os.getcwd() + r'/bert_uncased_batch32_lr3e5_ep5_1_sent_model'
classifier_model.save(saved_model_path, include_optimizer=False)

"""# Model Prediction"""

import os
import pandas as pd
import numpy as np
import re
#import shutil

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

saved_model_path = os.getcwd() + r'/bert_uncased_batch32_lr3e5_ep5_1_sent_model'

text_data = pd.read_excel( os.getcwd() + r'/sample_sentences.xlsx')
text_data['text'] = [re.split(r'\.|\!|\?|\n',x)[0].strip().lower()  for x in text_data['Sentences']]
text_data.drop_duplicates('text', inplace=True)
len(text_data)

reloaded_model = tf.saved_model.load(saved_model_path)

reloaded_results = tf.sigmoid(reloaded_model(tf.constant(list(text_data['text']))))

predicted_labels = ['Imperative' if score[0].numpy() >=0.5 else "Non-Imperative" for score in reloaded_results]

text_data['Predicted class'] = predicted_labels

text_data.to_excel(os.getcwd() +r'/bert_uncovered_cont_sent_results.xlsx', index=False)

