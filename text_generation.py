# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:16:01 2022

@author: metasystems
"""

import tensorflow as tf

import numpy as np
import os
import time
import matplotlib.pyplot as plt
import sys

os.chdir(r"E:\certificate\nlp")

#%%
def text_from_ids(ids):
    """
    Generate Text from characters

    Parameters
    ----------
    ids : Eager Tensor
        integer representation of text.

    Returns
    -------
    TYPE
        String blocks.

    """
    return tf.strings.reduce_join(chars_from_ids(ids), axis=-1)
#%%
class TextModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,
                                       return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
    
    def call(self, inputs, states=None, return_state=None, training=False):
        x = self.embedding(inputs, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state = states, training=training)
        x = self.dense(x, training=training)
        
        if return_state:
            return x, states
        else:
            return x
#%%
SEQ_LENGTH = 100
BATCH_SIZE = 128
BUFFER_SIZE = 10000
EPOCHS = 30

#%% Path to Dataset
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

#%% Read Text
text = open(path_to_file, "rb").read().decode(encoding='utf-8')
print(f"Text Length is {len(text)}")

#%%
vocab = sorted(set(text))
print(f"Unique Characters: {len(vocab)}")

#%% Encode/Decode
# generate integer representation from raw text
ids_from_chars = tf.keras.layers.StringLookup(vocabulary = list(vocab), mask_token=None)

# to generate text from integer representation
chars_from_ids = tf.keras.layers.StringLookup(vocabulary = ids_from_chars.get_vocabulary(), invert = True, mask_token=None) 

#%% Preprocess text
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))

#%% Generate Sequences
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
sequences = ids_dataset.batch(SEQ_LENGTH+1, drop_remainder=(True))

print(f"Sequence Number is {len(sequences)}")

#%% Split sequence to input data and corresponding target (label)
def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    
    return input_text, target_text

#%% Map Input Data and Labels into one dataset
dataset = sequences.map(split_input_target)

#%% Example
for input_example, target_example in dataset.take(1):
    print("Input :", text_from_ids(input_example).numpy())
    print("Target:", text_from_ids(target_example).numpy())

#%% Batching and Prefetching
dataset = (dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).prefetch(tf.data.AUTOTUNE))
print(dataset)

#%% Initialize Model
# Length of the vocabulary in StringLookup Layer
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

model = TextModel(vocab_size = vocab_size,
                  embedding_dim = embedding_dim,
                  rnn_units = rnn_units)

#%%
for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

#%% Get and Use Predictions
sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
print("This gives us, at each timestep, a prediction of the next character index:\n", sampled_indices)

print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
print()
print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())

#%%
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
example_batch_mean_loss = loss(target_example_batch, example_batch_predictions)
print(f"Prediction Shape: {example_batch_predictions.shape} # (batch_size, sequence_length, vocab_size)")
print(f"Mean Loss: {example_batch_mean_loss}")
print(f"Exponential Loss is {tf.exp(example_batch_mean_loss).numpy()}\nVocan Size is {vocab_size}\nModel is practically guessing characters")

#%% Compile Model
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = loss)

#%% Fit
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

history = model.fit(dataset,
                    epochs=EPOCHS,
                    callbacks=[checkpoint_callback])
#%% Plot Loss
plt.plot(history.history["loss"])
plt.title("Loss Curve")
plt.xlabel("Epochs")
plt.ylabel("Loss Value")

#%% Single Step Prediction. Run this in a loop to generate text
class OneStep(tf.keras.Model):
    def __init__(self, model, chars_from_ids, ids_from_chars, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.model = model
        self.chars_from_ids = chars_from_ids
        self.ids_from_chars = ids_from_chars
        
        # Create a mask to prevent [UNK] from being generated
        skip_ids = self.ids_from_chars(['[UNK'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index
            values = [-float('inf')] * len(skip_ids),
            indices = skip_ids,
            # Match the shape to the vocabulary
            dense_shape = [len(ids_from_chars.get_vocabulary())])
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)
    
    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert Strings to token IDs
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.ids_from_chars(input_chars).to_tensor()
        
        # Run the Model
        # predicted_logits.shape is [batch, char, next_char_logits]
        predicted_logits, states = self.model(inputs = input_ids,
                                              states = states,
                                              return_state = True)
        # Only use last prediction
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        
        # Apply the prediction mask: prevent [UNK] from being generated
        predicted_logits = predicted_logits + self.prediction_mask
        
        # Sample the output logits
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)
        
        # Convert from token ids to characters
        predicted_chars = self.chars_from_ids(predicted_ids)
        
        # Return the characters and model state
        return predicted_chars, states
#%% Generate some text
one_step_model = OneStep(model, chars_from_ids, ids_from_chars)

start = time.time()
states = None
next_char = tf.constant(["XENIYA: "])
result = [next_char]

# Generate next 1000 characters
for n in range(1000):
    next_char, states = one_step_model.generate_one_step(next_char, states=states)
    result.append(next_char)

result = tf.strings.join(result) # generate string from chars
end = time.time()
print(f"\n\n{'=' * 80}\n{result[0].numpy().decode('utf-8')}\n\n{'=' * 80}")
print(f'\nRun Time is {end-start}')

#%% Generate text in batches (here it is 5)
start = time.time()
states = None
next_char = tf.constant(['ROMEO:', 'ROMEO:', 'ROMEO:', 'ROMEO:', 'ROMEO:'])
result = [next_char]

for n in range(1000):
  next_char, states = one_step_model.generate_one_step(next_char, states=states)
  result.append(next_char)

result = tf.strings.join(result)
end = time.time()
print(result, '\n\n' + '_'*80)
print('\nRun time:', end - start)

#%% Save the model
tf.saved_model.save(one_step_model, 'one_step')
one_step_reloaded = tf.saved_model.load('one_step')
sys.exit()


































