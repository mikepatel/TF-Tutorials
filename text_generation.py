# 9/6/18

# Text generation
# using tf.keras
# using eager execution
# https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/eager/python/examples/generative_examples/text_generation.ipynb

# dataset: Shakespeare writing (Andrej Karpathy)
#

# Notes:
# RNN
# lower-level implementation
# character based model
# structure of generated text
# experiment w/ temperature parameter

################################################################################
# IMPORTs
import numpy as np
import unidecode
import time

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Embedding, CuDNNGRU, GRU, Dense
from tensorflow.keras.activations import sigmoid
from tensorflow.keras.initializers import glorot_uniform
#from tensorflow.keras.optimizers import Adam

tf.enable_eager_execution()

################################################################################
print(tf.__version__)

# load dataset
path_to_file = keras.utils.get_file(
    "shakespeare.txt",
    "https://storage.googleapis.com/yashkatariya/shakespeare.txt"
)

# explore data
text = unidecode.unidecode(open(path_to_file).read())
print("\nLength of dataset text: " + str(len(text)))
#print(text[:1000])

# create dictionaries to map chars <--> indices
unique = sorted(set(text))  # 'unique' contains all unique chars in file
char2idx = {u: i for i, u in enumerate(unique)}  # create mapping from unique char -> indices
idx2char = {i: u for i, u in enumerate(unique)}  # create mapping from indices -> unique char

################################################################################
# model parameters
max_sentence_length = 100  # 100 char chunks of input
vocab_size = len(unique)  # pool from all unique char
embedding_dim = 256
num_RNN_units = 1024
batch_size = 64
buffer_size = 10000  # used to shuffle dataset
num_epochs = 30
num_char_generated = 1000
start_string = "M"
temperature = 1.0  # low temps -> more predictable text, high temps -> surprising text

################################################################################
# Input and output tensors
# vectorize input and target text
# because model only "understands" numbers, not strings
# create vectors
input_text = []  # all char in chunk except last
target_text = []  # all char in chunk except first

# convert each char into numbers using char2idx
for f in range(0, len(text) - max_sentence_length, max_sentence_length):
    inputs = text[f: f+max_sentence_length]
    targets = text[f+1: f+1+max_sentence_length]

    input_text.append([char2idx[i] for i in inputs])
    target_text.append([char2idx[t] for t in targets])

print("Shape of input text: " + str(np.array(input_text).shape))
print("Shape of target text: " + str(np.array(target_text).shape))

################################################################################
# create batches and shuffle
# using tf.data
dataset = tf.data.Dataset.from_tensor_slices((input_text, target_text)).shuffle(buffer_size=buffer_size)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

################################################################################
# build model
# architecture:
# layer 1 - embedding
# layer 2 - GRU
# layer 3 - fully connected


class Model(Model):
    def __init__(self, vocab_size, embedding_dim, num_RNN_units, batch_size):
        super(Model, self).__init__()
        self.num_RNN_units = num_RNN_units
        self.batch_size = batch_size

        # layer 1
        self.embedding = Embedding(vocab_size, embedding_dim)

        # layer 2
        if tf.test.is_gpu_available():
            self.gru = CuDNNGRU(
                self.num_RNN_units,
                return_sequences=True,
                return_state=True,
                recurrent_initializer=glorot_uniform()
            )
        else:
            self.gru = GRU(
                self.num_RNN_units,
                return_state=True,
                recurrent_activation=sigmoid,
                recurrent_initializer=glorot_uniform()
            )

        # layer 3
        self.full_connect = Dense(vocab_size)

    def call(self, x, hidden):
        x = self.embedding(x)

        # output shape = (batch_size, max_sentence_length, hidden_size)
        # states shape = (batch_size, hidden_size)

        # states variable to preserve state of model
        # will be used to pass at every step during model training
        output, states = self.gru(x, initial_state=hidden)

        # reshape output before passing to Dense layer
        # after reshape, shape is (batch_size * max_sentence_length, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output of dense layer will be predictions for every time_steps(max_sentence_length)
        # shape after dense layer = (max_sentence_length * batch_size, vocab_size)
        x = self.full_connect(output)

        return x, states


################################################################################
# instantiate, configure optimizer and loss function
model = Model(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    num_RNN_units=num_RNN_units,
    batch_size=batch_size
)

# optimizer
optimizer = tf.train.AdamOptimizer()


# loss function
def loss_fn(real, predictions):
    return tf.losses.sparse_softmax_cross_entropy(labels=real, logits=predictions)


# train model
# use GradientTape()
# initialize hidden state w/ zeros
# iterate over dataset batch by batch
# calculate predictions and hidden states
for epoch in range(num_epochs):
    start = time.time()

    # initialize hidden start before each epoch
    hidden = model.reset_states()

    for(batch, (inp, targ)) in enumerate(dataset):
        with tf.GradientTape() as tape:
            # feed hidden state back into model
            predictions, hidden = model(inp, hidden)

            # reshape target for loss function
            targ = tf.reshape(targ, (-1,))
            loss = loss_fn(targ, predictions)

        gradients = tape.gradient(loss, model.variables)
        optimizer.apply_gradients(
            zip(gradients, model.variables),
            global_step=tf.train.get_or_create_global_step()
        )

        if batch % 100 == 0:
            print("Epoch {} Batch {} Loss {:.4f}".format(epoch+1, batch, loss))

    print("Epoch {} Loss {:.4f}".format(epoch+1, loss))
    print("Time taken for 1 epoch {} sec\n".format(time.time() - start))

################################################################################
# predict using trained model
# generate some text
# choose a starting string
# initialize hidden state
# set number of char that should be generated
# use predicted word as next input to model
# hidden state returned by model is fed back so that model begins to
# "understand" context (capitalization, make paragraphs...i.e. mimic Shakespeare style)

# vectorize (convert start string to numbers)
input_eval = [char2idx[s] for s in start_string]
input_eval = tf.expand_dims(input_eval, 0)

# create empty string to store results
gen_text = ""

# hidden state shape = (batch_size, num_RNN_units); batch_size = 1
hidden = [tf.zeros((1, num_RNN_units))]

for i in range(num_char_generated):
    predictions, hidden = model(input_eval, hidden)

    # use multinomal distribution to predict word returned by model
    predictions /= temperature
    predict_id = tf.multinomial(tf.exp(predictions), num_samples=1)[0][0].numpy()

    # pass previous hidden state and predicted word as next input to model
    input_eval = tf.expand_dims([predict_id], 0)

    gen_text += idx2char[predict_id]

################################################################################
# print output
print("\n#########################################################")
print(start_string + gen_text)
print("\n#########################################################")
