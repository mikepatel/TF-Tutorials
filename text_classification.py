# TF tutorial
# 8/14/18

# Basic Text Classification
# using tf.keras
# https://www.tensorflow.org/tutorials/keras/basic_text_classification

# classify movie reviews as either positive or negative (binary classification)

# dataset: IMDB
# training set size: 25k
# test set size: 25k

# Notes:
# num_epochs =  to prevent overfitting
# accuracy = 87.3% after 40 epochs

################################################################################
# IMPORT
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

################################################################################
print("TF version: ", tf.__version__)

# load dataset
imdb = keras.datasets.imdb
(train_feat, train_labels), (test_feat, test_labels) = imdb.load_data(num_words=10000)
# num_words -> keeps the top 10k most frequently occurring words in training set
# discard rarely occurring words to keep data size small and manageable
# exploring the data
# array of integers representing the words of the movie review
# class label 0 = negative review, 1 = positive review
# reviews will have varying length
print("Training entries: {}, labels: {}".format(len(train_feat), len(train_labels)))

# converting integers back to words
word_index = imdb.get_word_index()  # dict mapping words to integer index
word_index = {k: (v+3) for k, v in word_index.items()}

# some of the indices are reserved
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

# create a dictionary
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


#
def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# print(decode_review(train_feat[0]))

# preparing the data
# reviews -> arrays of ints -> tensors -> NN
# standardize lengths (reviews have varying length) using pad_sequences
train_feat = keras.preprocessing.sequence.pad_sequences(
    train_feat,
    value=word_index["<PAD>"],
    padding="post",
    maxlen=256
)
test_feat = keras.preprocessing.sequence.pad_sequences(
    test_feat,
    value=word_index["<PAD>"],
    padding="post",
    maxlen=256
)

################################################################################
# building the model
vocab_size = 10000  # input shape

model = Sequential()
model.add(Embedding(vocab_size, 16))  # integer-encoded vocab and word-index
model.add(GlobalAveragePooling1D())  # handle input of variable length
model.add(Dense(16, activation=relu))
model.add(Dense(1, activation=sigmoid))  # output float between 0 and 1
model.summary()

# configure learning
model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
################################################################################
# creating validation set
val_feat = train_feat[:10000]
partial_train_feat = train_feat[10000:]

val_labels = train_labels[:10000]
partial_train_labels = train_labels[10000:]

# training the model
history = model.fit(
    partial_train_feat,
    partial_train_labels,
    epochs=40,
    batch_size=512,
    validation_data=(val_feat, val_labels),
    verbose=1
)

# evaluating the model
results = model.evaluate(test_feat, test_labels)  # test_loss, test_accuracy
print(results)

################################################################################
# graphs and visualization
history_dict = history.history  # contains everything that happened during training
# 4 entries: loss, val_loss, acc, val_acc that are dict keys
# for training and validation
train_acc = history_dict["acc"]
valid_acc = history_dict["val_acc"]
train_loss = history_dict["loss"]
valid_loss = history_dict["val_loss"]

num_epochs = range(1, len(train_acc) + 1)

# plot for loss
plt.plot(num_epochs, train_loss, "bo", label="Training loss")  # "bo" is for blue dot
plt.plot(num_epochs, valid_loss, "b", label="Validation loss")  # "b" is for solid blue line
plt.title("Training and Validation Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# plot for accuracy
plt.plot(num_epochs, train_acc, "bo", label="Training accuracy")
plt.plot(num_epochs, valid_acc, "b", label="Validation accuracy")
plt.title("Training and Validation Accuracy")
plt.xlabel("Number of Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()