# TF tutorial
# 8/26/18

# Overfitting and Underfitting
# using tf.keras
# https://www.tensorflow.org/tutorials/keras/overfit_and_underfit

# classify movie reviews as either positive or negative (binary classification)

# dataset: IMDB
# training set size: 25k
# test set size: 25k

# Notes:
# prevent overfitting by using more training data
# prevent overfitting by regularization
# two types of regularization are weight regularization and dropout
################################################################################
# IMPORT
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.activations import relu, sigmoid
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt

################################################################################
print(tf.__version__)

NUM_WORDS = 10000
num_epochs = 20
batch_size = 512

# load dataset
imdb = keras.datasets.imdb
(train_feat, train_labels), (test_feat, test_labels) = imdb.load_data(num_words=NUM_WORDS)


def multi_hot_sequences(sequences, dimension):
    # create all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros(shape=(len(sequences), dimension))

    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s

    return results


train_feat = multi_hot_sequences(train_feat, dimension=NUM_WORDS)
test_feat = multi_hot_sequences(test_feat, dimension=NUM_WORDS)

################################################################################
# baseline model with Dense layers
baseline_model = Sequential()
baseline_model.add(Dense(16, activation=relu, input_shape=(NUM_WORDS,)))
baseline_model.add(Dense(16, activation=relu))
baseline_model.add(Dense(1, activation=sigmoid))

baseline_model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"]
)

baseline_model.summary()

baseline_history = baseline_model.fit(
    train_feat,
    train_labels,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(test_feat, test_labels),
    verbose=2
)
################################################################################
# smaller model with less hidden units
smaller_model = Sequential()
smaller_model.add(Dense(4, activation=relu, input_shape=(NUM_WORDS,)))
smaller_model.add(Dense(4, activation=relu))
smaller_model.add(Dense(1, activation=sigmoid))

smaller_model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"]
)

smaller_model.summary()

smaller_history = smaller_model.fit(
    train_feat,
    train_labels,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(test_feat, test_labels),
    verbose=2
)
################################################################################
# bigger model to showcase overfitting
bigger_model = Sequential()
bigger_model.add(Dense(512, activation=relu, input_shape=(NUM_WORDS,)))
bigger_model.add(Dense(512, activation=relu))
bigger_model.add(Dense(1, activation=sigmoid))

bigger_model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"]
)

bigger_model.summary()

bigger_history = bigger_model.fit(
    train_feat,
    train_labels,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(test_feat, test_labels),
    verbose=2
)

################################################################################
# Weight Regularization
# L2 weight regularization
L2_model = Sequential()
L2_model.add(Dense(16, kernel_regularizer=l2(0.001), activation=relu, input_shape=(NUM_WORDS,)))
L2_model.add(Dense(16, kernel_regularizer=l2(0.001), activation=relu))
L2_model.add(Dense(1, activation=sigmoid))

L2_model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"]
)

L2_model.summary()

L2_history = L2_model.fit(
    train_feat,
    train_labels,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(test_feat, test_labels),
    verbose=2
)

################################################################################
# Dropout
dropout_model = Sequential()
dropout_model.add(Dense(16, activation=relu, input_shape=(NUM_WORDS,)))
dropout_model.add(Dropout(0.5))
dropout_model.add(Dense(16, activation=relu))
dropout_model.add(Dropout(0.5))
dropout_model.add(Dense(1, activation=sigmoid))

dropout_model.compile(
    optimizer=Adam(),
    loss="binary_crossentropy",
    metrics=["accuracy", "binary_crossentropy"]
)

dropout_model.summary()

dropout_history = dropout_model.fit(
    train_feat,
    train_labels,
    epochs=num_epochs,
    batch_size=batch_size,
    validation_data=(test_feat, test_labels),
    verbose=2
)


################################################################################
# Plots
def plot_history(histories, key="binary_crossentropy"):
    plt.figure(figsize=(10, 10))

    for name, history in histories:
        val = plt.plot(
            history.epoch,
            history.history["val_" + key],
            "--",
            label=name.title() + " Val"
        )

        plt.plot(
            history.epoch,
            history.history[key],
            color=val[0].get_color(),
            label=name.title() + " Train"
        )

    plt.xlabel("Epochs")
    plt.ylabel(key.replace("_", " ").title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.show()


plot_history([
    ("baseline", baseline_history),
    ("smaller", smaller_history),
    ("bigger", bigger_history),
    ("L2", L2_history),
    ("dropout", dropout_history)
])
