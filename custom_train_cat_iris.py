# 9/29/18

# Custom Training
# using Datasets API
# using tf.keras
# using eager execution (TF must be 1.8+)
# https://www.tensorflow.org/tutorials/eager/custom_training_walkthrough

# classify not by image, but by flower measurements

# dataset: Iris
# size: 120 flowers
# 4 features: sepal_length, sepal_width, petal_length, petal_width
# class label: species (3 species types)
# class label = {0, 1, 2}

# Notes:
#   - build model, train, predict
#   - eager execution -> evaluate operations immediately instead of creating
#           a computational graph that is later executed

################################################################################
# IMPORTs
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu

import os
import matplotlib.pyplot as plt

tf.enable_eager_execution()

################################################################################
print("TF version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

################################################################################
# load dataset
train_dataset_url = "http://download.tensorflow.org/data/iris_training.csv"
train_dataset_filepath = tf.keras.utils.get_file(
    fname=os.path.basename(train_dataset_url),
    origin=train_dataset_url
)
print("Local copy of dataset file: {}\n".format(train_dataset_filepath))

# explore dataset
column_names = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species"
]

feature_names = column_names[:-1]
label_name = column_names[-1]

class_names = ["Iris setosa", "Iris versicolor", "Iris virginica"]

# feeding data into a model -> Dataset API
# create tf.data.Dataset object
# read and then transform data into a usable form for training
BATCH_SIZE = 32
train_dataset = tf.contrib.data.make_csv_dataset(
    train_dataset_filepath,
    batch_size=BATCH_SIZE,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1
)
# train_dataset object is (features, label) pairs where features is a dictionary: {"feature_name" : value}


# repackage features dictionary into a single array with shape: (batch_size, num_features)
# tf.stack -> from list of tensors, create a combined tensor with specified dimension
def pack_features_vector(features, labels):
    features = tf.stack(list(features.values()), axis=1)
    return features, labels


train_dataset = train_dataset.map(pack_features_vector)
features, labels = next(iter(train_dataset))
################################################################################
# build model
model = Sequential()
model.add(Dense(10, activation=relu, input_shape=(4,)))
model.add(Dense(10, activation=relu))
model.add(Dense(3))  # output layer

model.summary()


################################################################################
# train model
# loss function
def loss_fn(model, x, y):
    y_pred = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_pred)


# tf.GradientTape -> calculate gradients
def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss_fn(model, inputs, targets)

    return loss_value, tape.gradient(loss_value, model.trainable_variables)


# optimizer
learning_rate = 0.01
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
global_step = tf.train.get_or_create_global_step()

# training loop
train_loss_results = []
train_accuracy_results = []

NUM_EPOCHS = 251

for epoch in range(NUM_EPOCHS):
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    for feat, label in train_dataset:
        # optimize model
        loss_value, gradients = grad(model, feat, label)
        optimizer.apply_gradients(zip(gradients, model.variables), global_step)

        epoch_loss_avg(loss_value)  # add current batch loss
        pred = tf.argmax(model(feat), axis=1, output_type=tf.int32)
        epoch_accuracy(pred, label)  # compare actual vs predicted

    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d} : Loss: {:.3f}, Accuracy: {:03}".format(epoch, epoch_loss_avg.result(), epoch_accuracy.result()))

################################################################################
# Visualization of training
fig, axes = plt.subplots(2, sharex=True)
fig.suptitle("Training Metrics")

axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].plot(train_loss_results)

axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].plot(train_accuracy_results)

plt.show()

################################################################################
# Evaluation and prediction
# on test set
test_url = "http://download.tensorflow.org/data/iris_test.csv"
test_filepath = tf.keras.utils.get_file(
    fname=os.path.basename(test_url),
    origin=test_url
)

test_dataset = tf.contrib.data.make_csv_dataset(
    test_filepath,
    batch_size=BATCH_SIZE,
    column_names=column_names,
    label_name="species",
    num_epochs=1,
    shuffle=False
)

test_dataset = test_dataset.map(pack_features_vector)

test_accuracy = tfe.metrics.Accuracy()

for f, l in test_dataset:
    logits = model(f)
    pred = tf.argmax(logits, axis=1, output_type=tf.int32)
    test_accuracy(pred, l)

print("Test accuracy: {:.3%}".format(test_accuracy.result()))
