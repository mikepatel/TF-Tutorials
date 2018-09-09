# TF Tutorial
# 9/7/18

# using eager execution
# using tf.keras
# customized model building

# dataset: Iris
# - 120 examples
# - 4 features
# - 3 possible class labels (0: setosa, 1: versicolor, 2: virginica)

# Notes:

################################################################################
# IMPORTs
from __future__ import absolute_import, division, print_function
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu
import tensorflow.contrib.eager as tfe

################################################################################
# enable Eager Execution to return values directly instead of creating and then running a computation graph
tf.enable_eager_execution()

################################################################################
print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

################################################################################
# download Iris dataset
url = "http://download.tensorflow.org/data/iris_training.csv"
train_dataset_fp = tf.keras.utils.get_file(
    fname=os.path.basename(url),
    origin=url
)
print("Local copy of the dataset file: {}".format(train_dataset_fp))  # CSV file


# parse the CSV file
def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]]  # sets field types
    parsed_line = tf.decode_csv(line, example_defaults)

    # First 4 fields are features, combine into single tensor
    features = tf.reshape(parsed_line[:-1], shape=(4,))

    # Last field is the label
    label = tf.reshape(parsed_line[-1], shape=())

    return features, label  # features tensor, and label tensor


# features => MODEL => class label
# feed data into the model - one of the most important steps in ML and data science
train_dataset = tf.data.TextLineDataset(train_dataset_fp)
train_dataset = train_dataset.skip(1)             # skip the first header row
train_dataset = train_dataset.map(parse_csv)      # parse each row/line in CSV file
train_dataset = train_dataset.shuffle(buffer_size=1000)  # randomize
train_dataset = train_dataset.batch(32)  # 32 examples to train at once

# View a single example entry from a batch
features, label = iter(train_dataset).next()
print("example features:", features[0])
print("example label:", label[0])


################################################################################
# use Keras API to build models and layers
# consider model as a way to organize layers
def build_model():
    model = Sequential()
    model.add(Dense(10, activation=relu, input_shape=(4, )))  # input shape required,
    #  other layers will perform automatic shape inference
    model.add(Dense(10, activation=relu))
    model.add(Dense(3))  # output layer

    model.summary()

    return model


model = build_model()


################################################################################
# train model
def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


# optimization
learning_rate = 0.01
print("learning rate: ", learning_rate)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)

################################################################################
# training loop
# keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 701  # epoch = pass through dataset

for epoch in range(num_epochs):  # in each epoch, iterate through examples
    epoch_loss_avg = tfe.metrics.Mean()
    epoch_accuracy = tfe.metrics.Accuracy()

    # Training loop - using batches of 32
    for x, y in train_dataset:
        # Optimize the model
        grads = grad(model, x, y)
        optimizer.apply_gradients(zip(grads, model.variables),
                                  global_step=tf.train.get_or_create_global_step())

        # Track progress
        epoch_loss_avg(loss(model, x, y))  # add current batch loss
        # compare predicted label to actual label
        epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

    # end epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())

    if epoch % 50 == 0:
        print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))

################################################################################
# Visualization of training
fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
fig.suptitle('Training Metrics')

axes[0].set_ylabel("Loss", fontsize=14)
axes[0].plot(train_loss_results)

axes[1].set_ylabel("Accuracy", fontsize=14)
axes[1].set_xlabel("Epoch", fontsize=14)
axes[1].plot(train_accuracy_results)

plt.show()
