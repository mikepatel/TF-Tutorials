# TF tutorial
# 8/31/18

# Saving and Restoring Models
# using tf.keras
# https://www.tensorflow.org/tutorials/keras/save_and_restore_models

# Callbacks and Checkpoints

# dataset: MNIST
# training set size: 60k
# test set size: 10k

# Notes:
# saving training progress during and after
################################################################################
# IMPORT
from __future__ import absolute_import, division, print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import pathlib

################################################################################
print(tf.__version__)

# load dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# only using the first 1000 examples
train_images = train_images[:1000]
train_labels = train_labels[:1000]
test_images = test_images[:1000]
test_labels = test_labels[:1000]

# reshape
train_images = train_images.reshape(-1, 28*28) / 255.0
test_images = test_images.reshape(-1, 28*28) / 255.0


################################################################################
# Model definition
def build_model():
    m = Sequential()
    m.add(Dense(512, activation=relu, input_shape=(784, )))
    m.add(Dropout(0.2))
    m.add(Dense(10, activation=softmax))

    m.compile(
        optimizer=Adam(),
        loss=keras.losses.sparse_categorical_crossentropy,
        metrics=["accuracy"]
    )

    return m


################################################################################
# Using callbacks
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# create checkpoint callback
cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    verbose=1
)

model = build_model()
model.fit(
    train_images,
    train_labels,
    epochs=10,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback]  # pass callback to training
)

################################################################################
# create a different instantiation of model architecture
# can share weights since same architecture
# model instantiation is NOT trained, but will use to evaluate on test set
model = build_model()

loss, accuracy = model.evaluate(test_images, test_labels)
print("Untrained model, accuracy: {:5.2f}%".format(100*accuracy))

# loading checkpoint weights and re-evaluating
model.load_weights(checkpoint_path)
loss, accuracy = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*accuracy))

################################################################################
# use uniquely-saved checkpoints and checkpoint frequencies
# include epoch in filename
checkpoint_path = "training_2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = ModelCheckpoint(
    filepath=checkpoint_path,
    save_weights_only=True,
    period=5,  # save weights every 5 epochs
    verbose=1
)

model = build_model()  # new instantiation to train
model.fit(
    train_images,
    train_labels,
    epochs=50,
    validation_data=(test_images, test_labels),
    callbacks=[cp_callback],
    verbose=0
)

# look at checkpoints
# Sort the checkpoints by modification time.
checkpoints = pathlib.Path(checkpoint_dir).glob("*.index")
checkpoints = sorted(checkpoints, key=lambda cp: cp.stat().st_mtime)
checkpoints = [cp.with_suffix('') for cp in checkpoints]
latest = str(checkpoints[-1])
checkpoints

# to test, reset model and load latest checkpoint
model = build_model()
model.load_weights(latest)
loss, accuracy = model.evaluate(test_images, test_labels)
print("Restore model, accuracy: {:5.2f}%".format(100*accuracy))

################################################################################
# manually saving weights
model.save_weights("./checkpoints/my_checkpoint")

# restore weights
model = build_model()
model.load_weights("./checkpoints/my_checkpoint")

loss, accuracy = model.evaluate(test_images, test_labels)
print("Restore model, accuracy: {:5.2f}%".format(100*accuracy))

################################################################################
# save the entire model
# save weights, model architecture, optimizer configuration
# can continue training in web browser (TF.js)
model = build_model()
model.fit(
    train_images,
    train_labels,
    epochs=5
)

# save entire model
model.save("my_model.h5")

# recreate that exact model
new_model = keras.models.load_model("my_model.h5")
new_model.summary()
loss, accuracy = new_model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*accuracy))
