# TF tutorial
# 8/17/18

# Regression
# using tf.keras
# https://www.tensorflow.org/tutorials/keras/basic_regression

# predicting house prices

# dataset: Boston Housing
# training set size: 404
# test set size: 102
# labels are house prices ($k)

# Notes:
# regression-based, not classification
# very small dataset
# loss function of Mean Squared Error (MSE) useful for regression problems
# evaluation metric Mean Absolute Error (MAE) useful for regression problems
# since dataset is small, use small NN to prevent overfitting
# early stopping also useful to prevent overfitting

################################################################################
# IMPORT
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import relu
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

################################################################################
# DATA
print("TF version: ", tf.__version__)

# load dataset
boston_housing = keras.datasets.boston_housing
(train_data, train_labels), (test_data, test_labels) = boston_housing.load_data()

# shuffle training set
order = np.argsort(np.random.random(train_labels.shape))
train_data = train_data[order]
train_labels = train_labels[order]

print("Training set: {}".format(train_data.shape))  # 404 examples, 13 features
print("Testing set: {}".format(test_data.shape))  # 102 examples, 13 features
# features include: crime rate, property tax rate, pupil-teacher ratio...
# features have different scales -> explore and preprocess data

# exploring the data
column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX",
    "RM", "AGE", "DIS", "RAD", "TAX",
    "PTRATIO", "B", "LSTAT"
]
df = pd.DataFrame(train_data, columns=column_names)  # dataframe
#print(df.head())

# normalize features b/c different scales
# normalization should make training easier and
# less dependent on choice of input units
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
train_data = (train_data - mean) / std
test_data = (test_data - mean) / std


################################################################################
# BUILD MODEL
# create a build_model function to use multiple times
def build_model():
    m = Sequential()
    m.add(Dense(64, activation=relu, input_shape=(train_data.shape[1],)))
    m.add(Dense(64, activation=relu))
    m.add(Dense(1))  # output a single, continuous value

    # configure learning
    m.compile(
        loss="mse",
        optimizer=RMSprop(0.001),
        metrics=["mae"]
    )

    return m


# instantiate a model
model = build_model()
model.summary
################################################################################
# USE MODEL
# training the model
num_epochs = 300
early_stop = EarlyStopping(monitor="val_loss", patience=20)
history = model.fit(
    train_data,
    train_labels,
    epochs=num_epochs,
    validation_split=0.2,
    verbose=1,
    callbacks=[early_stop]  # automatically stop training when validation score doesn't improve
)

# visualizing the training
# use history data object to determine how long to train BEFORE
# the model stops making substantial gains
train_loss = history.history["mean_absolute_error"]
valid_loss = history.history["val_mean_absolute_error"]
plt.plot(history.epoch, np.array(train_loss), label="Training Loss")
plt.plot(history.epoch, np.array(valid_loss), label="Validation Loss")
plt.xlabel("Number of Epochs")
plt.ylabel("Mean Abs Error [1000$")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# predictions
test_loss, test_mae = model.evaluate(test_data, test_labels)
print("Test set Mean Abs Error: ${:7.2f}".format(test_mae * 1000))
test_predictions = model.predict(test_data).flatten()
print(test_predictions)