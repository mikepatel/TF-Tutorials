# TF tutorial
# 8/13/18

# Basic Image Classification
# using tf.keras
# https://www.tensorflow.org/tutorials/keras/basic_classification

# dataset: Fashion MNIST
# training set size: 60k
# test set size: 10k
# 28x28x1
# 10 class labels (articles of clothing)
# relatively small dataset

################################################################################
# IMPORTs
import tensorflow as tf
from tensorflow import keras  # tf.keras
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.activations import relu, softmax
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

################################################################################
# SETUP
print(tf.__version__)

# load dataset
# returns 4 NumPy arrays
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# class labels
class_names = [
    "T-shirt", "Trouser", "Jumper", "Dress", "Coat",
    "Sandal", "Dress Shirt", "Sneaker", "Bag", "Ankle Boot"
]  # correspond to class labels 0-9

'''
# exploring the data
print(train_images.shape)  # (60k, 28, 28)
print(len(train_labels))  # 60k labels
print(train_labels)  # each label is integer 0-9
print(test_images.shape)  # (10k, 28, 28)

plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.gca().grid(False)
plt.show()
'''

# preprocessing the data
# scale pixel values between 0 and 1
train_images = train_images / 255.0
test_images = test_images / 255.0
# display first 25 images from training set to verify correctness before
# feeding into NN
'''
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid("off")
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''

################################################################################
# building the model
model = keras.Sequential()
model.add(Flatten(input_shape=(28, 28)))  # transform from 2D array of 28x28 to 1D array of 784 pixels
model.add(Dense(128, activation=relu))
model.add(Dense(10, activation=softmax))

# compiling the model
model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=Adam(),
    metrics=["accuracy"]
)
################################################################################
# training the model
model.fit(train_images, train_labels, epochs=10)

# evaluating accuracy
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test accuracy: ", test_accuracy)

# making predictions
predictions = model.predict(test_images)
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid("off")
    plt.imshow(test_images[i], cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]

    if predicted_label == true_label:
        color = "green"
    else:
        color = "red"

    plt.xlabel("{} ({})".format(class_names[predicted_label],
                                class_names[true_label]),
               color=color)
plt.show()
