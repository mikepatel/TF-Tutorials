# 9/30/18

# CNN using Estimators
# using tf.layers
# Estimator API
# https://www.tensorflow.org/tutorials/estimators/cnn

# recognize handwritten digits

# dataset: MNIST
# training size: 60k
# testing size: 10k
# class labels: 10 (0-9)
# 28x28 monochrome images

# Notes:

# Architecture:
#   1. Conv1 (32 5x5 filters, with ReLU)
#   2. Pooling1 (max pooling with 2x2 filters, stride=2)
#   3. Conv2 (64 5x5 filters, with ReLU)
#   4. Pooling2 (max pooling with 2x2 filters, stride=2)
#   5. Dense1 (1024 neurons with dropout)
#   6. Dense2 (10 neurons, output layer)

################################################################################
# IMPORTs
from __future__ import absolute_import, division, print_function

import numpy as np
import os

import tensorflow as tf
from tensorflow.layers import conv2d, max_pooling2d, dense
from tensorflow.nn import relu, softmax
from tensorflow.losses import sparse_softmax_cross_entropy
tf.logging.set_verbosity(tf.logging.INFO)


################################################################################
# build model
def cnn_model_fn(features, labels, mode):
    # shape = [batch_size, image_height, image_width, channels]

    # Input layer
    # want BATCH_SIZE to be a hyperparameter, so
    # set batch_size = -1 to do automatic shape inferring
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])

    # Convolutional layer 1
    conv1 = conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=relu
    )

    # Pooling layer 1
    pool1 = max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )

    # Convolutional layer 2
    conv2 = conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=relu
    )

    # Pooling layer 2
    pool2 = max_pooling2d(
        inputs=conv2,
        pool_size=[2, 2],
        strides=2
    )

    # reshape
    pool2_flat = tf.reshape(
        pool2,
        [-1, 7*7*64]
    )

    # Dense layer 1
    dense1 = dense(
        inputs=pool2_flat,
        units=1024,
        activation=relu
    )

    # dropout
    dropout = tf.layers.dropout(
        inputs=dense1,
        rate=0.4,  # 40% randomly dropped
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # Output layer
    logits = dense(
        inputs=dropout,
        units=10
    )

    ###############
    predictions = {
        "classes" : tf.argmax(
            input=logits,
            axis=1
        ),
        "probabilities" : softmax(
            logits=logits,
            name="softmax_tensor"
        )
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Loss function
    loss = sparse_softmax_cross_entropy(
        labels=labels,
        logits=logits
    )

    #
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step()
        )

        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # evaluation metrics
    eval_metric_ops = {
        "accuracy" : tf.metrics.accuracy(
            labels=labels,
            predictions=predictions["classes"]
        )
    }

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


################################################################################
def main(unused_argv):
    # load dataset
    (training_data, training_labels), (eval_data, eval_labels) = tf.keras.datasets.mnist.load_data()
    training_data = np.asarray(training_data, dtype=np.float32)
    training_labels = np.asarray(training_labels, dtype=np.int32)
    eval_data = np.asarray(eval_data, dtype=np.float32)
    eval_labels = np.asarray(eval_labels, dtype=np.int32)

    # create Estimator
    mnist_classifier = tf.estimator.Estimator(
        model_fn=cnn_model_fn,
        model_dir=str(os.getcwd() + "/mnist_convnet_model")  # where checkpoints are saved
    )

    # logging
    tensors_to_log = {"probabilities" : "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    # train model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x" : training_data},
        y=training_labels,
        batch_size=128,
        num_epochs=None,
        shuffle=True
    )

    mnist_classifier.train(
        input_fn=train_input_fn,
        steps=20000,
        hooks=[logging_hook]
    )

    # evaluate model
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x" : eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )

    eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)


################################################################################
if __name__ == "__main__":
    tf.app.run()
