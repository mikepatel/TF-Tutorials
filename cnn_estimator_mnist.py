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

import tensorflow as tf
from tensorflow.layers import conv2d, max_pooling2d, dense

tf.logging.set_verbosity(tf.logging.INFO)

################################################################################
# build model


################################################################################
if __name__ == "__main__":
    tf.app.run()

