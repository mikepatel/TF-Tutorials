# 9/23/18

# Tutorial that covers:
# using eager execution
# using GPU acceleration
# using tensors
# https://www.tensorflow.org/tutorials/eager/eager_basics

# dataset:
#

# Notes:
# tensor = multi-dimensional array that have data type 'dtype' and shape 'shape'
# tensor vs numpy array (ndarray)
#   - tensors can be backed by accelerator memory (GPU, TPU)
#   - tensors are immutable
# explicit device placement on CPU or GPU
# tf.data.Dataset API -> build pipelines to feed data into model
#   - with eager execution enabled, no need to create tf.data.Iterator object - just use Python iteration

################################################################################
# IMPORTs
import tensorflow as tf


tf.enable_eager_execution()

################################################################################
print(tf.__version__)

print("GPU available: " + str(tf.test.is_gpu_available()))

