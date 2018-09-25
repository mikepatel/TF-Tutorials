# 9/24/18

# Automatic Differentiation
# using eager execution
# using GPU acceleration
# https://www.tensorflow.org/tutorials/eager/automatic_differentiation

# dataset:
#

# Notes:
# automatic differentiation = technique to numerically evaluate function derivatives
#   - aids in optimizing ML models
# gradient tapes = all operations applied to compute output of the function

################################################################################
# IMPORTs
import tensorflow as tf
import matplotlib.pyplot as plt

tf.enable_eager_execution()

################################################################################
print(tf.__version__)
tfe = tf.contrib.eager


def fn(x):
    return tf.square(x)


def gradient(fn):
    return lambda x: tfe.gradients_function(fn)(x)[0]


a = float(-10)
b = float(10)
num_points = 1000
x = tf.linspace(a, b, num_points)  # number of points btwn a and b

# PLOT
plt.plot(x, fn(x), label="f")  # x^2
plt.plot(x, gradient(fn)(x), label="1st derivative")  # 2x
plt.plot(x, gradient(gradient(fn))(x), label="2nd derivative")  # 2
plt.xlim(a, b)
plt.ylim(a, b)
plt.grid()
plt.legend()
plt.show()
