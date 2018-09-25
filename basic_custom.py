# 9/25/18

# Basic Custom Training
# low-level APIs (not tf.keras)
# https://www.tensorflow.org/tutorials/eager/custom_training

# Fitting a linear model

# dataset:

# Notes:
# as model trains, the same code to compute predictions should behave differently over time
# Variable object stores values
#   - computations using Variables are automatically traced when computing gradients
#   - mutable

################################################################################
# IMPORTs
import tensorflow as tf
import matplotlib.pyplot as plt

tfe = tf.contrib.eager
tf.enable_eager_execution()

################################################################################
print(tf.__version__)
print("GPU available: " + str(tf.test.is_gpu_available()))


################################################################################
# build model
# f(x) = Wx + b (linear model)
class Model(object):
    def __init__(self):
        # ideally initialize W, b parameters with random values
        self.W = tfe.Variable(5.0)  # initialize to 5.0
        self.b = tfe.Variable(0.0)  # initialize to 0.0

    def __call__(self, x):
        y = self.W * x + self.b
        return y


model = Model()  # instantiate
assert model(3.0).numpy() == 15.0


################################################################################
# define loss function
# L2 loss
def loss(pred_y, actual_y):
    loss = tf.reduce_mean(tf.square(pred_y - actual_y))
    return loss


################################################################################
# obtain training data
TRUE_W = 3.0
TRUE_b = 2.0
NUM_EXAMPLES = 1000

inputs = tf.random_normal(shape=[NUM_EXAMPLES])
noise = tf.random_normal(shape=[NUM_EXAMPLES])
outputs = inputs * TRUE_W + TRUE_b + noise

################################################################################
# visualization
before_preds = model(inputs)

#plt.scatter(inputs, model(inputs), c="r")
#plt.scatter(inputs, outputs, c="b")
#plt.title("Before any training: Ground truth vs Model predictions")
#plt.show()

print("Current loss: " + str(loss(model(inputs), outputs).numpy()))


################################################################################
# run training and optimize
def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as t:
        current_loss = loss(model(inputs), outputs)

    dW, db = t.gradient(current_loss, [model.W, model.b])
    model.W.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


model = Model()

# collect history of W-values and b-values (in order to plot)
Ws = []
bs = []
num_epochs = range(50)
lr = 0.1  # learning rate

for epoch in num_epochs:
    Ws.append(model.W.numpy())
    bs.append(model.b.numpy())

    current_loss = loss(model(inputs), outputs)

    train(model, inputs, outputs, learning_rate=lr)

    print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" %
          (epoch, Ws[-1], bs[-1], current_loss)
          )
################################################################################
# PLOTs
# Plot W and b curves
plt.plot(num_epochs, Ws, "r")
plt.plot(num_epochs, bs, "b")
plt.plot([TRUE_W]*len(num_epochs), "r--")
plt.plot([TRUE_b]*len(num_epochs), "b--")
plt.legend(["W", "b", "true W", "true b"])
plt.show()

# Plot before and after side-by-side
plt.subplot(2, 1, 1)
plt.scatter(inputs, before_preds, c="r")
plt.scatter(inputs, outputs, c="b")
plt.title("Before any training: Ground truth vs Model predictions")

plt.subplot(2, 1, 2)
plt.scatter(inputs, model(inputs), c="r")
plt.scatter(inputs, outputs, c="b")
plt.title("After training: Ground truth vs Model predictions")

plt.show()
