from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd


def read_dataset():
  d = pd.read_csv('sonar.csv')
  X = d[d.columns[:-1]].values
  Y = d[d.columns[-1:]].values.astype(float)
  print("X.shape", X.shape)
  return (X,Y)

# Read the dataset
X, Y = read_dataset()

# Shuffle the dataset to mix up the rows
X, Y = shuffle(X, Y)

# Convert the dataset into train and test datasets
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20)

# Inspect the shape of the train and test datasets
print("train_x.shape",train_x.shape)
print("train_y.shape",train_y.shape)
print("test_x.shape",test_x.shape)
print("test_y.shape",test_y.shape)

# Define the hyperparameters
learning_rate = 0.3
epochs = 1000
n_dim = X.shape[1] 
n_class = 1

# Inputs and outputs
x = tf.placeholder(tf.float32, [None, n_dim])
y_ = tf.placeholder(tf.float32, [None, n_class])


def ann_layer(x, size):
  w = tf.Variable(tf.truncated_normal(size))
  b = tf.Variable(tf.truncated_normal(size[-1:]))
  return tf.add(tf.matmul(x, w), b)

# Model
def multilayer_perceptron(x):
    l1 = tf.nn.relu(ann_layer(x, [60, 60]))
    l2 = tf.nn.sigmoid(ann_layer(l1, [60, 60]))
    l3 = tf.nn.sigmoid(ann_layer(l2, [60, 60]))
    l4 = tf.nn.relu(ann_layer(l3, [60, 60]))
    return ann_layer(l4, [60, 1])


# Initialization
y = multilayer_perceptron(x)
cost_function = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(epochs):
    sess.run(train_step, feed_dict={x:train_x, y_:train_y})
    pred = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accr = tf.reduce_mean(tf.cast(pred, tf.float32))
    pred_y = sess.run(y,feed_dict={x:test_x} )
    accr = (sess.run(accr,feed_dict={x:train_x, y_:train_y}))
    print('epoch: ', epoch,' - ', " accuracy: ", accr)
