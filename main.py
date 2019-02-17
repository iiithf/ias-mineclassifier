from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import numpy as np


def ann_layer(x, size):
  w = tf.Variable(tf.truncated_normal(size))
  b = tf.Variable(tf.truncated_normal(size[-1:]))
  return tf.add(tf.matmul(x, w), b)

def ann_network(x):
  h1 = tf.nn.relu(ann_layer(x, [60, 60]))
  h2 = tf.nn.sigmoid(ann_layer(h1, [60, 60]))
  h3 = tf.nn.sigmoid(ann_layer(h2, [60, 60]))
  h4 = tf.nn.relu(ann_layer(h3, [60, 60]))
  return ann_layer(h4, [60, 2])


def get_data(name, test_per):
  d = pd.read_csv(name)
  x = d[d.columns[:-1]].values
  y = d[d.columns[-2:]].values.astype(float)
  for i in range(y.shape[0]):
    y[i][0] = 1 - y[i][1]
  x, y = shuffle(x, y)
  return train_test_split(x, y, test_size=test_per)


rate, epochs = (0.3, 1000)
train_x, test_x, train_y, test_y = get_data('sonar.csv', 0.2)
inps, outs = (len(train_x[0]), 1)
hids = int(0.5*(inps+outs))
print('train_x.shape:', train_x.shape)
print('train_y.shape:', train_y.shape)
print('test_x.shape:', test_x.shape)
print('test_y.shape:', test_y.shape)

x = tf.placeholder(tf.float32, [None, inps])
y_ = tf.placeholder(tf.float32, [None, outs])
y = ann_network(x)
cost_func = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.GradientDescentOptimizer(rate).minimize(cost_func)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
# savr = tf.train.Saver()
for epoch in range(epochs):
  sess.run(train_step, {x: train_x, y_: train_y})
  cost = sess.run(cost_func, {x: train_x, y_: train_y})
  # pred = tf.equal(y, y_)
  # accr = tf.reduce_mean(tf.cast(pred, tf.float32))
  # accr_v = sess.run(accr, {x: train_x, y_: train_y})
  # print('Epoch %d: %f accuracy' % (epoch, accr_v))
# savr.save(sess, 'sonar')
