import tensorflow as tf
import random
import csv


def csv_read(name):
  rows = []
  with open(name) as f:
    for row in csv.reader(f):
      rows.append(row)
  return rows

def list_split(lst, per):
  i = int(per*len(lst))
  return (lst[:i], lst[i+1:])

def ann_layer(x, size):
  w = tf.Variable(tf.truncated_normal(size))
  b = tf.Variable(tf.truncated_normal(size[-1]))
  y = tf.add(tf.matmul(x, w), b)
  return tf.nn.relu(y)

def ann_network(x, size):
  l = ann_layer(x, size[0:2])
  for i in range(1, len(size)-1):
    l = ann_layer(l, size[i:i+2])
  return l

def ann_train(outp, labl, rate):
  cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outp, labels=labl))
  return tf.train.GradientDescentOptimizer(rate).minimize(cost)


def get_data(name, train_per):
  rows = []
  for row in csv_read(name)[1:]:
    rows.append({'x': row[:-1], 'y_': row[-1:]})
  random.shuffle(rows)
  return list_split(rows, train_per)



(rate, epochs) = (0.3, 1000)
(trains, tests) = get_data('sonar.csv', 0.8)
inps = len(trains[0].x)
outs = len(trains[0].y_)
hids = int(0.5*(inps+outs))
size = [inps, hids, outs]

x = tf.placeholder(tf.float32, inps)
y_ = tf.placeholder(tf.float32, outs)
y = ann_network(x, size)
step = ann_train(y, y_, rate)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(epochs):
  for train in trains:
    sess.run(step, train)
  

