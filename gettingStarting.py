#!/usr/bin/env python
# Filename: gettingStarting 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 August, 2017
"""


import tensorflow as tf


# The Computational Graph

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly

print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ",sess.run(node3))


# placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)

adder_node = a + b  # + provides a shortcut for tf.add(a, b)
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))


add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b:4.5}))


# Variables

W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b
init = tf.global_variables_initializer()
# It is important to realize init is a handle to the TensorFlow sub-graph that initializes all the global variables.
# Until we call sess.run, the variables are uninitialized.
sess.run(init)
print(sess.run(linear_model,{x:[1,2,3,4]}))


# loss function
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4],y:[0,-1,-2,-3]}))

# assign the value of W and b
fixW = tf.assign(W,[-1.])
fixb = tf.assign(b,[1.])
sess.run([fixW,fixb])
print(sess.run(loss,{x:[1,2,3,4],y:[0,-1,-2,-3]}))


# tf.train API
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init)  # reset values to incorrect defaults
x_train=[1,2,3,4]
y_train=[0,-1,-2,-3]
for i in range(1000):
    print('iter %d'%i,sess.run([W,b]))
    sess.run(train,{x:x_train,y:y_train})

print(sess.run([W,b]))

# evaluate training accuracy
curr_W, curr_b, curr_loss = sess.run([W,b,loss],{x:x_train,y:y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

########################################################################
# using tf.contrib.learn

# NumPy is often used to load, manipulate and preprocess data.
import numpy as np

# Declare list of features. We only have one real-valued feature. There are many
# other types of columns that are more complicated and useful.
features = [tf.contrib.layers.real_valued_column('x',dimension=1)]


# An estimator is the front end to invoke training (fitting) and evaluation
# (inference). There are many predefined types like linear regression,
# logistic regression, linear classification, logistic classification, and
# many neural network classifiers and regressors. The following code
# provides an estimator that does linear regression.
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

# TensorFlow provides many helper methods to read and set up data sets.
# Here we use two data sets: one for training and one for evaluation
# We have to tell the function how many batches
# of data (num_epochs) we want and how big each batch should be.
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_train}, y_train,
                                              batch_size=4,
                                              num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
    {"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)

# We can invoke 1000 training steps by invoking the  method and passing the
# training data set.
estimator.fit(input_fn=input_fn, steps=1000)

# Here we evaluate how well our model did.
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)

########################################################################
# A custom model
# Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  # Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  # Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  # Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  # ModelFnOps connects subgraphs we built to the
  # appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)
# define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, 4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
# Here we evaluate how well our model did.
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)