#!/usr/bin/env python
# Filename: mnist_tutorial.py 
"""
introduction:

authors: Huang Lingcao
email:huanglingcao@gmail.com
add time: 04 August, 2017
"""

import tensorflow as tf

# Load MNIST Data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)





x = tf.placeholder(tf.float32,shape=[None,784])
y_ = tf.placeholder(tf.float32,shape=[None,10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

####################################################################################
# a shallow model

# # Start TensorFlow InteractiveSession
# sess = tf.InteractiveSession()
#
# # Before Variables can be used within a session, they must be initialized using that session.
# sess.run(tf.global_variables_initializer())

# # predicted class and loss function
# y = tf.matmul(x,W) + b
#
# # loss function
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
#
# # train
# train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#
# for _ in range(1000):
#     batch = mnist.train.next_batch(100)
#     train_step.run(feed_dict={x:batch[0],y_:batch[1]})
#
# # Evaluate the Model
# correct_prediction = tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
#
# accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
#
# print(accuracy.eval(feed_dict={x:mnist.test.images, y_:mnist.test.labels}))

########################################################################################
# Build a Multilayer Convolutional Network
def weight_variables(shape):
    initial = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)

def bias_variables(shape):
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)

def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# first convolutional layer
W_conv1 = weight_variables([5,5,1,32])
b_conv1 = bias_variables([32])

#To apply the layer, we first reshape x to a 4d tensor, with the second and third dimensions corresponding to image width and height,
#  and the final dimension corresponding to the number of color channels.
x_image = tf.reshape(x,[-1,28,28,1])

h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second Convolutional layer
W_conv2 = weight_variables([5,5,32,64])
b_conv2 = bias_variables([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer
# Now that the image size has been reduced to 7x7,
# we add a fully-connected layer with 1024 neurons to allow processing on the entire image.
#  We reshape the tensor from the pooling layer into a batch of vectors,
# multiply by a weight matrix, add a bias, and apply a ReLU.

W_fc1 = weight_variables([7*7*64,1024])
b_fc1 = bias_variables([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

# dropout (avoiding overfitting)
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# Readout layer
W_fc2 = weight_variables([1024,10])
b_fc2 = bias_variables([10])

y_conv = tf.matmul(h_fc1_drop,W_fc2) + b_fc2

# Evaluate the Model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1),tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        batch = mnist.train.next_batch(50)
        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x:batch[0],y_:batch[1],keep_prob:1.0})
            print('step %d, training accuracy %g '%(i, train_accuracy))

            # test_accuracy = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
            # print('test accuracy %g' % test_accuracy)

        train_step.run(feed_dict={x:batch[0],y_:batch[1],keep_prob:0.5})

    test_accuracy = accuracy.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels, keep_prob:1.0})
    print('test accuracy %g'%test_accuracy )



# The final test set accuracy after running this code should be approximately 99.2