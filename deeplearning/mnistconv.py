# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 16:10:28 2017

@author: Aramis
"""

import os
from time import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

x = tf.placeholder(tf.float32, [None, 784], name='input')
y_ = tf.placeholder(tf.float32, [None, 10], name='labels')

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


with tf.name_scope('conv_layer_1'):
    W = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1), name='weights')
    b = tf.Variable(tf.constant(0.1, shape=[32]), name='biases')

    x_image = tf.reshape(x,  [-1, 28, 28, 1])

    h_conv = tf.nn.relu(conv2d(x_image, W) + b)
    h_pool = max_pool_2x2(h_conv)

with tf.name_scope('conv_layer_2'):
    W = tf.Variable(tf.truncated_normal([5,5,32,64], stddev=0.1), name='weights')
    b = tf.Variable(tf.constant(0.1, shape=[64]), name='biases')

    h_conv = tf.nn.relu(conv2d(h_pool, W) + b)
    h_pool = max_pool_2x2(h_conv)

with tf.name_scope('fully_connected_1'):
    W = tf.Variable(tf.truncated_normal([7 * 7 * 64, 1024], stddev=0.1), name='weights')
    b = tf.Variable(tf.constant(0.1, shape=[1024]), name='biases')

    h_pool_fc = tf.reshape(h_pool, [-1, 7 * 7 * 64])
    h_fc = tf.nn.relu(tf.matmul(h_pool_fc, W) + b)

with tf.name_scope('fully_connected_2'):
    keep_prob = tf.placeholder(tf.float32)
    h_drop = tf.nn.dropout(h_fc, keep_prob)

    W = tf.Variable(tf.truncated_normal([1024,10], stddev=0.1), name='weights')
    b = tf.Variable(tf.constant(0.1, shape=[10]), name='biases')

    y_conv = tf.matmul(h_drop, W) + b

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_conv, labels=y_))

train = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

output = tf.argmax(y_conv, 1)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    batch = mnist.train.next_batch(50)
    aa,bb = sess.run((y_conv, y_), feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print(aa)
    print(bb)
    #print('Test Accuracy: {}'.format(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})))
    #print('Model saved to {}'.format(saver.save(sess, os.path.abspath(os.path.join(os.getcwd(), '../model/mnist-model')))))