# -*- coding: utf-8 -*-
"""
Created on Sun Jan  1 16:10:58 2017

@author: Aramis
"""

import os
import time

import numpy as np
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def abs_path(relative):
    return os.path.abspath(os.path.join(os.getcwd(), relative))

def load_model():
    return tf.train.import_meta_graph(abs_path('model/mnist-model.meta'))

def test_image(img):
    img_resized = [np.resize(img, (28 * 28))]
    with tf.Session() as sess:
        load_model().restore(sess, tf.train.get_checkpoint_state(abs_path('model/')).model_checkpoint_path)
        graph = tf.get_default_graph()
        graph_input = graph.get_tensor_by_name('input:0')
        graph_output = graph.get_tensor_by_name('add_4:0')
        graph_keep_prob = graph.get_tensor_by_name('Placeholder_1:0')
        graph_y_conv = graph.get_tensor_by_name('add_3:0')
        return sess.run(graph_output, feed_dict={graph_input: img_resized, graph_keep_prob: 1.0})[0]

def self_test():
    with tf.Session() as sess:
        mnist = input_data.read_data_sets('data/MNIST', one_hot=True)
        load_model().restore(sess, tf.train.get_checkpoint_state(abs_path('model/')).model_checkpoint_path)
        graph = tf.get_default_graph()
        graph_input = graph.get_tensor_by_name('input:0')
        graph_labels = graph.get_tensor_by_name('Placeholder:0')
        graph_keep_prob = graph.get_tensor_by_name('Placeholder_1:0')
        graph_accuracy = graph.get_tensor_by_name('Mean_1:0')
        print(sess.run(graph_accuracy, feed_dict={graph_input:mnist.test.images, graph_labels:mnist.test.labels, graph_keep_prob: 1.0}))