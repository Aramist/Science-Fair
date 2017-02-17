import sys
import os
import os.path as path

import numpy as np
import tensorflow as tf

from load.nist import LoadData
from deeplearning.trainnisttwo import NistNet

import_data = LoadData()
neural_net = NistNet()

neural_net.build(train=True)

input = neural_net.input
labels = neural_net.y_labels

saver = tf.train.Saver()

def save(session, step=0):
    saver.save(sess, path.abspath(path.join('.', 'trained-model/letterconv')), step)


#PLEASE WORK IM BEGGING YOU



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batch_count = 0
    #while import_data.has_training_data():
    while batch_count < 60 and import_data.has_training_data():
        next_batch, labels_train = import_data.get_training_batch(20)
        #fc3, prob = sess.run((neural_net.fc_3, neural_net.prob), feed_dict={input:next_batch, labels: labels_train})
        _, acc, correct, prediction = sess.run((neural_net.train_step, neural_net.accuracy, neural_net.correct, neural_net.prediction), feed_dict={input:next_batch, labels:labels_train})
        print(acc)
        print(correct)
        print(prediction)
        print('-----------------------------------')
        batch_count += 1
