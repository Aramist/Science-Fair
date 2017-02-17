import tensorflow as tf
import numpy as np

from load.nist import LoadData

class NistNet(object):

    def __init__(self):
        pass

    def build(self, train=False):
        #Nine conv layers, three pools , and FC
        self.input = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 1], name='graph_input')
        #self.input = tf.reshape(self.input, [-1, 128, 128, 1])

        #Conv Layer 1
        self.conv1_1 = self.conv_layer(self.input, 1, 64, 'conv1_1')
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, 'conv1_2')
        self.conv1_3 = self.conv_layer(self.conv1_2, 64, 64, 'conv1_3')
        self.pool_1 = self.max_pool(self.conv1_3, 'pool_1')
        #Output of this layer is 64x64x64

        #Conv layer 2
        self.conv2_1 = self.conv_layer(self.pool_1, 64, 128, 'conv2_1')
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, 'conv2_2')
        self.conv2_3 = self.conv_layer(self.conv2_2, 128, 128, 'conv2_3')
        self.pool_2 = self.max_pool(self.conv2_3, 'pool_2')
        #Output of this layer is 32x32x128

        #Conv layer 3
        self.conv3_1 = self.conv_layer(self.pool_2, 128, 256, 'conv3_1')
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, 'conv3_2')
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, 'conv3_3')
        self.pool_3 = self.max_pool(self.conv3_3, 'pool_3')
        #Output of this layer is 16x16x256

        #Conv layer 4
        self.conv4_1 = self.conv_layer(self.pool_3, 256, 512, 'conv3_1')
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, 'conv3_2')
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, 'conv3_3')
        self.pool_4 = self.max_pool(self.conv4_3, 'pool_3')
        #Output of this layer is 8x8x512

        #Fully Connected Layer 1
        self.fc_1 = self.fc_layer(self.pool_4, 8*8*512, 2048, 'fc_1')
        self.relu_1 = tf.nn.dropout(tf.nn.relu(self.fc_1), 0.6)

        #Fully Connected Layer 2
        self.fc_2 = self.fc_layer(self.relu_1, 2048, 2048, 'fc_2')
        self.relu_2 = tf.nn.dropout(tf.nn.relu(self.fc_2), 0.6)

        #Fully Connected Layer 3
        self.fc_3 = self.fc_layer(self.relu_2, 2048, 26, 'fc_3')

        #For predictions
        self.prob = tf.nn.softmax(self.fc_3, name='prob')

        if train: self.build_train_step()

    def build_train_step(self):
        '''Creates the training step'''
        with tf.variable_scope('train_step'):
            self.y_labels = tf.placeholder(dtype=tf.float32, shape=[None, 26],name='Labels')
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.fc_3, labels=self.y_labels))
            self.train_step = tf.train.GradientDescentOptimizer(0.001).minimize(self.cross_entropy)
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.y_labels, 1), tf.argmax(self.fc_3, 1)), tf.float32))
            self.prediction = tf.argmax(tf.nn.softmax(self.fc_3), 1)
            self.test = tf.reduce_sum(self.prediction)
            self.correct = tf.argmax(self.y_labels, 1)

    def get_variable(self, initial_value, variable_name):
        '''Returns a tf.Variable with initial_value and variable_name'''
        return tf.Variable(initial_value, name=variable_name)

    def conv_layer(self, input, in_channels, out_channels, layer_name):
        '''Returns the conv2d method with 1x1 stride, 3x3 filter, and 1 padding with ReLU applied'''
        with tf.variable_scope(layer_name):
            weights_initial = tf.truncated_normal([3, 3, in_channels, out_channels], 0.0, 0.0001)
            filters = self.get_variable(weights_initial, layer_name + '_filters')
            bias_initial = tf.truncated_normal([out_channels], 0.0, 0.00001)
            biases = self.get_variable(bias_initial, layer_name + '_biases')

            conv = tf.nn.conv2d(input, filters, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias)

            return relu

    def fc_layer(self, input, input_size, output_size, layer_name):
        '''Returns a fully connected layer with given input and output sizes'''
        with tf.variable_scope(layer_name):
            weights_init = tf.truncated_normal([input_size, output_size], stddev=0.0001)
            weights = self.get_variable(weights_init, layer_name + '_weights')

            biases_init = tf.truncated_normal([output_size], 0.0, 0.00001)
            biases = self.get_variable(biases_init, layer_name + '_biases')

            hidden_layer = tf.reshape(input, [-1, input_size])
            fully_connected = tf.nn.bias_add(tf.matmul(hidden_layer, weights), biases)

            return fully_connected

    def max_pool(self, input, layer_name):
        '''Returns the maxpool operation on input with a 2x2 stride and 2x2 filter'''
        with tf.variable_scope(layer_name):
            return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=layer_name)
    
    def save(self, session, path):
        pass
