import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS

class ImageEmbedding(object):
    def __init__(self, hidden_num, channels, conv_initializer, k=5):
        self.hidden_num = hidden_num
        self.channels = channels
        with tf.variable_scope('image_embedding', reuse=tf.AUTO_REUSE):
            self.conv1_kernel = tf.get_variable('conv1_kernel', [k, k, self.channels, self.hidden_num],
                                                initializer=conv_initializer)

            self.conv2_kernel = tf.get_variable('conv2_kernel', [k, k, self.hidden_num, self.hidden_num],
                                                initializer=conv_initializer)
        self.activation = tf.nn.relu

    def model(self, images):
        conv = tf.nn.conv2d(images, self.conv1_kernel, [1, 1, 1, 1], padding='SAME') # images: (25, 84, 84, 3), conv: (25, 84, 84, 32)
        conv1 = tf.nn.relu(conv, name='conv1_post_activation') # model/conv1_post_activation:0 (25, 84, 84, 32)

        pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1],
                               padding='SAME', name='pool1') # model/pool1:0(25, 42, 42, 32)
        norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm1') # model/norm1:0 (25, 42, 42, 32)

        conv2 = tf.nn.conv2d(norm1, self.conv2_kernel, [1, 1, 1, 1], padding='SAME') # model/CONV2D_1:0 (25, 42, 42, 32)
        conv2_act = tf.nn.relu(conv2, name='conv2_post_activation') # model/conv2_post_activation:0 (25, 42, 42, 32)
        norm2 = tf.nn.lrn(conv2_act, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,
                          name='norm2') # model/norm2:0 (25, 42, 42, 32)
        pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1],
                               strides=[1, 2, 2, 1], padding='SAME', name='pool2') # model/pool2:0 (25,21,21,32)

        with tf.variable_scope('local3', reuse=tf.AUTO_REUSE):
            image_reshape = tf.reshape(pool2, [images.get_shape().as_list()[0], -1]) # model/local3/REshape:0 (25, 14112)
            dim = image_reshape.get_shape()[1].value # 14112
            local3_weight = tf.get_variable(name='weight', shape=[dim, 384],
                                            initializer=tf.truncated_normal_initializer(stddev=0.04)) # model/local3/weight:0 (14112, 384)
            local3_biases = tf.get_variable(name='biases', shape=[384], initializer=tf.constant_initializer(0.1))  # model/local3/biases:0 (384,)
            local3=tf.nn.relu(tf.matmul(image_reshape, local3_weight)+local3_biases, name='local3_dense') # model/local3/local3_dense:0 (25, 384)

        with tf.variable_scope('local4', reuse=tf.AUTO_REUSE):
            local4_weight = tf.get_variable(name='weight', shape=[384, FLAGS.hidden_dim],
                                            initializer=tf.truncated_normal_initializer(stddev=0.04)) # model/local4/weight:0 (384, 40)
            local4_biases = tf.get_variable(name='biases', shape=[FLAGS.hidden_dim], initializer=tf.constant_initializer(0.1)) # # model/local4/biases:0 (40,)
            local4 = tf.nn.relu(tf.matmul(local3, local4_weight) + local4_biases, name='local4_dense') # model/local4/local4_dense:0 (25, 40)
        return local4
