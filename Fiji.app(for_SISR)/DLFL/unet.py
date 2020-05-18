# uncompyle6 version 3.6.7
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.1 (default, Apr  4 2017, 09:40:21) 
# [GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.38)]
# Embedded file name: c:\Users\UCLA\Desktop\1.52_copy\python files\unet.py
# Compiled at: 2018-04-29 16:14:08
# Size of source mod 2**32: 3713 bytes
import numpy as np
import tensorflow as tf


def conv2d(inp, shp, name, strides=(1, 1, 1, 1), padding='SAME'):
    with tf.device('/cpu:0'):
        filters = tf.compat.v1.get_variable((name + '/filters'), shp, initializer=tf.compat.v1.truncated_normal_initializer(stddev=(np.sqrt(2.0 / (shp[0] * shp[1] * shp[3])))))
        biases = tf.compat.v1.get_variable((name + '/biases'), [shp[(-1)]], initializer=(tf.constant_initializer(0)))
    return tf.nn.bias_add(tf.nn.conv2d(inp, filters, strides=strides, padding=padding), biases)


def leakyRelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def fc_layer(inp, shp, name):
    with tf.name_scope('fc_layer'):
        with tf.device('/cpu:0'):
            weights = tf.compat.v1.get_variable((name + '/weights'), shp, initializer=(tf.initializers.GlorotUniform))
            biases = tf.compat.v1.get_variable((name + '/biases'), [shp[(-1)]], initializer=(tf.constant_initializer(0)))
    return tf.nn.bias_add(tf.matmul(inp, weights), biases)


def normal_block(inp, name, is_training):
    with tf.name_scope('normal_block'):
        ch = inp.get_shape().as_list()[(-1)]
        conv1 = leakyRelu(conv2d(inp, [3, 3, ch, ch], name + '/conv1'))
        conv2 = leakyRelu(conv2d(conv1, [3, 3, ch, ch * 2], (name + '/conv2'), strides=(1,
                                                                                        2,
                                                                                        2,
                                                                                        1)))
    return conv2


class Generator(object):

    def __init__(self, inp, config):
        self.dic = {}
        self.config = config
        cur = inp
        print(cur.get_shape())
        for i in range(self.config.n_levels):
            cur = self.down(cur, i)

        ch = cur.get_shape().as_list()[(-1)]
        cur = leakyRelu(conv2d(cur, [3, 3, ch, ch], 'center'))
        for i in range(self.config.n_levels):
            cur = self.up(cur, self.config.n_levels - i - 1)

        self.output = conv2d(cur, [3, 3, self.config.n_channels // 2, 1], 'last_layer')

    def down(self, inp, lvl):
        name = 'down{}'.format(lvl)
        in_ch = inp.get_shape().as_list()[(-1)]
        out_ch = self.config.n_channels if lvl == 0 else in_ch * 2
        mid_ch = (in_ch + out_ch) // 2
        conv1 = leakyRelu(conv2d(inp, [3, 3, in_ch, mid_ch], name + '/conv1'))
        conv2 = leakyRelu(conv2d(conv1, [3, 3, mid_ch, mid_ch], name + '/conv2'))
        conv3 = leakyRelu(conv2d(conv2, [3, 3, mid_ch, out_ch], name + '/conv3'))
        tmp = tf.pad(inp, [[0, 0], [0, 0], [0, 0], [0, out_ch - in_ch]], 'CONSTANT')
        self.dic[name] = conv3 + tmp
        return tf.nn.avg_pool((self.dic[name]), ksize=(1, 2, 2, 1), strides=(1, 2,
                                                                             2, 1), padding='SAME')

    def up(self, inp, lvl):
        name = 'up{}'.format(lvl)
        size = self.config.image_size >> lvl
        image = tf.compat.v1.image.resize_bilinear(inp, [size, size])
        image = tf.concat([image, self.dic[name.replace('up', 'down')]], axis=3)
        in_ch = image.get_shape().as_list()[(-1)]
        out_ch = in_ch // 4
        mid_ch = (in_ch + out_ch) // 2
        conv1 = leakyRelu(conv2d(image, [3, 3, in_ch, mid_ch], name + '/conv1'))
        conv2 = leakyRelu(conv2d(conv1, [3, 3, mid_ch, mid_ch], name + '/conv2'))
        conv3 = leakyRelu(conv2d(conv2, [3, 3, mid_ch, out_ch], name + '/conv3'))
        return conv3


class Discriminator(object):

    def __init__(self, inp, config):
        cur = leakyRelu(conv2d(inp, [3, 3, 1, config.n_channels], 'conv1'))
        for i in range(config.n_blocks):
            cur = normal_block(cur, 'n_block{}'.format(i), config.is_training)

        cur = tf.reduce_mean(cur, axis=(1, 2))
        ch = cur.get_shape().as_list()[(-1)]
        cur = leakyRelu(fc_layer(cur, [ch, ch], 'fcl1'))
        self.output = tf.nn.sigmoid(fc_layer(cur, [ch, 1], 'fcl2'))
# okay decompiling unet.pyc
