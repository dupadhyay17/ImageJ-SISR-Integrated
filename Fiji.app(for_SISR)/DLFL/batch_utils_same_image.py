# uncompyle6 version 3.6.7
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.1 (default, Apr  4 2017, 09:40:21) 
# [GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.38)]
# Embedded file name: c:\Users\wangh\Google Drive\Documents\Journal\DL_FL\Plugin\Plugin-jar-pack\DLFL\batch_utils_same_image.py
# Compiled at: 2018-07-11 19:04:26
# Size of source mod 2**32: 7199 bytes
import random
import tensorflow as tf
import threading

import numpy as np
from PIL import Image


class feed_dict(object):

    def __init__(self, images, image_, label_, config):
        self.images = images
        self.image_ = image_
        self.label_ = label_
        self.config = config
        self.capacity = config.q_limit
        image_shape = self.image_.get_shape().as_list()[1:]
        label_shape = self.label_.get_shape().as_list()[1:]
        self.threads = []
        self.queue = tf.RandomShuffleQueue(shapes=[image_shape, label_shape], dtypes=[
         tf.float32, tf.float32],
          capacity=(self.capacity),
          min_after_dequeue=0)
        self.enqueue_op = self.queue.enqueue_many([self.image_, self.label_])

    def get_batch(self):
        return self.queue.dequeue_many(self.config.batch_size)

    def create_thread(self, sess, thread_id, n_threads):
        for image_batch, label_batch in self.batch_generator(self.images[thread_id::n_threads]):
            sess.run((self.enqueue_op), feed_dict={self.image_: image_batch, self.label_: label_batch})

    def start_threads(self, sess, n_threads):
        for i in range(n_threads):
            thread = threading.Thread(target=(self.create_thread), args=(sess, i, n_threads))
            self.threads.append(thread)
            thread.start()


class BatchLoader(object):

    def __init__(self, images, image_, label_, config):
        self.images = images
        self.image_ = image_
        self.label_ = label_
        self.config = config
        self.capacity = config.q_limit
        image_shape = self.image_.get_shape().as_list()[1:]
        label_shape = self.label_.get_shape().as_list()[1:]
        self.threads = []
        self.queue = tf.queue.RandomShuffleQueue(shapes=[image_shape, label_shape], dtypes=[
         tf.float32, tf.float32],
          capacity=(self.capacity),
          min_after_dequeue=0)
        self.enqueue_op = self.queue.enqueue_many([self.image_, self.label_])

    def get_batch(self):
        return self.queue.dequeue_many(self.config.batch_size)

    def create_thread(self, sess, thread_id, n_threads):
        for image_batch, label_batch in self.batch_generator(self.images[thread_id::n_threads]):
            sess.run((self.enqueue_op), feed_dict={self.image_: image_batch, self.label_: label_batch})

    def start_threads(self, sess, n_threads):
        for i in range(n_threads):
            thread = threading.Thread(target=(self.create_thread), args=(sess, i, n_threads))
            self.threads.append(thread)
            thread.start()


class TrainBatchLoader(BatchLoader):

    def __init__(self, images, image_, label_, config):
        super().__init__(images, image_, label_, config)

    def batch_generator(self, paths):
        s = self.config.image_size
        stride = s * 4 // 5
        while True:
            for path in paths:
                with Image.open(path) as (img):
                    image = np.array(img, dtype='f')
                    image = image.reshape((image.shape[0], image.shape[1], 1))
                with Image.open(path.replace('input', 'target')) as (img):
                    label = np.array(img, dtype='f')
                    label = label.reshape((label.shape[0], label.shape[1], 1))
                images, labels = [], []
                for _ in range(30):
                    xx = random.randint(0, image.shape[0] - s)
                    yy = random.randint(0, image.shape[1] - s)
                    if np.mean((image[xx:xx + s, yy:yy + s]), axis=(0, 1)) > 3:
                        img = image[xx:xx + s, yy:yy + s].copy()
                        lab = label[xx:xx + s, yy:yy + s].copy()
                        if random.randint(0, 1):
                            img = np.fliplr(img)
                            lab = np.fliplr(lab)
                        rot = random.randint(0, 3)
                        img = np.rot90(img, k=rot)
                        lab = np.rot90(lab, k=rot)
                        images.append(img)
                        labels.append(lab)

                if len(images) > 0:
                    yield (
                     np.array(images), np.array(labels))


class ValidBatchLoader(BatchLoader):

    def __init__(self, images, image_, label_, config):
        super().__init__(images, image_, label_, config)
        image_shape = self.image_.get_shape().as_list()[1:]
        label_shape = self.label_.get_shape().as_list()[1:]
        self.queue = tf.queue.FIFOQueue(shapes=[image_shape, label_shape], dtypes=[
         tf.float32, tf.float32],
          capacity=(self.capacity))
        self.enqueue_op = self.queue.enqueue_many([self.image_, self.label_])

    def batch_generator(self, paths):
        s = self.config.image_size
        while True:
            for path in paths:
                with Image.open(path) as (img):
                    image = np.array(img, dtype='f')
                    image = image.reshape((image.shape[0], image.shape[1], 1))
                with Image.open(path.replace('input', 'target')) as (img):
                    label = np.array(img, dtype='f')
                    label = label.reshape((label.shape[0], label.shape[1], 1))
                images, labels = [], []
                for _ in range(30):
                    xx = random.randint(0, image.shape[0] - s)
                    yy = random.randint(0, image.shape[1] - s)
                    if np.mean((image[xx:xx + s, yy:yy + s]), axis=(0, 1)) > 3:
                        img = image[xx:xx + s, yy:yy + s].copy()
                        lab = label[xx:xx + s, yy:yy + s].copy()
                        if random.randint(0, 1):
                            img = np.fliplr(img)
                            lab = np.fliplr(lab)
                        rot = random.randint(0, 3)
                        img = np.rot90(img, k=rot)
                        lab = np.rot90(lab, k=rot)
                        images.append(img)
                        labels.append(lab)

                if len(images) > 0:
                    yield (
                     np.array(images), np.array(labels))
# okay decompiling batch_utils_same_image.pyc
