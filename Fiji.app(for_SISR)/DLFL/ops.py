# uncompyle6 version 3.6.7
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.1 (default, Apr  4 2017, 09:40:21) 
# [GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.38)]
# Embedded file name: c:\Users\UCLA\Desktop\1.52_copy\python files\ops.py
# Compiled at: 2018-04-29 16:14:48
# Size of source mod 2**32: 982 bytes
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.client import device_lib
import tensorflow as tf
from tqdm import tqdm
import numpy as np, sys, unet as network

def average_gradients(tower_grads):
    average_grads = []
    for gv in zip(*tower_grads):
        grads = [tf.expand_dims(g, 0) for g, _ in gv]
        grad = tf.reduce_mean(tf.concat(grads, 0), 0)
        average_grads.append((grad, gv[0][1]))

    return average_grads


def add_batch_norm_dependencies(loss):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    if update_ops:
        updates = tf.tuple(update_ops)
        loss = control_flow_ops.with_dependencies(updates, loss)
    return loss


def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def print_out(file, text):
    file.write(text + '\n')
    file.flush()
    print(text)
    sys.stdout.flush()
# okay decompiling ops.pyc
