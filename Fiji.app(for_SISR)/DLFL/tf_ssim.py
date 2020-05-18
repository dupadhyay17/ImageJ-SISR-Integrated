# uncompyle6 version 3.6.7
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.1 (default, Apr  4 2017, 09:40:21) 
# [GCC 4.2.1 Compatible Apple LLVM 8.1.0 (clang-802.0.38)]
# Embedded file name: c:\Users\UCLA\Desktop\1.52_copy\python files\tf_ssim.py
# Compiled at: 2018-03-06 16:26:40
# Size of source mod 2**32: 2440 bytes
import tensorflow as tf
from tensorflow.python.util import nest

def _with_flat_batch(flat_batch_fn):

    def fn(x, *args, **kwargs):
        shape = tf.shape(x)
        flat_batch_x = tf.reshape(x, tf.concat([[-1], shape[-3:]], axis=0))
        flat_batch_r = flat_batch_fn(flat_batch_x, *args, **kwargs)
        r = nest.map_structure(lambda x: tf.reshape(x, tf.concat([shape[:-3], x.shape[1:]], axis=0)), flat_batch_r)
        return r

    return fn


def structural_similarity(X, Y, K1=0.001, K2=0.003, win_size=8, data_range=255.0, use_sample_covariance=True):
    """
    Structural SIMilarity (SSIM) index between two images
    Args:
        X: A tensor of shape `[..., in_height, in_width, in_channels]`.
        Y: A tensor of shape `[..., in_height, in_width, in_channels]`.
    Returns:
        The SSIM between images X and Y.
    Reference:
        https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/_structural_similarity.py
    Broadcasting is supported.
    """
    X = tf.convert_to_tensor(X)
    Y = tf.convert_to_tensor(Y)
    ndim = 2
    nch = tf.shape(X)[(-1)]
    filter_func = _with_flat_batch(tf.nn.depthwise_conv2d)
    kernel = tf.cast(tf.fill([win_size, win_size, nch, 1], 1 / win_size ** 2), X.dtype)
    filter_args = {'filter':kernel,  'strides':[1] * 4,  'padding':'VALID'}
    NP = win_size ** ndim
    if use_sample_covariance:
        cov_norm = NP / (NP - 1)
    else:
        cov_norm = 1.0
    ux = filter_func(X, **filter_args)
    uy = filter_func(Y, **filter_args)
    uxx = filter_func((X * X), **filter_args)
    uyy = filter_func((Y * Y), **filter_args)
    uxy = filter_func((X * Y), **filter_args)
    vx = cov_norm * (uxx - ux * ux)
    vy = cov_norm * (uyy - uy * uy)
    vxy = cov_norm * (uxy - ux * uy)
    R = data_range
    C1 = (K1 * R) ** 2
    C2 = (K2 * R) ** 2
    A1, A2, B1, B2 = (
     2 * ux * uy + C1,
     2 * vxy + C2,
     ux ** 2 + uy ** 2 + C1,
     vx + vy + C2)
    D = B1 * B2
    S = A1 * A2 / D
    ssim = tf.reduce_mean(S, axis=[-3, -2, -1])
    return ssim
# okay decompiling tf_ssim.pyc
