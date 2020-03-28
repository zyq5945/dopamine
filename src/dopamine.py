# Copyright 2020 The zyq5945 Authors. All Rights Reserved.
# Licensed under the BSD License.

import tensorflow as tf
from tensorflow.keras import layers

assert tf.__version__.startswith('2.')

"""
Dopamine is a trainable connections layer without bias
"""


class Dopamine(layers.Layer):
    """
    input_shape:input_shape
    tf2 Sequential fit bug, tf2 must have batch_size
    """

    def __init__(self, input_shape, batch_size=None, use_bias=True, dtype=tf.float32):
        # assert (not tf.__version__.startswith('2.0')) or (batch_size != None)
        super(Dopamine, self).__init__()
        self.kernel = self.add_weight('Dopamine_w', input_shape, dtype=dtype)
        self.kernel.assign(tf.ones(input_shape, dtype=dtype))
        self.inputs_shape = [batch_size] + input_shape
        if use_bias:
            self.bias = self.add_weight('Dopamine_b', input_shape, dtype=dtype)
            self.bias.assign(tf.zeros(input_shape, dtype=dtype))
        else:
            self.bias = None

    def call(self, inputs, training=None):
        # tf2 Sequential fit bug, inputs.shape[0] is sometimes None
        return dopamine(inputs, self.kernel, self.bias, self.inputs_shape)


def dopamine(inputs, kernel, bias=None, inputs_shape=None):
    shape = inputs.shape
    if is_invalid_dim_number(shape[0]):
        shape = inputs_shape
    if is_invalid_dim_number(shape[0]):
        raise Exception("inputs shape value error!!!")
    temp = tf.broadcast_to(kernel, shape)
    out = tf.multiply(inputs, temp)
    if bias is not None:
        out = out + bias
    return out


def is_invalid_dim_number(n):
    return (not isinstance(n, int)) or (n <= 0)


def set_layers_dbs(layers, batch_size):
    for l in layers:
        if isinstance(l, Dopamine):
            l.inputs_shape[0] = batch_size

