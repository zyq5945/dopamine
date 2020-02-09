# Copyright 2020 The zyq5945 Authors. All Rights Reserved.
# Licensed under the BSD License.

import  tensorflow as tf
from    tensorflow.keras import layers


assert tf.__version__.startswith('2.')

"""
Dopamine is a trainable connections layer without bias
"""
class Dopamine(layers.Layer):
	"""
	input_shape:input_shape
	tf2.0 Sequential fit bug, tf2.0 must have batch_size	
	"""
	def __init__(self, input_shape, batch_size = None, dtype=tf.float32):
		#assert (not tf.__version__.startswith('2.0')) or (batch_size != None)
		super(Dopamine, self).__init__()
		self.kernel = self.add_weight('Dopamine_w', input_shape, dtype=dtype)	
		self.kernel.assign(tf.ones(input_shape, dtype=dtype))
		self.inputs_shape = [batch_size] + input_shape

	def call(self, inputs, training=None):
		#tf2.0 Sequential fit bug, inputs.shape[0] is sometimes None
		return dopamine(inputs, self.kernel, None, self.inputs_shape)


"""
DopamineEx is a trainable connections layer with bias
"""
class DopamineEx(Dopamine):
	"""
	input_shape:input_shape
	tf2.0 Sequential fit bug, tf2.0 must have batch_size	
	"""
	def __init__(self, input_shape, batch_size = None, dtype=tf.float32):
		#assert (not tf.__version__.startswith('2.0')) or (batch_size != None)
		super(DopamineEx, self).__init__(input_shape, batch_size, dtype)
		self.bias = self.add_weight('DopamineEx_b', input_shape, dtype=dtype)		
		self.bias.assign(tf.zeros(input_shape, dtype=dtype))

	def call(self, inputs, training=None):
		#tf2.0 Sequential fit bug, inputs.shape[0] is sometimes None
		return dopamine(inputs, self.kernel, self.bias, self.inputs_shape)

def dopamine(inputs, kernel, bias = None, inputs_shape = None):
	shape = inputs.shape
	if (shape[0] is None) or (not shape[0]):
		shape = inputs_shape
	temp = tf.broadcast_to(kernel, shape)
	out = tf.multiply(inputs, temp)
	if bias is not None:
	 	out = out + bias
	return out;
