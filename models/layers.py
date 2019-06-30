"""
Implement customized layers.
"""
import tensorflow as tf


class ConstConv2d(tf.keras.layers.Layer):
    """
    Implement a conv2d layer which is not trainable (no weight & bias
    variables).
    """
    def __init__(self, weights, bias, weights_name, bias_name, name=None):
        """
        Initialization.
        """
        super().__init__()

        self._filters = tf.constant(weights, name=weights_name)
        self._bias = tf.constant(bias, name=bias_name)
        self._name = name or 'ConstConv2d'

    def call(self, tensors):
        """
        Do the layer job.
        """
        with tf.name_scope(self._name) as scope:
            tensors = tf.nn.conv2d(
                tensors, self._filters, [1] * 4, padding='SAME')
            tensors = tf.nn.bias_add(tensors, self._bias)
            tensors = tf.nn.relu(tensors, name=scope)

        return tensors
