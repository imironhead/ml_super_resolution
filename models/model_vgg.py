"""
https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py

weights:
    https://github.com/fchollet/deep-learning-models/releases/
"""
import collections
import os

import h5py
import tensorflow as tf

from . import layers


class VGG19(tf.keras.Model):
    """
    Implement a non-trainable VGG19 model.
    """
    def __init__(self, weights_path):
        """
        Read pretrained VGG19 weights and build layers.
        """
        super().__init__()

        if weights_path is None or not os.path.isfile(weights_path):
            raise ValueError(f'Invalid path: {weights_path}')

        self._weights_path = weights_path
        self._vgg_layers = []
        self._mean_pixel_color = None

    def build(self, input_shape):
        """
        Build layers for the model.

        Parameters:
            input_shape: Known input shape when building the mode. Note that we
                do not need to call this method. Tensorflow will do the trick.
        """
        vgg_weights = collections.defaultdict(dict)

        with h5py.File(self._weights_path, 'r') as data:
            for scope_name, block in data.items():
                for const_name, weights in block.items():
                    vgg_weights[scope_name][const_name] = weights[()]

        # NOTE:
        layer_names = [
            'block1_conv1', 'block1_conv2', 'block1_pool',
            'block2_conv1', 'block2_conv2', 'block2_pool',
            'block3_conv1', 'block3_conv2', 'block3_conv3', 'block3_conv4',
            'block3_pool',
            'block4_conv1', 'block4_conv2', 'block4_conv3', 'block4_conv4',
            'block4_pool',
            'block5_conv1', 'block5_conv2', 'block5_conv3', 'block5_conv4',
            'block5_pool']

        for layer_name in layer_names:
            if layer_name.endswith('pool'):
                layer = tf.keras.layers.MaxPool2D(padding='same')
            else:
                layer = layers.ConstConv2d(
                    vgg_weights[layer_name][f'{layer_name}_W_1:0'],
                    vgg_weights[layer_name][f'{layer_name}_b_1:0'],
                    f'{layer_name}_W_1:0',
                    f'{layer_name}_b_1:0')

            self._vgg_layers.append((layer_name, layer))

        self._mean_pixel_color = tf.constant([103.939, 116.779, 123.68])

    @tf.function
    def call(self, tensors):
        """
        Do the graph job.
        """
        tensor_table = {}

        tensors = tensors - self._mean_pixel_color

        for layer_name, layer in self._vgg_layers:
            tensor_table[layer_name] = tensors = layer(tensors)

        return tensor_table
