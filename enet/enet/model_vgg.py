"""
https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg19.py

weights:
    https://github.com/fchollet/deep-learning-models/releases/download/v0.1/
"""
import numpy as np
import tensorflow as tf


def conv(tensors, weights, name):
    """
    connect to a conv2d layer and return the result.
    """
    filter_name = '{}_W_1'.format(name)
    biases_name = '{}_b_1'.format(name)

    const_filter = weights[name][filter_name]
    const_biases = weights[name][biases_name]

    tensors = tf.nn.conv2d(tensors, const_filter, [1] * 4, padding='SAME')
    tensors = tf.nn.bias_add(tensors, const_biases)
    tensors = tf.nn.relu(tensors)

    return tensors


def pool(tensors, name):
    """
    """
    return tf.nn.max_pool(
        tensors,
        ksize=[1, 2, 2, 1],
        strides=[1, 2, 2, 1],
        padding='SAME',
        name=name)


def load_vgg_weights(weights_path):
    """
    """
    weights = {}

    # NOTE: weights might be on google cloud storage
    if not tf.gfile.Exists(weights_path):
        return weights

    with tf.gfile.GFile(weights_path, mode='rb') as npz:
        data = np.load(npz, encoding='bytes')

    for name in data.files:
        scope_name = name[:12]
        const_name = name[:-2]

        if scope_name not in weights:
            weights[scope_name] = {}

        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
            weights[scope_name][const_name] = \
                tf.constant(data[name], name=const_name)

    return weights


def build_vgg19_model(source_images, weights):
    """
    source_images:
        input image batch with BGR format and range (0.0, 255.0)

        shape is [N, H, W, C]
    """
    model = {'source_images': source_images}

    # NOTE: RGB to BGR
    tensors = tf.reverse(source_images, [-1])

    # NOTE: substract mean pixel color
    tensors = tensors - tf.constant([103.939, 116.779, 123.68])

    # NOTE: build the network
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
            tensors = pool(tensors, layer_name)
        else:
            tensors = conv(tensors, weights, layer_name)

        model[layer_name] = tensors

    return model

