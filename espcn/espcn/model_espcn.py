"""
"""
import tensorflow as tf


def build_model(lr_source, scaling_factor=3, hr_target=None):
    """
    build espcn model

    lr_source:  source image batch tensor to be super resolved
    hr_target:  target image batch as training labels in sub-pixel convolved
                shape, build a partial model for testing if hr_target is None
    scaling_factor:
                factor of up scaling. the delpth of the final layer should be
                3 * scaling_factor * scaling_factor (sub-pixel layer)
    """
    model = {
        'lr_source': lr_source,
    }

    initializer = tf.truncated_normal_initializer(stddev=0.02)

    # NOTE: arXiv:1609.05158v2, 3.2
    #       for the espcn, we set l = 3, (f1, n1) = (5, 64), (f2, n2) = (3, 32)
    #       and f3 = 3 in our evaluations.
    #
    # NOTE: arXiv:1609.05158v2, 3.2
    #       we choose tanh instead ofrelu as the activation function for the
    #       final model motivated by our experimental results.
    tensors = tf.layers.conv2d(
        lr_source,
        filters=64,
        kernel_size=5,
        strides=1,
        padding='same',
        activation=tf.nn.tanh,
        kernel_initializer=initializer,
        name='f1')

    tensors = tf.layers.conv2d(
        tensors,
        filters=32,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.tanh,
        kernel_initializer=initializer,
        name='f2')

    # NOTE: fit depth to sub-pixel convolution layer
    sr_result = tf.layers.conv2d(
        tensors,
        filters=3 * (scaling_factor ** 2),
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.tanh,
        kernel_initializer=initializer,
        name='f3')

    model['sr_result'] = sr_result

    if hr_target is None:
        # NOTE: we do not want to train this model (freeze it to gain better
        #       performance)
        return model

    model['hr_target'] = hr_target

    # NOTE: arXiv:1609.05158v2, 2.2
    #       and calculate the pixel-wise mean squared error (MSE) of the
    #       reconstruction as an objective function to train the network.
    loss = tf.losses.mean_squared_error(
        sr_result, hr_target, reduction=tf.losses.Reduction.MEAN)

    # NOTE: arXiv:1609.05158v2, 3.2
    #       initial learning rate is  set to 0.01 and final learning rate is
    #       set to 0.0001 and updated gradually when the improvement of the
    #       cost function is smaller than a threshold mu.
    learning_rate = tf.get_variable(
        'learning_rate',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0.01, dtype=tf.float32),
        dtype=tf.float32)

    step = tf.train.get_or_create_global_step()

    optimizer = tf.train \
        .AdamOptimizer(learning_rate=learning_rate) \
        .minimize(loss, global_step=step)

    model['step'] = step
    model['loss'] = loss
    model['optimizer'] = optimizer
    model['learning_rate'] = learning_rate

    return model

