"""
"""
import tensorflow as tf


def build_model(sd_images, hd_images=None, num_layers=20, use_adam=False):
    """
    sd_images: lo resolution images to be super resolved
    hd_images: hi resolution images as ground truth of super resolved results
    num_layers: num of conv_relu layers
    """
    model = {}

    # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
    #       deep convolutional networks
    #
    #       5.2
    #       for weight initialization, we use the method described in He et al.
    #       [10]. this is a theoretically sound procedure for networks
    #       utilizing rectified linear units (ReLu).
    #
    #       arXiv:1502.01852v1, delving deep into rectifiers: surpassing
    #       human-level performance on imagenet classification
    #
    #       2.2
    #       xavier
    initializer = tf.contrib.layers.xavier_initializer()

    # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
    #       deep convolutional networks, 3.2
    #
    #       the training is regularized by weight decay (l2 penalty multiplied
    #       by 0.0001).
    regularizer = tf.contrib.layers.l2_regularizer(0.0001)

    tensors = sd_images

    # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
    #       deep convolutional networks, 2.1
    #
    #       our output image has the same size as the input image by padding
    #       zeros every layer during training whereas output from SRCNN is
    #       smaller than the input.
    #
    #       use 'same' padding, guess default padded value is zero

    for i in range(num_layers - 1):
        # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
        #       deep convolutional networks, 3.1
        #
        #       we use d layers where layers except the first and the last are
        #       of the same type: 64 filters of the size 3x3x64 where a filter
        #       operates on 3x3 spatial region across 64 channels (feature
        #       maps).
        tensors = tf.layers.conv2d(
            tensors,
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=initializer,
            kernel_regularizer=regularizer)

    # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
    #       deep convolutional networks, 3.1
    #
    #       the last layer, used for image reconstruction, consists of a single
    #       filter of size 3x3x64
    #
    #       residual
    tensors = tf.layers.conv2d(
        tensors,
        filters=3,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=None,
        kernel_initializer=initializer,
        kernel_regularizer=regularizer)

    # NOTE: super resolved
    sr_images = sd_images + tensors

    sr_images = tf.identity(sr_images, 'sr_images')

    model['sd_images'] = sd_images
    model['sr_images'] = sr_images

    # NOTE: building predicting model if hd_images is None
    if hd_images is None:
        return model

    # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
    #       deep convolutional networks, 3.2
    #
    #       the loss function now becomes 0.5 * || r - f(x) ||^2, where f(x) is
    #       the network prediction
    loss = tf.losses.mean_squared_error(
        hd_images,
        sr_images,
        reduction=tf.losses.Reduction.MEAN)

    loss = loss + sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

    # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
    #       deep convolutional networks
    #
    #       table 1
    #       different learning rates are tested
    #
    #       5.2
    #       learning rate was initially set to 0.1 and then decreased by a
    #       factor of 10 every 20 epochs.
    learning_rate = tf.get_variable(
        'learning_rate',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0.1, dtype=tf.float32),
        dtype=tf.float32)

    step = tf.train.get_or_create_global_step()

    if use_adam:
        trainer = tf.train \
            .AdamOptimizer(learning_rate=learning_rate) \
            .minimize(loss, global_step=step)
    else:
        # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
        #       deep convolutional networks
        #
        #       5.2
        #       momentum and weight decay parameters are set to 0.9 and 0.0001,
        #       respectively.
        #
        # NOTE: not sure if it's the same as tf.train.MomentumOptimizer
        trainer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=0.9)

        # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
        #       deep convolutional networks
        #
        #       3.2
        #       for maximal speed of convergence, we clip the gradients to
        #       [-theta / learning_rate, theta / learning_rate], where gamma
        #       denotes the current learning rate.
        #
        # NOTE: can not find the value of theta on the paper
        gradient_cap = tf.constant(0.01, dtype=tf.float32, name='gradient_cap')

        cap = gradient_cap / learning_rate

        grad_var_pairs = trainer.compute_gradients(loss, tf.trainable_variables())

        capped_grad_var_pairs = []

        for gradient, variable in grad_var_pairs:
            gradient = tf.clip_by_value(gradient, -cap, cap)

            capped_grad_var_pairs.append((gradient, variable))

        trainer = trainer.apply_gradients(
            capped_grad_var_pairs, global_step=step)

    model['step'] = step
    model['loss'] = loss
    model['trainer'] = trainer
    model['hd_images'] = hd_images
    model['learning_rate'] = learning_rate

    return model

