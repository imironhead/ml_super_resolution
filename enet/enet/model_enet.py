"""
"""
import tensorflow as tf

import enet.model_vgg as model_vgg


def residual_block(tensors, num_filters):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    x_tensors = tf.layers.conv2d(
        tensors,
        filters=num_filters,
        kernel_size=3,
        strides=1,
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=initializer)

    x_tensors = tf.layers.conv2d(
        x_tensors,
        filters=num_filters,
        kernel_size=1,
        strides=1,
        padding='same',
        activation=None,
        kernel_initializer=initializer)

    return tf.nn.relu(tensors + x_tensors)


def build_generator(sd_images, scope_name='generator'):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    # NOTE: get image size to compute the size of super-resoloved images
    n, h, w, c = sd_images.get_shape().as_list()

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        # NOTE: a bicubic version for the output cause the model try to learn
        #       residuals, we only do 4x super resolution experiment
        bq_images = tf.image.resize_bicubic(
            sd_images, [4 * h, 4 * w], align_corners=True)

        # NOTE: arXiv: 1612.07919v2, table 1
        # NOTE: to 64 channels
        tensors = tf.layers.conv2d(
            sd_images,
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=initializer)

        # NOTE: arXiv: 1612.07919v2, table 1, we use 3x3 comvolution kernels,
        #       10 residual blocks and RGB images
        for _ in range(10):
            tensors = residual_block(tensors, 64)

        # NOTE: arXiv: 1612.07919v2, table 1, nearest neighbor upsampling
        for factor in [2, 4]:
            tensors = tf.image.resize_nearest_neighbor(
                tensors, [factor * h, factor * w])

            tensors = tf.layers.conv2d(
                tensors,
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.relu,
                kernel_initializer=initializer)

        # NOTE: one more?
        tensors = tf.layers.conv2d(
            tensors,
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=tf.nn.relu,
            kernel_initializer=initializer)

        # NOTE: to residual image, how should we clamp the final image?
        tensors = tf.layers.conv2d(
            tensors,
            filters=3,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None,
            kernel_initializer=initializer)

        # NOTE: bicubic upscaled image plus learned residual images as
        #       super-resolved images
        sr_images = bq_images + tensors

    return sr_images


def build_discriminator(hd_images, scope_name='discriminator'):
    """
    # NOTE: arXiv: 1612.07919, supplementary, table 1
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    tensors = hd_images

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
        for i in range(5):
            filters = 2 ** (i + 5)

            tensors = tf.layers.conv2d(
                tensors,
                filters=filters,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.leaky_relu,
                kernel_initializer=initializer)

            tensors = tf.layers.conv2d(
                tensors,
                filters=filters,
                kernel_size=3,
                strides=2,
                padding='same',
                activation=tf.nn.leaky_relu,
                kernel_initializer=initializer)

        tensors = tf.layers.flatten(tensors)

        tensors = tf.layers.dense(
            tensors,
            units=1024,
            activation=tf.nn.leaky_relu,
            kernel_initializer=initializer)

        tensors = tf.layers.dense(
            tensors,
            units=1,
            activation=tf.nn.sigmoid,
            kernel_initializer=initializer)

    return tensors


def generator_loss(fake):
    """
    """
    # NOTE: arXiv:1612.07915, weights come from table 2
    return 2.0 * tf.losses.log_loss(tf.ones_like(fake), fake)


def discriminator_loss(fake, real):
    """
    """
    loss_fake = tf.losses.log_loss(tf.zeros_like(fake), fake)
    loss_real = tf.losses.log_loss(tf.ones_like(real), real)

    return loss_fake + loss_real


def perceptual_loss(sr_images_vgg, hd_images_vgg):
    """
    arXiv:1612.07915v2, 4.2.2
    to capture both low-level and high-level features, we use a combination of
    the second and fifth pooling layers and compute the MSE on their feature
    activations.
    """
    loss_pool_2 = tf.losses.mean_squared_error(
        sr_images_vgg['block2_pool'], hd_images_vgg['block2_pool'])
    loss_pool_5 = tf.losses.mean_squared_error(
        sr_images_vgg['block5_pool'], hd_images_vgg['block5_pool'])

    # NOTE: arXiv:1612.07915, weights come from table 2
    return 0.2 * loss_pool_2 + 0.02 * loss_pool_5


def texture_matching_loss(sr_images_vgg, hd_images_vgg):
    """
    arXiv:1612.07919v2, 4.2.3
    """
    # NOTE: supplementary, table 2
    layers = [
        ('block1_conv1', 3e-7),
        ('block2_conv1', 1e-6),
        ('block3_conv1', 1e-6),
    ]

    loss = 0

    for name, weight in layers:
        sr_tensors = sr_images_vgg[name]
        hd_tensors = hd_images_vgg[name]

        shape = tf.shape(sr_tensors)

        h, w, c = shape[1], shape[2], shape[3]

        # NOTE: 5.3 for the perceptual loss Lp and the texture loss Lt, we
        #       normalized feature activations to have mean of one.
        sr_norm = tf.norm(sr_tensors, axis=[1, 2], keepdims=True)
        hd_norm = tf.norm(hd_tensors, axis=[1, 2], keepdims=True)

        sr_tensors = tf.div(sr_tensors, sr_norm)
        hd_tensors = tf.div(hd_tensors, hd_norm)

        # NOTE: 4.2.3 empirically, we found a patch size of 16x16 pixels to
        #       result in the best balance between faithful texture generation
        #       and the overall perceptual quality of the images.
        # NOTE: [N, H/16, W/16, 16*16*C]
        sr_tensors = tf.extract_image_patches(
            sr_tensors,
            ksizes=[1, 16, 16, 1],
            strides=[1, 16, 16, 1],
            rates=[1, 1, 1, 1],
            padding='VALID')

        hd_tensors = tf.extract_image_patches(
            hd_tensors,
            ksizes=[1, 16, 16, 1],
            strides=[1, 16, 16, 1],
            rates=[1, 1, 1, 1],
            padding='VALID')

        sr_tensors = tf.reshape(sr_tensors, [-1, 1, h * w, c])
        hd_tensors = tf.reshape(hd_tensors, [-1, 1, h * w, c])

        # NOTE: gram matriices of shape [N, C, H*W, H*W]
        sr_grams = tf.matmul(sr_tensors, sr_tensors, transpose_a=True)
        hd_grams = tf.matmul(hd_tensors, hd_tensors, transpose_a=True)

        loss += weight * tf.losses.mean_squared_error(sr_grams, hd_grams)

    return loss


def build_enet(sd_images, hd_images, vgg19_path):
    """
    NOTE: ENet-PAT
    """
    model = {}

    # NOTE: generator which do the super resolution
    sr_images = build_generator(sd_images, 'g_')

    model['sd_images'] = sd_images
    model['sr_images'] = sr_images

    # NOTE: model for generating super resolved images
    if hd_images is None:
        return model

    model['hd_images'] = hd_images

    # NOTE: load vgg weights
    vgg19_weights = model_vgg.load_vgg_weights(vgg19_path)

    # NOTE: vgg features for perceptual loss and texture matching loss
    hd_images_vgg = model_vgg.build_vgg19_model(hd_images, vgg19_weights)
    sr_images_vgg = model_vgg.build_vgg19_model(sr_images, vgg19_weights)

    # NOTE: descriminate real hd images
    real = build_discriminator(hd_images, 'd_')

    # NOTE: discriminate fake hd images
    fake = build_discriminator(sr_images, 'd_')

    # NOTE: generator loss
    g_loss = generator_loss(fake)

    # NOTE: adversarial loss
    a_loss = discriminator_loss(fake, real)

    # NOTE: perceptual loss
    p_loss = perceptual_loss(sr_images_vgg, hd_images_vgg)

    # NOTE: texture matching loss
    t_loss = texture_matching_loss(sr_images_vgg, hd_images_vgg)

    # NOTE: collect variables to separate g/d training op
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('d_')]
    g_vars = [v for v in tf.trainable_variables() if v.name.startswith('g_')]

    step = tf.train.get_or_create_global_step()

    g_losses = g_loss + p_loss + t_loss

    g_trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
    g_trainer = g_trainer.minimize(g_losses, var_list=g_vars, global_step=step)

    d_trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
    d_trainer = d_trainer.minimize(a_loss, var_list=d_vars)

    model['step'] = step
    model['g_loss'] = g_loss
    model['p_loss'] = p_loss
    model['t_loss'] = t_loss
    model['a_loss'] = a_loss
    model['g_loss_all'] = g_losses
    model['g_trainer'] = g_trainer
    model['d_trainer'] = d_trainer

    return model

