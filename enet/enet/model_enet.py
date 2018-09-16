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


def normalize(tensors):
    """
    NOTE: 5.3 for the perceptual loss Lp and the texture loss Lt, we normalized
          feature activations to have mean of one.
    """
    mean = tf.reduce_mean(tensors, axis=-1, keepdims=True)

    return tensors / (mean + 0.000001)


def build_generator(sd_images, bq_images, hd_images, scope_name='generator'):
    """
    """
    initializer = tf.truncated_normal_initializer(stddev=0.02)

    # NOTE: during training, we need all dimensions for the dense layer in
    #       discriminator, and we use 32x32x3 sd images.
    # NOTE: during generation, we use unknown size images, so the width and
    #       height here are unknown (determined during super resolving)
    if hd_images is None:
        shape = tf.shape(sd_images)

        n, h, w, c = shape[0], shape[1], shape[2], shape[3]
    else:
        n, h, w, c = -1, 32, 32, 3

    with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
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
    return tf.losses.log_loss(
        tf.ones_like(fake), fake, reduction=tf.losses.Reduction.MEAN)


def discriminator_loss(fake, real):
    """
    """
    loss_fake = tf.losses.log_loss(
        tf.zeros_like(fake), fake, reduction=tf.losses.Reduction.MEAN)
    loss_real = tf.losses.log_loss(
        tf.ones_like(real), real, reduction=tf.losses.Reduction.MEAN)

    return loss_fake + loss_real


def perceptual_loss(sr_images_vgg, hd_images_vgg):
    """
    arXiv:1612.07915v2, 4.2.2
    to capture both low-level and high-level features, we use a combination of
    the second and fifth pooling layers and compute the MSE on their feature
    activations.
    """
    # NOTE: 5.3 for the perceptual loss Lp and the texture loss Lt, we
    #       normalized feature activations to have mean of one.
    sr_block2_pool = normalize(sr_images_vgg['block2_pool'])
    hd_block2_pool = normalize(hd_images_vgg['block2_pool'])
    sr_block5_pool = normalize(sr_images_vgg['block5_pool'])
    hd_block5_pool = normalize(hd_images_vgg['block5_pool'])

    loss_pool_2 = tf.losses.mean_squared_error(
        sr_block2_pool, hd_block2_pool, reduction=tf.losses.Reduction.MEAN)
    loss_pool_5 = tf.losses.mean_squared_error(
        sr_block5_pool, hd_block5_pool, reduction=tf.losses.Reduction.MEAN)

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
        sr_tensors = normalize(sr_tensors)
        hd_tensors = normalize(hd_tensors)

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

        sr_tensors = tf.reshape(sr_tensors, [-1, h * w // 256, 256, c])
        hd_tensors = tf.reshape(hd_tensors, [-1, h * w // 256, 256, c])

        # NOTE: gram matriices of shape [N, C, H*W, H*W]
        sr_grams = tf.matmul(sr_tensors, sr_tensors, transpose_a=True)
        hd_grams = tf.matmul(hd_tensors, hd_tensors, transpose_a=True)

        loss += weight * tf.losses.mean_squared_error(
            sr_grams, hd_grams, reduction=tf.losses.Reduction.MEAN)

    return loss


def build_enet(sd_images, bq_images, hd_images, pat_model, vgg19_path):
    """
    NOTE: ENet-PAT

    sd_images: -1.0 ~ +1.0
    bq_images: -1.0 ~ +1.0
    hd_images: -1.0 ~ +1.0
    """
    model = {}

    # NOTE: generator which do the super resolution
    sr_images = build_generator(sd_images, bq_images, hd_images, 'g_')

    model['sd_images'] = sd_images
    model['bq_images'] = bq_images
    model['sr_images'] = sr_images

    # NOTE: model for generating super resolved images
    if hd_images is None:
        return model

    model['hd_images'] = hd_images

    # NOTE: load vgg weights
    vgg19_weights = model_vgg.load_vgg_weights(vgg19_path)

    # NOTE: vgg features for perceptual loss and texture matching loss
    #       vgg net requires pixel values in range 0.0 ~ 255.0
    hd_vgg_input = hd_images * 127.5 + 127.5
    sr_vgg_input = sr_images * 127.5 + 127.5

    hd_images_vgg = model_vgg.build_vgg19_model(hd_vgg_input, vgg19_weights)
    sr_images_vgg = model_vgg.build_vgg19_model(sr_vgg_input, vgg19_weights)

    # NOTE: descriminate real hd images
    real = build_discriminator(hd_images, 'd_')

    # NOTE: discriminate fake hd images
    fake = build_discriminator(sr_images, 'd_')

    # NOTE: perceptual loss
    g_losses = p_loss = perceptual_loss(sr_images_vgg, hd_images_vgg)

    if 'a' in pat_model:
        # NOTE: adversarial loss
        a_loss = discriminator_loss(fake, real)

        # NOTE: generator loss
        g_loss = generator_loss(fake)

        if 't' in pat_model:
            g_losses = g_losses + g_loss * 2.0
        else:
            g_losses = g_losses + g_loss

        model['a_loss'] = a_loss
        model['g_loss'] = g_loss

    # NOTE: texture matching loss
    if 't' in pat_model:
        t_loss = texture_matching_loss(sr_images_vgg, hd_images_vgg)

        g_losses = g_losses + t_loss

        model['t_loss'] = t_loss

    # NOTE: collect variables to separate g/d training op
    d_vars = [v for v in tf.trainable_variables() if v.name.startswith('d_')]
    g_vars = [v for v in tf.trainable_variables() if v.name.startswith('g_')]

    step = tf.train.get_or_create_global_step()

    g_trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
    g_trainer = g_trainer.minimize(g_losses, var_list=g_vars, global_step=step)

    if 'a' in pat_model:
        d_trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        d_trainer = d_trainer.minimize(a_loss, var_list=d_vars)

        model['d_trainer'] = d_trainer

    model['step'] = step
    model['p_loss'] = p_loss
    model['g_loss_all'] = g_losses
    model['g_trainer'] = g_trainer

    return model

