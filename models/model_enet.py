"""
Replicate EnhanceNet.

arxiv: 1612.07915
title: EnhanceNet: Single Image Super-Resolution Through Automated Texture
       Synthesis
"""
import functools

import tensorflow as tf

from . import model_vgg


def normalize(tensors):
    """
    Arxiv 1612.07915, 5.3,
    for the perceptual loss Lp and the texture loss Lt, we normalized feature
    activations to have mean of one.
    """
    mean = tf.reduce_mean(tensors, axis=-1, keepdims=True)

    return tensors / (mean + 0.000001)


class ResidualLayer(tf.keras.layers.Layer):
    """
    Implement a basic residual layer.
    """
    def __init__(self, filters):
        """
        Initialize layers.
        """
        super().__init__()

        self._layer_0 = tf.keras.layers.Convolution2D(
            filters=filters,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu')

        self._layer_1 = tf.keras.layers.Convolution2D(
            filters=filters,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None)

        self._layer_add = tf.keras.layers.Add()
        self._layer_relu = tf.keras.layers.ReLU()

    def call(self, tensors):
        """
        Do the residual layer job.
        """
        raw_tensors = tensors

        tensors = self._layer_0(tensors)
        tensors = self._layer_1(tensors)
        tensors = self._layer_add([tensors, raw_tensors])
        tensors = self._layer_relu(tensors)

        return tensors


class Generator(tf.keras.Model):
    """
    Implement the generator of ENet.
    """
    def __init__(self):
        """
        Build generator layers.
        """
        super().__init__()

        self._generator_layers = []

        # NOTE: arXiv: 1612.07919v2, table 1
        # NOTE: To 64 channels.
        self._generator_layers.append(tf.keras.layers.Convolution2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu'))

        # NOTE: arXiv: 1612.07919v2, table 1,
        #       We use 3x3 comvolution kernels, 10 residual blocks and RGB
        #       images.
        self._generator_layers.extend([ResidualLayer(64) for _ in range(10)])

        # NOTE: arXiv: 1612.07919v2, table 1, nearest neighbor upsampling
        for _ in range(2):
            self._generator_layers.append(tf.keras.layers.UpSampling2D())

            self._generator_layers.append(tf.keras.layers.Convolution2D(
                filters=64,
                kernel_size=3,
                strides=1,
                padding='same',
                activation='relu'))

        # NOTE: one more?
        self._generator_layers.append(tf.keras.layers.Convolution2D(
            filters=64,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu'))

        # NOTE: to residual image, how should we clamp the final image?
        self._generator_layers.append(tf.keras.layers.Convolution2D(
            filters=3,
            kernel_size=3,
            strides=1,
            padding='same',
            activation=None))

        self._layer_add = tf.keras.layers.Add()

    @tf.function
    def call(self, sd_images, bq_images):
        """
        Required:
        - sd_images
            Tensors of low resolution images.

        - bq_images
            Tensors of low resolution images but in 4x size. Bicubic up-scaled
            from sd_images as the paper.

        Return:
        - sr_images
            Tensors of super resolved images.
        """
        tensors = functools.reduce(
            lambda tensors, layer: layer(tensors),
            self._generator_layers,
            sd_images)

        tensors = self._layer_add([tensors, bq_images])

        return tensors


class Discriminator(tf.keras.Model):
    """
    arXiv: 1612.07919, supplementary, table 1
    """
    def __init__(self):
        """
        Build discriminator layers.
        """
        super().__init__()

        self._discriminator_layers = []

        for i in range(5):
            filters = 2 ** (i + 5)

            self._discriminator_layers.append(tf.keras.layers.Convolution2D(
                filters=filters,
                kernel_size=3,
                strides=1,
                padding='same',
                activation=tf.nn.leaky_relu))

            self._discriminator_layers.append(tf.keras.layers.Convolution2D(
                filters=filters,
                kernel_size=3,
                strides=2,
                padding='same',
                activation=tf.nn.leaky_relu))

        self._discriminator_layers.append(tf.keras.layers.Flatten())

        self._discriminator_layers.append(
            tf.keras.layers.Dense(1024, activation=tf.nn.leaky_relu))

        self._discriminator_layers.append(
            tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

    @tf.function
    def call(self, hd_images):
        """
        Required:
        - hd_images:
            Tensors of high resolution images for discrimination.

        Return:
            Tensors of confidence (sigmoid).
        """
        return functools.reduce(
            lambda tensors, layer: layer(tensors),
            self._discriminator_layers,
            hd_images)


class GeneratorTrace:
    """
    Implement the network for reducing perceptual loss, texture matching loss,
    generator loss (fool the discriminator).
    """
    def __init__(self, generator, discriminator, vgg19):
        """
        Keep core models & build loss functions.
        """
        self._generator = generator
        self._discriminator = discriminator
        self._vgg19 = vgg19
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self._mean_squared_error = tf.keras.losses.MeanSquaredError(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def perceptual_loss(self, sr_images_vgg, hd_images_vgg):
        """
        arXiv:1612.07915v2, 4.2.2,
        To capture both low-level and high-level features, we use a combination
        of the second and fifth pooling layers and compute the MSE on their
        feature activations.
        """
        # NOTE: arXiv:1612.07915, 5.3,
        #       For the perceptual loss Lp and the texture loss Lt, we
        #       normalized feature activations to have mean of one.
        sr_block2_pool = normalize(sr_images_vgg['block2_pool'])
        hd_block2_pool = normalize(hd_images_vgg['block2_pool'])
        sr_block5_pool = normalize(sr_images_vgg['block5_pool'])
        hd_block5_pool = normalize(hd_images_vgg['block5_pool'])

        loss_pool_2 = self._mean_squared_error(hd_block2_pool, sr_block2_pool)
        loss_pool_5 = self._mean_squared_error(hd_block5_pool, sr_block5_pool)

        # NOTE: arXiv:1612.07915, weights come from table 2
        loss = 0.2 * loss_pool_2 + 0.02 * loss_pool_5

        return loss

    def texture_matching_loss(self, sr_images_vgg, hd_images_vgg):
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

            height, width, num_channels = shape[1], shape[2], shape[3]

            # NOTE: 5.3 for the perceptual loss Lp and the texture loss Lt, we
            #       normalized feature activations to have mean of one.
            sr_tensors = normalize(sr_tensors)
            hd_tensors = normalize(hd_tensors)

            # NOTE: 4.2.3 empirically, we found a patch size of 16x16 pixels to
            #       result in the best balance between faithful texture
            #       generation and the overall perceptual quality of the
            #       images.
            # NOTE: [N, H/16, W/16, 16*16*C]
            sr_tensors = tf.image.extract_patches(
                sr_tensors,
                sizes=[1, 16, 16, 1],
                strides=[1, 16, 16, 1],
                rates=[1, 1, 1, 1],
                padding='VALID')

            hd_tensors = tf.image.extract_patches(
                hd_tensors,
                sizes=[1, 16, 16, 1],
                strides=[1, 16, 16, 1],
                rates=[1, 1, 1, 1],
                padding='VALID')

            new_shape = [-1, height * width // 256, 256, num_channels]

            sr_tensors = tf.reshape(sr_tensors, new_shape)
            hd_tensors = tf.reshape(hd_tensors, new_shape)

            # NOTE: Gram matriices of shape [N, C, H*W, H*W].
            sr_grams = \
                tf.linalg.matmul(sr_tensors, sr_tensors, transpose_a=True)
            hd_grams = \
                tf.linalg.matmul(hd_tensors, hd_tensors, transpose_a=True)

            loss += weight * self._mean_squared_error(hd_grams, sr_grams)

        return loss

    def __call__(self, sd_images, bq_images, hd_images):
        """
        Required:
        - sd_images:
            Tensors of low resolution images for super resolution.
        - bq_images:
            Tensors of bicubic up-scaled sd_images.
        - hd_images:
            Tensors of high resolution images, as ground truth.

        Return:
            Information to train the model for one step.
        """
        with tf.GradientTape() as tape:
            sr_images = self._generator(sd_images, bq_images)

            # NOTE: VGG features for perceptual loss and texture matching loss.
            #       VGG net requires pixel values in range 0.0 ~ 255.0.
            sr_images_vgg = sr_images * 127.5 + 127.5
            hd_images_vgg = hd_images * 127.5 + 127.5

            # NOTE: VGG net is pre-trained in BGR channel order.
            sr_images_vgg = tf.reverse(sr_images_vgg, [-1])
            hd_images_vgg = tf.reverse(hd_images_vgg, [-1])

            sr_images_vgg = self._vgg19(sr_images_vgg)
            hd_images_vgg = self._vgg19(hd_images_vgg)

            loss_perceptual = \
                self.perceptual_loss(sr_images_vgg, hd_images_vgg)

            loss_texture_matching = \
                self.texture_matching_loss(sr_images_vgg, hd_images_vgg)

            # NOTE:
            fake = self._discriminator(sr_images)

            loss_generator = self._cross_entropy(tf.ones_like(fake), fake)

            loss = \
                2.0 * loss_generator + loss_perceptual + loss_texture_matching

        gradients = tape.gradient(loss, self._generator.trainable_variables)

        return {
            'loss': loss,
            'variables': self._generator.trainable_variables,
            'gradients': gradients
        }


class DiscriminatorTrace:
    """
    Implement the network for reducing discriminator loss (denfend generator).
    """
    def __init__(self, generator, discriminator):
        """
        Keep core models & build functions.
        """
        self._generator = generator
        self._discriminator = discriminator
        self._cross_entropy = tf.keras.losses.BinaryCrossentropy(
            reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def __call__(self, sd_images, bq_images, hd_images):
        """
        Required:
        - sd_images
            Tensors of low resolution images for super resolution.
        - bq_images
            Tensors of images which are 4x bicubuc up-scaled from sd_images.
        - hd_images
            Tensors of high resolution images, as ground truth.

        Return:
            Information to train the model for one step.
        """
        with tf.GradientTape() as tape:
            sr_images = self._generator(sd_images, bq_images)

            real = self._discriminator(hd_images)
            fake = self._discriminator(sr_images)

            real_loss = self._cross_entropy(tf.ones_like(real), real)
            fake_loss = self._cross_entropy(tf.zeros_like(fake), fake)

            loss = real_loss + fake_loss

        gradients = tape.gradient(
            loss, self._discriminator.trainable_variables)

        return {
            'loss': loss,
            'variables': self._discriminator.trainable_variables,
            'gradients': gradients,
        }


def build_models(**kwargs):
    """
    Build sub models for training ENet.
    """
    generator = Generator()
    discriminator = Discriminator()
    vgg19 = model_vgg.VGG19(kwargs['vgg19_weights_path'])

    generator_trace = GeneratorTrace(generator, discriminator, vgg19)
    discriminator_trace = DiscriminatorTrace(generator, discriminator)

    return {
        'extensions': {
            'generator': generator_trace,
            'discriminator': discriminator_trace,
        },
        'principals': {
            'generator': generator,
            'discriminator': discriminator,
        }
    }
