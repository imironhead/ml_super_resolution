"""
Replicate EnhanceNet.

arxiv: 1612.07915
title: EnhanceNet: Single Image Super-Resolution Through Automated Texture
       Synthesis
"""
import functools

import tensorflow as tf

from . import model_vgg


def normalize(tensor):
    """
    Arxiv 1612.07915, 5.3,
    For the perceptual loss Lp and the texture loss Lt, we normalized feature
    activations to have mean of one.

    Arguments:
        tensor: A tensor to be normalized.

    Return:
        A normalized tensor with the same shape as the input tensor.
    """
    mean = tf.math.reduce_mean(tensor, axis=-1, keepdims=True)

    return tensor / (mean + 0.000001)


def mean_squared_error(tensor_a, tensor_b):
    """
    Compute the mean square errors of 2 tensors. Elementwise square errors are
    firstly reduce summed. Then all elements across entire batch are averaged
    to build the final tensor.

    Arguments:
        tensor_a: A tensor.
        tensor_b: A tensor with the same shape of tensor_a.

    Return:
        All dimensions reduced mean squared error.
    """
    tensor = tf.math.subtract(tensor_a, tensor_b)
    tensor = tf.math.square(tensor)
    tensor = tf.math.reduce_sum(tensor, axis=-1)
    tensor = tf.math.reduce_mean(tensor)

    return tensor


def mean_sigmoid_cross_entropy(labels, logits):
    """
    Compute the average sigmoid cross entropy over all elements.

    Arguments:
        labels: The label tensor.
        logits: The logit tensor. Note that it shouldn't have been sigmoid
            applied.

    Return:
        Averaged binary cross entropy over all elements.
    """
    tensor = tf.nn.sigmoid_cross_entropy_with_logits(labels, logits)
    tensor = tf.math.reduce_mean(tensor)

    return tensor


def mean_real_sigmoid_cross_entropy(logits):
    """
    Work like mean_sigmoid_cross_entropy and all labels are ones.

    Arguments:
        logits: The logit tensor. Note that it shouldn't have been sigmoid
            applied.

    Return:
        Averaged binary cross entropy over all elements.
    """
    return mean_sigmoid_cross_entropy(tf.ones_like(logits), logits)


def mean_fake_sigmoid_cross_entropy(logits):
    """
    Work like mean_sigmoid_cross_entropy and all labels are zeros.

    Arguments:
        logits: The logit tensor. Note that it shouldn't have been sigmoid
            applied.

    Return:
        Averaged binary cross entropy over all elements.
    """
    return mean_sigmoid_cross_entropy(tf.zeros_like(logits), logits)


class ResidualLayer(tf.keras.layers.Layer):
    """
    Implement a basic residual layer.
    """

    def __init__(self, filters, **kwargs):
        """
        Initialize sub-layers for residual layer.

        Arguments:
            filters: The size of the last dimention of the output tensor.
        """
        super().__init__(**kwargs)

        self._filters = filters
        self._layer_0 = None
        self._layer_1 = None
        self._layer_add = None
        self._layer_relu = None

    def build(self, input_shape):
        """
        Build layers for the model.

        Parameters:
            input_shape: Known input shape when building the mode. Note that we
                do not need to call this method. Tensorflow will do the trick.
        """
        self._layer_0 = tf.keras.layers.Convolution2D(
            filters=self._filters,
            kernel_size=3,
            strides=1,
            padding='same',
            activation='relu')

        self._layer_1 = tf.keras.layers.Convolution2D(
            filters=self._filters,
            kernel_size=1,
            strides=1,
            padding='same',
            activation=None)

        self._layer_add = tf.keras.layers.Add()
        self._layer_relu = tf.keras.layers.ReLU()

    def call(self, inputs):
        """
        Compute the residual block.

        Arguments:
            inputs: A tensor.

        Return:
            A tensor.
        """
        tensor = self._layer_0(inputs)
        tensor = self._layer_1(tensor)
        tensor = self._layer_add([tensor, inputs])
        tensor = self._layer_relu(tensor)

        return tensor


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
        self._layer_add = None

    def build(self, input_shape):
        """
        Build layers for the model.

        Parameters:
            input_shape: Known input shape when building the mode. Note that we
                do not need to call this method. Tensorflow will do the trick.
        """
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
    def call(self, inputs):
        """
        Generate super resolved image tensor.

        Arguments:
            inputs: A list with 2 tensors. The first tensor represents the
                image to be super resolved. The second one represent the
                bicubic up scaled version (4x) of the first one.

        Return:
            A tensor represents super-resolved images.
        """
        sd_image, bq_image = inputs

        tensor = functools.reduce(
            lambda tensor_, layer: layer(tensor_),
            self._generator_layers,
            sd_image)

        tensor = self._layer_add([tensor, bq_image])

        return tensor


class Discriminator(tf.keras.Model):
    """
    arXiv: 1612.07919, supplementary, table 1
    Implement the discriminator network of ENet.
    """

    def __init__(self):
        """
        Build discriminator layers.
        """
        super().__init__()

        self._discriminator_layers = []

    def build(self, input_shape):
        """
        Build layers for the model.

        Parameters:
            input_shape: Known input shape when building the mode. Note that we
                do not need to call this method. Tensorflow will do the trick.
        """
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
            tf.keras.layers.Dense(1, activation=None))

    @tf.function
    def call(self, inputs):
        """
        Discriminate if a tensor is real.

        Arguments:
            inputs: A tensor represents images which should be high resolution
                or super resolved image.

        Return:
            Tensors of confidence (without sigmoid).
        """
        return functools.reduce(
            lambda tensor, layer: layer(tensor),
            self._discriminator_layers,
            inputs)


class GeneratorTrace(tf.keras.Model):
    """
    Implement the network for reducing perceptual loss, texture matching loss,
    generator loss (fool the discriminator).
    """

    def __init__(self, generator, discriminator, vgg19):
        """
        Keep core models & build loss functions.

        Arguments:
            generator: A network which accepts a low resolution image tensor
                (1x) and a bicubic up-scaled image tensor (4x). Return a super-
                resolved image tensor.
            discriminator: A network which accepts a image tensor and return
                logits.
            vgg19: A network of VGG 19 which will not be trained.
        """
        super().__init__()

        self._generator = generator
        self._discriminator = discriminator
        self._vgg19 = vgg19

    @property
    def trainable_variables(self):
        """
        Return trainable variables of the network. When training this network,
        we only want to train its generator part.
        """
        return self._generator.trainable_variables

    def perceptual_loss(self, sr_images_vgg, hd_images_vgg):
        """
        arXiv:1612.07915v2, 4.2.2,
        To capture both low-level and high-level features, we use a combination
        of the second and fifth pooling layers and compute the MSE on their
        feature activations.

        Arguments:
            sr_images_vgg: Layers of VGG19 with super-resolved image tensor.
            hd_images_vgg: Layers of VGG19 with high definition image tensor.

        Return:
            Perceptual loss between 2 tensors.
        """
        # NOTE: arXiv:1612.07915, 5.3,
        #       For the perceptual loss Lp and the texture loss Lt, we
        #       normalized feature activations to have mean of one.
        sr_block2_pool = normalize(sr_images_vgg['block2_pool'])
        hd_block2_pool = normalize(hd_images_vgg['block2_pool'])
        sr_block5_pool = normalize(sr_images_vgg['block5_pool'])
        hd_block5_pool = normalize(hd_images_vgg['block5_pool'])

        loss_pool_2 = mean_squared_error(hd_block2_pool, sr_block2_pool)
        loss_pool_5 = mean_squared_error(hd_block5_pool, sr_block5_pool)

        # NOTE: arXiv:1612.07915, weights come from table 2
        loss = 0.2 * loss_pool_2 + 0.02 * loss_pool_5

        return loss

    def texture_matching_loss(self, sr_images_vgg, hd_images_vgg):
        """
        arXiv:1612.07919v2, 4.2.3
        Compute texture matching loss with 2 VGG19 networks.

        Arguments:
            sr_images_vgg: Layers of VGG19 with super-resolved image tensor.
            hd_images_vgg: Layers of VGG19 with high definition image tensor.

        Return:
            Texture matching loss between 2 tensors.
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

            loss += weight * mean_squared_error(hd_grams, sr_grams)

        return loss

    @tf.function
    def call(self, inputs):
        """
        Compute the loss of generating super resolved image tensor.

        Arguments:
            inputs: A list consists of 3 tensors. The 1st tensor represents low
                resolution images. The 2nd tensor represents bicubic up-scaled
                version of the 1st one. The 3rd tensor represents high
                definition images.

        Return:
            Loss of generating super resolved image tensor.
        """
        sd_images, bq_images, hd_images = inputs

        sr_images = self._generator([sd_images, bq_images])

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

        loss_generator = mean_real_sigmoid_cross_entropy(fake)

        loss = 2.0 * loss_generator + loss_perceptual + loss_texture_matching

        return loss


class DiscriminatorTrace(tf.keras.Model):
    """
    Implement the network for reducing discriminator loss (denfend generator).
    """

    def __init__(self, generator, discriminator):
        """
        Keep core models & build functions.

        Arguments:
            generator: A network which accepts a low resolution image tensor
                (1x) and a bicubic up-scaled image tensor (4x). Return a super-
                resolved image tensor.
            discriminator: A network which accepts a image tensor and return
                logits.
        """
        super().__init__()

        self._generator = generator
        self._discriminator = discriminator

    @property
    def trainable_variables(self):
        """
        Return trainable variables of the network. When training this network,
        we only want to train its discriminator part.
        """
        return self._discriminator.trainable_variables

    @tf.function
    def call(self, inputs):
        """
        Compute the loss of discriminating super resolved and high definition
        image tensors.

        Arguments:
            inputs: A list consists of 3 tensors. The 1st tensor represents low
                resolution images. The 2nd tensor represents bicubic up-scaled
                version of the 1st one. The 3rd tensor represents high
                definition images.

        Return:
            The loss of discriminating super resolved and high definition
            image tensors.
        """
        sd_images, bq_images, hd_images = inputs

        sr_images = self._generator([sd_images, bq_images])

        real = self._discriminator(hd_images)
        fake = self._discriminator(sr_images)

        real_loss = mean_real_sigmoid_cross_entropy(real)
        fake_loss = mean_fake_sigmoid_cross_entropy(fake)

        loss = real_loss + fake_loss

        return loss


def build_models(**kwargs):
    """
    Build sub models for training ENet.

    Arguments:
        vgg19_weights_path: Path to the pre-trained VGG19 weights.
    """
    if 'vgg19_weights_path' not in kwargs:
        raise ValueError('ENet needs pre-trained VGG19 weights.')

    generator = Generator()
    discriminator = Discriminator()
    vgg19 = model_vgg.VGG19(kwargs['vgg19_weights_path'])

    generator_trace = GeneratorTrace(generator, discriminator, vgg19)
    discriminator_trace = DiscriminatorTrace(generator, discriminator)

    return {
        # NOTE: We will train extension models.
        'extensions': {
            'generator': generator_trace,
            'discriminator': discriminator_trace,
        },
        # NOTE: Extension models use shared weights of principal models. We
        #       need principal model to save trained weights.
        'principals': {
            'generator': generator,
            'discriminator': discriminator,
        }
    }
