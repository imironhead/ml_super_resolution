"""
"""
import os

import numpy as np
import skimage.io
import skimage.transform
import skimage.util
import tensorflow as tf


def image_batches(source_dir_path, scaling_factors, image_size, batch_size):
    """
    source_dir_path:
        read images inside this directory
    scaling_factors:
        different scale factors for training
    image_size:
        randomly crop all images to image_size (width) x image_size (width)
    batch_size:
        number of images per batch
    """
    def image_paths():
        """
        a generator that yields image path within source_dir_path in random
        order
        """
        paths = tf.gfile.ListDirectory(source_dir_path)

        while True:
            np.random.shuffle(paths)

            for path in paths:
                yield os.path.join(source_dir_path, path)

    # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very deep
    #       convolutional networks
    #       figure 5
    if scaling_factors is None or len(scaling_factors) <= 0:
        scaling_factors = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

    # NOTE: sanity check
    if any([s <= 1 for s in scaling_factors]):
        raise Exception('invalide scaling factors')

    # NOTE: to downscale sizes
    scaling_sizes = [int(image_size / s) for s in scaling_factors]

    # NOTE: infinit image path generator
    image_path_generator = image_paths()

    # NOTE: container to keep images in a batch
    sd_images = []
    hd_images = []

    while True:
        hd_image_path = next(image_path_generator)

        # NOTE: read image, images may come from google cloud storage
        with tf.gfile.GFile(hd_image_path, 'rb') as hd_image_file:
            hd_image = skimage.io.imread(hd_image_file)

        # NOTE: drop small images
        h, w, c = hd_image.shape

        if h < image_size or w < image_size or c != 3:
            continue

        # NOTE: data augmentation, random crop to image_size x image_size x 3
        x = np.random.randint(w - image_size)
        y = np.random.randint(h - image_size)

        hd_image = hd_image[y:y+image_size, x:x+image_size, :]

        # NOTE: data augmentation, random horizontal flip
        if 1 == np.random.choice([0, 1]):
            hd_image = hd_image[:, ::-1, :]

        # NOTE: build sd version
        size = np.random.choice(scaling_sizes)

        sd_image = skimage.transform.resize(
            hd_image, [size, size], mode='edge')

        sd_image = skimage.transform.resize(
            sd_image,
            [image_size, image_size],
            mode='edge')

        # NOTE: cast uint8 to float32
        sd_image = skimage.util.img_as_float32(sd_image)
        hd_image = skimage.util.img_as_float32(hd_image)

        # NOTE: from [0.0, 1.0] to [-1.0, +1.0]
        sd_image = sd_image * 2.0 - 1.0
        hd_image = hd_image * 2.0 - 1.0

        sd_images.append(sd_image)
        hd_images.append(hd_image)

        # NOTE: yield a collected batch
        if len(sd_images) == batch_size:
            sd_images = np.stack(sd_images, axis=0)
            hd_images = np.stack(hd_images, axis=0)

            yield sd_images, hd_images

            sd_images = []
            hd_images = []

