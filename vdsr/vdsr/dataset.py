"""
"""
import os

import numpy as np
import skimage.io
import skimage.filters
import skimage.transform
import skimage.util
import tensorflow as tf


def hd_image_to_sd_image(hd_image, scaling_factor):
    """
    down scale image then scale back to build sd version. assume the type of
    the image is already float.
    """
    hd_h, hd_w, c = hd_image.shape

    # NOTE: downscaled size
    sd_h = int(hd_h / scaling_factor)
    sd_w = int(hd_w / scaling_factor)

    # NOTE: in scikit-image 0.14, anti_aliasing fails if mode is ;edge'
    #       https://github.com/scikit-image/scikit-image/issues/3299
    #       DIY, sigma function is from the implementation of scikit-image
    sigma = np.maximum(0.0, 0.5 * (scaling_factor - 1.0))

    bl_image = skimage.filters.gaussian(hd_image, sigma, mode='nearest')

    # NOTE: downscale then upscale
    sd_image = skimage.transform.resize(
        bl_image, [sd_h, sd_w], mode='edge', anti_aliasing=False)

    sd_image = skimage.transform.resize(
        sd_image, [hd_h, hd_w], mode='edge', anti_aliasing=False)

    return sd_image


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
        scaling_factors = [2.0, 3.0, 4.0]

    # NOTE: sanity check
    if any([s <= 1 for s in scaling_factors]):
        raise Exception('invalide scaling factors')

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

        # NOTE: cast uint8 to float32
        hd_image = skimage.util.img_as_float32(hd_image)

        # NOTE: build sd version
        scaling_factor = np.random.choice(scaling_factors)

        sd_image = hd_image_to_sd_image(hd_image, scaling_factor)

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

