"""
"""
import os

import numpy as np
import skimage.filters
import skimage.io
import tensorflow as tf


def extract_image_patches(hr_image_path, upscaling_factor, hr_patch_size):
    """
    arXiv:1609.05158v2, 3.2
    this ensures that all pixels in the original image appear once and only
    once as the ground truth of the training data.

    do not know what the f is.
    """
    # NOTE: read image, images may come from google cloud storage
    with tf.gfile.GFile(hr_image_path, 'rb') as hr_image_file:
        hr_image = skimage.io.imread(hr_image_file)

    # NOTE: adjust pixel range from [0.0, 1.0] to [-1.0, 1.0]
    hr_image = hr_image / 127.5 - 1.0

    # NOTE: make gaussian blur version
    sigma = np.maximum(0.0, 0.5 * (upscaling_factor - 1.0))

    # NOTE: the input image is converted according to the conventions of
    #       img_as_float.
    bl_image = skimage.filters.gaussian(hr_image, sigma, mode='nearest')

    offset = upscaling_factor // 2

    h, w, c = hr_image.shape

    lr_patch_size = hr_patch_size // upscaling_factor

    # generator
    for y in range(0, h - hr_patch_size, hr_patch_size):
        for x in range(0, w - hr_patch_size, hr_patch_size):
            hr_patch = hr_image[y:y+hr_patch_size, x:x+hr_patch_size]

            lr_patch = bl_image[
                y+offset:y+offset+hr_patch_size:upscaling_factor,
                x+offset:x+offset+hr_patch_size:upscaling_factor]

            # NOTE: reshape for espcn training
            # NOTE: assume hr_patch is (mono channel)
            #       [
            #           [[1.1], [1.2], [2.1], [2.2]],
            #           [[1.3], [1.4], [2.3], [2.4]],
            #           [[3.1], [3.2], [4.1], [4.2]],
            #           [[3.3], [3.4], [4.3], [4.4]],
            #       ]
            #
            #       split to 2 parts along axis 1 (horizontally)
            #       [                   |   [
            #           [[1.1], [1.2]], |       [[2.1], [2.2]],
            #           [[1.3], [1.4]], |       [[2.3], [2.4]],
            #           [[3.1], [3.2]], |       [[4.1], [4.2]],
            #           [[3.3], [3.4]], |       [[4.3], [4.4]],
            #       ]                   |   ]
            hr_sub_patches = np.split(hr_patch, lr_patch_size, axis=1)

            # NOTE: reshape each sub patch:
            #       [                           |   [
            #           [[1.1, 1.2, 1.3, 1.4]], |       [[2.1, 2.2, 2.3, 2.4]],
            #           [[3.1, 3.2, 3.3, 3.4]], |       [[4.1, 4.2, 4.3, 4.4]],
            #       ]                           |   ]
            new_shape = [lr_patch_size, 1, -1]

            hr_sub_patches = [hsp.reshape(new_shape) for hsp in hr_sub_patches]

            # NOTE: concat to make final label for expcn
            #       [
            #           [[1.1, 1.2, 1.3, 1.4], [2.1, 2.2, 2.3, 2.4]],
            #           [[3.1, 3.2, 3.3, 3.4], [4.1, 4.2, 4.3, 4.4]],
            #       ]
            hr_patch = np.concatenate(hr_sub_patches, axis=1)

            yield hr_patch, lr_patch


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: enum all images in source dir
    image_paths = tf.gfile.ListDirectory(FLAGS.source_dir_path)

    for image_path in image_paths:
        # NOTE: read image and prepare low resolution version
        patches = extract_image_patches(
            os.path.join(FLAGS.source_dir_path, image_path),
            FLAGS.upscaling_factor,
            FLAGS.upscaling_factor * FLAGS.low_resolution_patch_size)

        image_name, _ = os.path.splitext(image_path)

        # NOTE: extract image patches and save them
        for i, (hr_patch, lr_patch) in enumerate(patches):
            hr_patch_name = image_name + '_{:0>4}_hr.npy'.format(i)
            lr_patch_name = image_name + '_{:0>4}_lr.npy'.format(i)

            hr_patch_path = os.path.join(FLAGS.result_dir_path, hr_patch_name)
            lr_patch_path = os.path.join(FLAGS.result_dir_path, lr_patch_name)

            # NOTE: do not know how to write to a npz file (zip inside).
            #       eather 'wb' or 'w+b' has permission problems
            with tf.gfile.GFile(hr_patch_path, 'wb') as hr_patch_file:
                np.save(hr_patch_file, hr_patch)

            with tf.gfile.GFile(lr_patch_path, 'wb') as lr_patch_file:
                np.save(lr_patch_file, lr_patch)


if __name__ == '__main__':
    # NOTE: extract sub-image patches from images in source_dir_path and save
    #       to result_dir_path
    tf.app.flags.DEFINE_string('source_dir_path', None, '')
    tf.app.flags.DEFINE_string('result_dir_path', None, '')

    # NOTE: arXiv:1609.05158v2, 3.2
    #       In the training phase, 17r x 17r pixel sub-images are extracted
    #       from the training ground truth images I_HR, where r is the
    #       upscaling factor.
    tf.app.flags.DEFINE_integer('upscaling_factor', 3, '')
    tf.app.flags.DEFINE_integer('low_resolution_patch_size', 17, '')

    tf.app.run()

