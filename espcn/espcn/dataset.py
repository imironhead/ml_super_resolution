"""
"""
import itertools
import os

import numpy as np
import tensorflow as tf


def decode_patch_pair(scaling_factor=3):
    """
    return a method that decode lr_patch & hr_patch from one TFRecord
    with pre-defined upscaling factor.
    """
    def decode(patch_pair_record):
        features = tf.parse_single_example(
            patch_pair_record,
            features={
                'lr_pixels': tf.FixedLenFeature([], tf.string),
                'lr_height': tf.FixedLenFeature([], tf.int64),
                'lr_width': tf.FixedLenFeature([], tf.int64),
                'lr_depth': tf.FixedLenFeature([], tf.int64),
                'hr_pixels': tf.FixedLenFeature([], tf.string),
                'hr_height': tf.FixedLenFeature([], tf.int64),
                'hr_width': tf.FixedLenFeature([], tf.int64),
                'hr_depth': tf.FixedLenFeature([], tf.int64),
            })

        lrh = tf.cast(features['lr_height'], tf.int32)
        lrw = tf.cast(features['lr_width'], tf.int32)
        lrd = tf.cast(features['lr_depth'], tf.int32)

        hrh = tf.cast(features['hr_height'], tf.int32)
        hrw = tf.cast(features['hr_width'], tf.int32)
        hrd = tf.cast(features['hr_depth'], tf.int32)

        lr_patch = tf.decode_raw(features['lr_pixels'], tf.float32)
        hr_patch = tf.decode_raw(features['hr_pixels'], tf.float32)

        # NOTE: lr_patch will be convolve in the model which mean its number of
        #       channels must be know before any convolution (for we need to
        #       create a variable of kernel with known shape).
        # NOTE: lr_patch always has 3 channels (RGB)
        #       hr_patch always has 3 * (factor ** 2) channels. the (factor **
        #       2) part is the result of sub-pixel convolution.
        lr_patch = tf.reshape(lr_patch, [lrh, lrw, 3])
        hr_patch = tf.reshape(hr_patch, [hrh, hrw, 3 * (scaling_factor ** 2)])

        return lr_patch, hr_patch

    return decode


def build_image_batch_iterator(dir_path, batch_size=32, upscaling_factor=3):
    """
    read TFRecord batch. each record contains one lr_patch & one hr_patch.
    """
    # NOTE: build path list dataset, the '*' is must for google cloud storage
    path_pattern = os.path.join(dir_path, '*.tfrecord')

    data = tf.data.Dataset.list_files(path_pattern, shuffle=True)

    # NOTE: the path generator never ends
    data = data.repeat()

    # NOTE: read tfrecord
    data = tf.data.TFRecordDataset(data, num_parallel_reads=16)

    # NOTE: decode tfrecord to get lr_patch and hr_patch
    data = data.map(decode_patch_pair(upscaling_factor))

    # NOTE: combine images to batch
    data = data.batch(batch_size=batch_size)

    # NOTE: create the final iterator
    iterator = data.make_initializable_iterator()

    return iterator


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
    range_y = range(0, h - hr_patch_size, hr_patch_size)
    range_x = range(0, w - hr_patch_size, hr_patch_size)

    for x, y, u, v in itertools.product(range_x, range_y, [-1, 1], [-1, 1]):
        hr_patch = hr_image[y:y+hr_patch_size, x:x+hr_patch_size]

        lr_patch = bl_image[
            y+offset:y+offset+hr_patch_size:upscaling_factor,
            x+offset:x+offset+hr_patch_size:upscaling_factor]

        # NOTE: left-right / top-bottom flipping
        hr_patch = hr_patch[::u, ::v]
        lr_patch = lr_patch[::u, ::v]

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

        yield lr_patch, hr_patch


def int64_feature(v):
    """
    create a feature which contains a 64-bits integer
    """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[v]))


def image_feature(image):
    """
    create a feature which contains 32-bits floats in binary format.
    """
    image = image.astype(np.float32).tostring()

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[image]))


def write_patch(patch_path, lr_patch, hr_patch):
    """
    write associated patches into a TFRecord file
    """
    with tf.python_io.TFRecordWriter(patch_path) as writer:
        feature = {
            'lr_pixels': image_feature(lr_patch),
            'lr_height': int64_feature(lr_patch.shape[0]),
            'lr_width': int64_feature(lr_patch.shape[1]),
            'lr_depth': int64_feature(lr_patch.shape[2]),
            'hr_pixels': image_feature(hr_patch),
            'hr_height': int64_feature(hr_patch.shape[0]),
            'hr_width': int64_feature(hr_patch.shape[1]),
            'hr_depth': int64_feature(hr_patch.shape[2]),
        }

        example = tf.train.Example(features=tf.train.Features(feature=feature))

        writer.write(example.SerializeToString())


def main(_):
    """
    extract patches from all images in a specified directory. all patches are
    made with 2 versions (origin & low-resolution) and saved to a TFRecord.
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: enum all images in source dir
    image_paths = tf.gfile.ListDirectory(FLAGS.source_dir_path)

    for image_path in image_paths:
        # NOTE: read image and prepare low resolution version
        patches = extract_image_patches(
            os.path.join(FLAGS.source_dir_path, image_path),
            FLAGS.upscaling_factor,
            FLAGS.upscaling_factor * FLAGS.lr_patch_size)

        image_name, _ = os.path.splitext(image_path)

        # NOTE: extract image patches and save them
        for i, (lr_patch, hr_patch) in enumerate(patches):
            patch_name = image_name + '_{:0>4}.tfrecord'.format(i)
            patch_path = os.path.join(FLAGS.result_dir_path, patch_name)

            write_patch(patch_path, lr_patch, hr_patch)


if __name__ == '__main__':
    import skimage.filters
    import skimage.io

    # NOTE: extract sub-image patches from images in source_dir_path and save
    #       to result_dir_path
    tf.app.flags.DEFINE_string('source_dir_path', None, '')
    tf.app.flags.DEFINE_string('result_dir_path', None, '')

    # NOTE: arXiv:1609.05158v2, 3.2
    #       In the training phase, 17r x 17r pixel sub-images are extracted
    #       from the training ground truth images I_HR, where r is the
    #       upscaling factor.
    tf.app.flags.DEFINE_integer('upscaling_factor', 3, '')
    tf.app.flags.DEFINE_integer('lr_patch_size', 17, '')

    tf.app.run()

