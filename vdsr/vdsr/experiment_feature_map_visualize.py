"""
"""
import os

import numpy as np
import skimage.io
import skimage.util
import tensorflow as tf

import vdsr.dataset as dataset


def load_sd_images():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: read image, images may come from google cloud storage
    with tf.gfile.GFile(FLAGS.hd_image_path, 'rb') as hd_image_file:
        hd_image = skimage.io.imread(hd_image_file)

    # NOTE: cast uint8 to float32
    hd_image = skimage.util.img_as_float32(hd_image)

    sd_image = dataset.hd_image_to_sd_image(hd_image, FLAGS.scaling_factor)

    # NOTE: from [0.0, 1.0] to [-1.0, +1.0]
    sd_image = sd_image * 2.0 - 1.0

    sd_images = np.expand_dims(sd_image, axis=0)

    return sd_images


def build_model(session):
    """
    """
    # NOTE: restore the model
    FLAGS = tf.app.flags.FLAGS

    saver = tf.train.import_meta_graph(FLAGS.meta_path)

    saver.restore(session, FLAGS.ckpt_path)

    graph = tf.get_default_graph()

    # NOTE: collect named tensors
    model = {
        'sd_images': graph.get_tensor_by_name('sd_images:0'),
        'sr_images': graph.get_tensor_by_name('sr_images:0'),
    }

    for i in range(1, 30):
        conv_name = 'conv.{}:0'.format(i)
        relu_name = 'relu.{}:0'.format(i)

        try:
            model[conv_name] = graph.get_tensor_by_name(conv_name)
            model[relu_name] = graph.get_tensor_by_name(relu_name)
        except KeyError:
            # NOTE: reach the end (default is 20 layers)
            break

    return model


def encode_image(tensors):
    """
    encode a single image (sd/sr/residual)
    """
    img_tensor = tf.squeeze(tensors, [0])

    png_tensor = tf.saturate_cast(img_tensor * 127.5 + 127.5, tf.uint8)

    png_tensor = tf.image.encode_png(png_tensor)

    return png_tensor


def encode_feature_map(tensors):
    """
    """
    # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very deep
    #       convolutional networks, 3.1
    #
    #       we use d layers where layers except the first and the last are of
    #       the same type: 64 filters of the size 3x3x64 where a filter
    #       operates on 3x3 spatial region across 64 channels (feature maps).
    shape = tf.shape(tensors)

    h, w, c = shape[1], shape[2], shape[3]

    feature_tensors = tf.squeeze(tensors, [0])

    # NOTE: split to results of 64 filters
    feature_tensors = tf.split(feature_tensors, 64, axis=-1)

    # NOTE: concat to build rows
    feature_tensors = \
        [tf.concat(feature_tensors[i:i+8], axis=1) for i in range(0, 64, 8)]

    # NOTE: concat to build entire map
    feature_tensors = tf.concat(feature_tensors, axis=0)

    # NOTE: encode as png
    png_tensor = tf.saturate_cast(feature_tensors * 127.5 + 127.5, tf.uint8)

    png_tensor = tf.image.encode_png(png_tensor)

    return png_tensor


def build_feature_maps(model):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    feature_maps = {}

    # NOTE: source sd image
    feature_maps['sd_image'] = {
        'tensor': encode_image(model['sd_images']),
        'path': os.path.join(FLAGS.result_dir_path, 'sd_image.png'),
    }

    # NOTE: result sr image
    feature_maps['sr_image'] = {
        'tensor': encode_image(model['sr_images']),
        'path': os.path.join(FLAGS.result_dir_path, 'sr_image.png'),
    }

    # NOTE: feature maps
    for i in range(1, 30):
        conv_name = 'conv.{}:0'.format(i)
        relu_name = 'relu.{}:0'.format(i)

        conv_png = 'conv.{}.png'.format(i)
        relu_png = 'relu.{}.png'.format(i)

        if relu_name not in model:
            # NOTE: this is the latest layer
            feature_maps[conv_name] = {
                'tensor': encode_image(model[conv_name]),
                'path': os.path.join(FLAGS.result_dir_path, conv_png),
            }
            break

        # NOTE: encode intermediate conv feature maps
        feature_maps[conv_name] = {
            'tensor': encode_feature_map(model[conv_name]),
            'path': os.path.join(FLAGS.result_dir_path, conv_png),
        }

        # NOTE: encode intermediate relu feature maps
        feature_maps[relu_name] = {
            'tensor': encode_feature_map(model[relu_name]),
            'path': os.path.join(FLAGS.result_dir_path, relu_png),
        }

    return feature_maps


def main(_):
    """
    """
    sd_images = load_sd_images()

    with tf.Session() as session:
        model = build_model(session)

        feature_maps = build_feature_maps(model)

        fetch = {k: feature_maps[k]['tensor'] for k in feature_maps}

        feeds = {model['sd_images']: sd_images}

        fetched = session.run(fetch, feed_dict=feeds)

        for key in fetched:
            with tf.gfile.GFile(feature_maps[key]['path'], 'wb') as f:
                f.write(fetched[key])


if __name__ == '__main__':
    # NOTE: path to the meta for this experiment
    tf.app.flags.DEFINE_string('meta_path', None, 'path to the graph')

    # NOTE: path to the weights for this experiment
    tf.app.flags.DEFINE_string('ckpt_path', None, 'path to the weights')

    # NOTE: path to a image. we will build its sd version and super resolve it.
    tf.app.flags.DEFINE_string(
        'hd_image_path', None, 'path to the hd image for this experiment')

    # NOTE:
    tf.app.flags.DEFINE_string(
        'result_dir_path', None, 'path to a directory for keeping results')

    # NOTE: scaling factor for building the sd version image.
    tf.app.flags.DEFINE_integer(
        'scaling_factor', 2, 'scaling factor for this experiment')

    tf.app.run()

