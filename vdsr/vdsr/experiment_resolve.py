"""
"""
import os
import time

import numpy as np
import skimage.io
import skimage.transform
import skimage.util
import tensorflow as tf

import vdsr.dataset as dataset


def load_image(hd_image_path, scaling_factor, ground_truth_mode):
    """
    """
    # NOTE: read image, images may come from google cloud storage
    with tf.gfile.GFile(hd_image_path, 'rb') as hd_image_file:
        hd_image = skimage.io.imread(hd_image_file)

    # NOTE: cast uint8 to float32
    hd_image = skimage.util.img_as_float32(hd_image)

    if ground_truth_mode:
        sd_image = dataset.hd_image_to_sd_image(hd_image, scaling_factor)
    else:
        hd_h, hd_w, _ = hd_image.shape

        sd_h = int(scaling_factor * hd_h)
        sd_w = int(scaling_factor * hd_w)

        sd_image = skimage.transform.resize(
            hd_image, [sd_h, sd_w], mode='edge', anti_aliasing=False)

    # NOTE: from [0.0, 1.0] to [-1.0, +1.0]
    sd_image = sd_image * 2.0 - 1.0
    hd_image = hd_image * 2.0 - 1.0

    # NOTE: to batch shape
    sd_images = np.expand_dims(sd_image, axis=0)
    hd_images = np.expand_dims(hd_image, axis=0)

    return sd_images, hd_images


def build_model(session):
    """
    """
    # NOTE: restore the model
    FLAGS = tf.app.flags.FLAGS

    saver = tf.train.import_meta_graph(FLAGS.meta_path)

    saver.restore(session, FLAGS.ckpt_path)

    graph = tf.get_default_graph()

    # NOTE: collect named tensors
    sd_images = graph.get_tensor_by_name('sd_images:0')
    sr_images = graph.get_tensor_by_name('sr_images:0')
    hd_images = graph.get_tensor_by_name('hd_images:0')

    # NOTE: encode super resolved image as png
    img_tensor = sr_images[0]

    png_tensor = tf.saturate_cast(img_tensor * 127.5 + 127.5, tf.uint8)

    png_tensor = tf.image.encode_png(png_tensor)

    return {
        'sd_images': sd_images,
        'sr_images': sr_images,
        'hd_images': hd_images,
        'sr_image_png': png_tensor,
        'hd_sd_psnrs': tf.image.psnr(hd_images, sd_images, 2.0),
        'hd_sr_psnrs': tf.image.psnr(hd_images, sr_images, 2.0),
        'hd_sd_ssims': tf.image.ssim(hd_images, sd_images, 2.0),
        'hd_sr_ssims': tf.image.ssim(hd_images, sr_images, 2.0),
    }


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    sd_images, hd_images = load_image(
        FLAGS.hd_image_path, FLAGS.scaling_factor, FLAGS.ground_truth_mode)

    with tf.Session() as session:
        model = build_model(session)

        feeds = {
            model['sd_images']: sd_images,
        }

        fetch = {
            'sr_images': model['sr_images'],
            'sr_image_png': model['sr_image_png'],
        }

        # NOTE: we can get psnr & ssim only when there is ground-truth.
        if FLAGS.ground_truth_mode:
            feeds[model['hd_images']] = hd_images

            fetch['hd_sd_psnrs'] = model['hd_sd_psnrs']
            fetch['hd_sr_psnrs'] = model['hd_sr_psnrs']
            fetch['hd_sd_ssims'] = model['hd_sd_ssims']
            fetch['hd_sr_ssims'] = model['hd_sr_ssims']

        begin_time = time.time()

        fetched = session.run(fetch, feed_dict=feeds)

        # NOTE: this is not correct since psnr & ssim are included.
        total_time = time.time() - begin_time

    with tf.gfile.GFile(FLAGS.sr_image_path, 'wb') as f:
        f.write(fetched['sr_image_png'])

    if 'hd_sd_psnrs' in fetched and 'hd_sr_psnrs' in fetched:
        print('psnr(sd, sr): {}, {}'.format(
            fetched['hd_sd_psnrs'][0], fetched['hd_sr_psnrs'][0]))

    if 'hd_sd_ssims' in fetched and 'hd_sr_ssims' in fetched:
        print('ssim(sd, sr): {}, {}'.format(
            fetched['hd_sd_ssims'][0], fetched['hd_sr_ssims'][0]))


if __name__ == '__main__':
    # NOTE: path to the meta for this experiment
    tf.app.flags.DEFINE_string('meta_path', None, 'path to the graph')

    # NOTE: path to the weights for this experiment
    tf.app.flags.DEFINE_string('ckpt_path', None, 'path to the weights')

    # NOTE: true:  build sd_image by upscale hd_image (size changed)
    #       false: build sd_image by blur hd_image (size kept)
    # NOTE: if false, there will be no psrn & ssim.
    tf.app.flags.DEFINE_boolean(
        'ground_truth_mode', True, 'true if use hd_image as ground truth')

    # NOTE: path to the image for the source of super resolving.
    tf.app.flags.DEFINE_string(
        'hd_image_path', None, 'path to the image of super resolving source')

    # NOTE: path to the image for the result of super resolving.
    tf.app.flags.DEFINE_string(
        'sr_image_path', None, 'path to the image of super resolved result')

    # NOTE: scaling factor for building the sd version image.
    tf.app.flags.DEFINE_float(
        'scaling_factor', 2.0, 'scaling factor for this experiment')

    tf.app.run()
