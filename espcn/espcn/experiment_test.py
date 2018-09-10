"""
"""
import os

import numpy as np
import skimage.filters
import skimage.io
import tensorflow as tf

import espcn.dataset as dataset
import espcn.model_espcn as model_espcn


def build_model():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    meta_path = FLAGS.ckpt_path + '.meta'
    ckpt_path = FLAGS.ckpt_path

    model = model_espcn.build_test_model(meta_path, ckpt_path)

    hr_targets = tf.placeholder(
        shape=[None, None, None, 3 * (model['scaling_factor'] ** 2)],
        dtype=tf.float32)

    model['hr_targets'] = hr_targets

    # NOTE: build evaluation metrics
    #       remap to [0.0, 1.0] for potential color space conversion
    sr_results = model['sr_results'] * 0.5 + 0.5
    hr_targets = model['hr_targets'] * 0.5 + 0.5

    sr_results = tf.clip_by_value(sr_results, 0.0, 1.0)
    hr_targets = tf.clip_by_value(hr_targets, 0.0, 1.0)

    if FLAGS.score_space == 'y':
        shape = tf.shape(hr_targets)

        h, w = shape[1], shape[2]

        w *= model['scaling_factor'] ** 2

        sr_results = tf.reshape(sr_results, [-1, h, w, 3])
        hr_targets = tf.reshape(hr_targets, [-1, h, w, 3])

        sr_results = tf.image.rgb_to_yuv(sr_results)
        hr_targets = tf.image.rgb_to_yuv(hr_targets)

        sr_results = sr_results[:, :, :, :1]
        hr_targets = hr_targets[:, :, :, :1]

    model['psnrs'] = tf.image.psnr(hr_targets, sr_results, 1.0)
    model['ssims'] = tf.image.ssim(hr_targets, sr_results, 1.0)

    return model


def prepare_image_pair(hr_image_path, upscaling_factor):
    """
    """
    # NOTE: read image, images may come from google cloud storage
    with tf.gfile.GFile(hr_image_path, 'rb') as hr_image_file:
        hr_image = skimage.io.imread(hr_image_file)

    # NOTE: trim the size for we want psnr & ssim (need same size images)
    h, w, c = hr_image.shape

    h -= h % upscaling_factor
    w -= w % upscaling_factor

    hr_image = hr_image[:h, :w]

    # NOTE: adjust pixel range from [0.0, 1.0] to [-1.0, 1.0]
    hr_image = hr_image / 127.5 - 1.0

    # NOTE: make gaussian blur version
    sigma = np.maximum(0.0, 0.5 * (upscaling_factor - 1.0))

    # NOTE: the input image is converted according to the conventions of
    #       img_as_float.
    bl_image = skimage.filters.gaussian(hr_image, sigma, mode='nearest')

    offset = upscaling_factor // 2

    lr_image = bl_image[offset::upscaling_factor, offset::upscaling_factor]

    # NOTE: reshape hr_image so we do not have tp construct super-resolved
    #       images to evaluate psnr/ssim
    hr_patches = np.split(hr_image, w // upscaling_factor, axis=1)

    hr_patches = \
        [np.reshape(im, [h // upscaling_factor, 1, -1]) for im in hr_patches]

    hr_image = np.concatenate(hr_patches, axis=1)

    return lr_image, hr_image


def evaluate_images():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    model = build_model()

    # NOTE: enum file names
    names = tf.gfile.ListDirectory(FLAGS.data_path)

    # NOTE: drop non-image extensions
    names = [n for n in names if n[-4:] in ['.png', '.jpg', '.bmp']]

    # NOTE: finalize image paths
    image_paths = [os.path.join(FLAGS.data_path, n) for n in names]

    psnrs = []
    ssims = []

    with tf.Session() as session:
        for image_path in image_paths:
            lr_image, hr_image = \
                prepare_image_pair(image_path, model['scaling_factor'])

            feeds = {
                model['lr_sources']: np.expand_dims(lr_image, 0),
                model['hr_targets']: np.expand_dims(hr_image, 0),
            }

            fetch = {
                'psnrs': model['psnrs'],
                'ssims': model['ssims'],
            }

            fetched = session.run(fetch, feed_dict=feeds)

            psnrs.append(fetched['psnrs'][0])
            ssims.append(fetched['ssims'][0])

            print('name: {:>32}, psnr: {:.4f}, ssim: {:.4f}'.format(
                os.path.basename(image_path), psnrs[-1], ssims[-1]))

    print('data: {}'.format(FLAGS.data_path))
    print('psnr: {0:.4f}'.format(np.mean(psnrs)))
    print('ssim: {0:.4f}'.format(np.mean(ssims)))


def super_resolve_image():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    model = build_model()

    # NOTE: read image, images may come from google cloud storage
    with tf.gfile.GFile(FLAGS.data_path, 'rb') as lr_image_file:
        lr_image = skimage.io.imread(lr_image_file)

    # NOTE: adjust pixel range from [0.0, 1.0] to [-1.0, 1.0]
    lr_image = lr_image / 127.5 - 1.0

    lrh, lrw, _ = lr_image.shape

    with tf.Session() as session:
        feeds = {
            model['lr_sources']: np.expand_dims(lr_image, 0),
        }

        sr_results = session.run(model['sr_results'], feed_dict=feeds)

    sf = model['scaling_factor']

    sr_patches = np.split(sr_results[0], lrw, axis=1)

    sr_patches = [np.reshape(p , [lrh * sf, sf, 3]) for p in sr_patches]

    sr_result = np.concatenate(sr_patches, axis=1)

    sr_result = sr_result * 0.5 + 0.5

    with tf.gfile.GFile(FLAGS.result_path, 'wb') as sr_image_file:
        skimage.io.imsave(sr_image_file, sr_result)


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if tf.gfile.IsDirectory(FLAGS.data_path):
        evaluate_images()
    else:
        super_resolve_image()


if __name__ == '__main__':
    tf.app.flags.DEFINE_string(
        'data_path', None, 'path to the test data directory')

    tf.app.flags.DEFINE_string(
        'ckpt_path', None, 'path to the checkpoint')

    tf.app.flags.DEFINE_string(
        'result_path', None, 'path for the super-resolved image')

    tf.app.flags.DEFINE_string(
        'score_space', 'y', 'evaluate on y(uv) or rgb')

    tf.app.run()

