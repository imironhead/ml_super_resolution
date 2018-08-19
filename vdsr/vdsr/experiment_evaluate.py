"""
"""
import os
import time

import numpy as np
import skimage.io
import skimage.util
import tensorflow as tf

import vdsr.dataset as dataset


def load_image(hd_image_path, scaling_factor):
    """
    """
    # NOTE: read image, images may come from google cloud storage
    with tf.gfile.GFile(hd_image_path, 'rb') as hd_image_file:
        hd_image = skimage.io.imread(hd_image_file)

    # NOTE: cast uint8 to float32
    hd_image = skimage.util.img_as_float32(hd_image)

    sd_image = dataset.hd_image_to_sd_image(hd_image, scaling_factor)

    # NOTE: from [0.0, 1.0] to [-1.0, +1.0]
    sd_image = sd_image * 2.0 - 1.0
    hd_image = hd_image * 2.0 - 1.0

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

    return {
        'sd_images': sd_images,
        'sr_images': sr_images,
        'hd_images': hd_images,
        'hd_sd_psnrs': tf.image.psnr(hd_images, sd_images, 2.0),
        'hd_sr_psnrs': tf.image.psnr(hd_images, sr_images, 2.0),
        'hd_sd_ssims': tf.image.ssim(hd_images, sd_images, 2.0),
        'hd_sr_ssims': tf.image.ssim(hd_images, sr_images, 2.0),
    }


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: enum file names
    names = tf.gfile.ListDirectory(FLAGS.hd_image_dir_path)

    # NOTE: drop non-image extensions
    names = [n for n in names if n[-4:] in ['.png', '.jpg', '.bmp']]

    # NOTE: finalize image paths
    image_paths = [os.path.join(FLAGS.hd_image_dir_path, n) for n in names]

    hd_sd_psnrs = []
    hd_sr_psnrs = []
    hd_sd_ssims = []
    hd_sr_ssims = []

    total_time = 0.0

    with tf.Session() as session:
        model = build_model(session)

        # NOTE: do it one by one for we do not know the image size
        for image_path in image_paths:
            sd_images, hd_images = load_image(image_path, FLAGS.scaling_factor)

            feeds = {
                model['sd_images']: sd_images,
                model['hd_images']: hd_images,
            }

            fetch = {
                'hd_sd_psnrs': model['hd_sd_psnrs'],
                'hd_sr_psnrs': model['hd_sr_psnrs'],
                'hd_sd_ssims': model['hd_sd_ssims'],
                'hd_sr_ssims': model['hd_sr_ssims'],
            }

            begin_time = time.time()

            fetched = session.run(fetch, feed_dict=feeds)

            total_time += time.time() - begin_time

            hd_sd_psnrs.append(fetched['hd_sd_psnrs'][0])
            hd_sr_psnrs.append(fetched['hd_sr_psnrs'][0])
            hd_sd_ssims.append(fetched['hd_sd_ssims'][0])
            hd_sr_ssims.append(fetched['hd_sr_ssims'][0])

    hd_sd_psnr = np.mean(hd_sd_psnrs)
    hd_sr_psnr = np.mean(hd_sr_psnrs)
    hd_sd_ssim = np.mean(hd_sd_ssims)
    hd_sr_ssim = np.mean(hd_sr_ssims)

    print('x{}'.format(FLAGS.scaling_factor))
    print('time (s)     : {}'.format(total_time / len(image_paths)))
    print('psnr (sd, sr): {}, {}'.format(hd_sd_psnr, hd_sr_psnr))
    print('ssim (sd, sr): {}, {}'.format(hd_sd_ssim, hd_sr_ssim))


if __name__ == '__main__':
    # NOTE: path to the meta for this experiment
    tf.app.flags.DEFINE_string('meta_path', None, 'path to the graph')

    # NOTE: path to the weights for this experiment
    tf.app.flags.DEFINE_string('ckpt_path', None, 'path to the weights')

    # NOTE: path to a dir contains hd images. we will build their sd versions
    #       and super resolve them to collect psnr & ssim.
    tf.app.flags.DEFINE_string(
        'hd_image_dir_path', None, 'path to a dir contains hd images')

    # NOTE: scaling factor for building the sd version image.
    tf.app.flags.DEFINE_integer(
        'scaling_factor', 2, 'scaling factor for this experiment')

    tf.app.run()

