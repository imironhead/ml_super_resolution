"""
"""
import os
import tensorflow as tf

import enet.datasets as datasets
import enet.model_enet as model_enet


def build_resolving_model(source_image_path):
    """
    """
    sd_image = tf.read_file(source_image_path)

    sd_image = tf.image.decode_image(sd_image, channels=3)

    # NOTE: to float32 images
    sd_image = tf.image.convert_image_dtype(x, tf.float32))

    sd_image = sd_image * 2.0 - 1.0

    # NOTE: expand batch dimension
    sd_images = tf.expand_dims(sd_image, 0)

    model = model_enet.build_enet(sd_images, None, None)

    sr_image = tf.squeeze(model['sr_images'], [0])

    sr_image = tf.saturate_cast(sr_image * 127.5 + 127.5, tf.uint8)

    sr_image = tf.image.encode_png(sr_image)

    model['sr_image_png'] = sr_image

    return model


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    model = build_resolving_model(FLAGS.source_image_path)

    source_ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)

    with tf.Session() as session:
        saver = tf.train.Saver()

        tf.train.Saver().restore(session, source_ckpt_path)

        # NOTE: initialize the data iterator
        sr_image_png = session.run(model['sr_image_png'])

    with tf.gfile.GFile(FLAGS.target_image_path, 'wb') as f:
        f.write(sr_image_png)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('source_image_path', None, '')
    tf.app.flags.DEFINE_string('target_image_path', None, '')
    tf.app.flags.DEFINE_string('ckpt_path', None, '')

    tf.app.run()

