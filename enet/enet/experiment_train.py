"""
"""
import os
import tensorflow as tf

import enet.datasets as datasets
import enet.model_enet as model_enet


def build_training_model():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    sd_images = tf.placeholder(
        shape=[None, 32, 32, 3], dtype=tf.float32, name='sd_images')

    bq_images = tf.placeholder(
        shape=[None, 128, 128, 3], dtype=tf.float32, name='bq_images')

    hd_images = tf.placeholder(
        shape=[None, 128, 128, 3], dtype=tf.float32, name='hd_images')

    model = model_enet.build_enet(
        sd_images, bq_images, hd_images, FLAGS.model, FLAGS.vgg19_path)

    model['image_batches'] = datasets.image_batches(
        FLAGS.train_dir_path, 4.0, batch_size=FLAGS.batch_size)

    return model


def build_summaries(model):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    summaries = {}

    # NOTE: discriminator loss summaries
    if 'a_loss' in model:
        summaries['discriminator'] = \
            tf.summary.scalar('discriminator_loss', model['a_loss'])

    # NOTE: generator loss summaries
    summaries_generator = []

    if 'g_loss' in model:
        summaries_generator.append(
            tf.summary.scalar('generator_loss', model['g_loss']))

    if 'p_loss' in model:
        summaries_generator.append(
            tf.summary.scalar('perceptual_loss', model['p_loss']))

    if 't_loss' in model:
        summaries_generator.append(
            tf.summary.scalar('texture_loss', model['t_loss']))

    if len(summaries_generator) > 0:
        summaries['generator'] = tf.summary.merge(summaries_generator)

    # NOTE: build image summaries (real v.s. fake)
    sd_images = model['bq_images']
    sr_images = model['sr_images']
    hd_images = model['hd_images']

    images = tf.concat([sd_images, sr_images, hd_images], axis=2)

    images = tf.reshape(images, [1, FLAGS.batch_size * 128, 3 * 128, 3])

    images = tf.saturate_cast(images * 127.5 + 127.5, tf.uint8)

    summaries['images'] = tf.summary.image('bq_sr_hd', images, max_outputs=4)

    return summaries


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: sanity check
    if FLAGS.model not in ['p', 'pa', 'pat']:
        FLAGS.model = 'pat'

    model = build_training_model()

    reporter = tf.summary.FileWriter(FLAGS.log_path)

    summaries = build_summaries(model)

    source_ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    target_ckpt_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')

    with tf.Session() as session:
        saver = tf.train.Saver()

        if source_ckpt_path is None:
            session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(session, source_ckpt_path)

        while True:
            step = session.run(model['step'])

            if step % 1000 == 999:
                saver.save(session, target_ckpt_path, global_step=step)

            # NOTE: train discriminator
            if step % 3 == 0 and 'd_trainer' in model:
                sd_images, bq_images, hd_images = next(model['image_batches'])

                feeds = {
                    model['sd_images']: sd_images,
                    model['bq_images']: bq_images,
                    model['hd_images']: hd_images,
                }

                fetch = {
                    'step': model['step'],
                    'trainer': model['d_trainer']
                }

                if 'discriminator' in summaries:
                    fetch['summary_losses'] = summaries['discriminator']

                fetched = session.run(fetch, feed_dict=feeds)

                if 'summary_losses' in fetched:
                    reporter.add_summary(fetched['summary_losses'], step)

            # NOTE: train generator
            if 'g_trainer' in model:
                sd_images, bq_images, hd_images = next(model['image_batches'])

                feeds = {
                    model['sd_images']: sd_images,
                    model['bq_images']: bq_images,
                    model['hd_images']: hd_images,
                }

                fetch = {
                    'step': model['step'],
                    'trainer': model['g_trainer']
                }

                if 'generator' in summaries:
                    fetch['summary_losses'] = summaries['generator']

                if 'images' in summaries and step % 100 == 0:
                    fetch['summary_images'] = summaries['images']

                fetched = session.run(fetch, feed_dict=feeds)

                if 'summary_losses' in fetched:
                    reporter.add_summary(fetched['summary_losses'], step)
                if 'summary_images' in fetched:
                    reporter.add_summary(fetched['summary_images'], step)


if __name__ == '__main__':
    tf.app.flags.DEFINE_string('train_dir_path', None, '')
    tf.app.flags.DEFINE_string('valid_dir_path', None, '')
    tf.app.flags.DEFINE_string('vgg19_path', None, '')
    tf.app.flags.DEFINE_string('ckpt_path', None, '')
    tf.app.flags.DEFINE_string('log_path', None, '')
    tf.app.flags.DEFINE_string('model', 'pat', '')

    tf.app.flags.DEFINE_integer('batch_size', 64, '')

    tf.app.run()

