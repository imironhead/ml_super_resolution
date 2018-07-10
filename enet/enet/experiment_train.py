"""
"""
import os
import tensorflow as tf

import enet.datasets as datasets
import enet.model_enet as model_enet


def build_training_model(dataset_path, batch_size, vgg19_path):
    """
    """
    data_iterator = datasets.build_image_batch_iterator(
        dataset_path, 4, batch_size=batch_size)

    hd_images = data_iterator.get_next()

    sd_images = tf.image.resize_images(hd_images, [32, 32])

    model = model_enet.build_enet(sd_images, hd_images, vgg19_path)

    model['data_iterator'] = data_iterator

    return model


def build_summaries(model, batch_size):
    """
    """
    # NOTE: build loss summaries
    summary_d_loss = tf.summary.scalar('discriminator_loss', model['a_loss'])
    summary_g_loss = tf.summary.scalar('generator_loss', model['g_loss'])
    summary_p_loss = tf.summary.scalar('perceptual_loss', model['p_loss'])
    summary_t_loss = tf.summary.scalar('texture_loss', model['t_loss'])

    summary_discriminator = summary_d_loss
    summary_generator = tf.summary.merge([
        summary_g_loss, summary_p_loss, summary_t_loss])

    # NOTE: build image summaries (real v.s. fake)
    sd_images = tf.image.resize_images(model['sd_images'], [128, 128])
    sr_images = model['sr_images']
    hd_images = model['hd_images']

    images = tf.concat([sd_images, sr_images, hd_images], axis=2)

    images = tf.reshape(images, [1, batch_size * 128, 3 * 128, 3])

    images = tf.saturate_cast(images * 127.5 + 127.5, tf.uint8)

    summary_images = tf.summary.image('sd_sr_hd', images, max_outputs=4)

    return {
        'discriminator': summary_discriminator,
        'generator': summary_generator,
        'images': summary_images,
    }


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    model = build_training_model(
        FLAGS.train_dir_path, FLAGS.batch_size, FLAGS.vgg19_path)

    reporter = tf.summary.FileWriter(FLAGS.log_path)

    summaries = build_summaries(model, FLAGS.batch_size)

    source_ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    target_ckpt_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')

    with tf.Session() as session:
        saver = tf.train.Saver()

        if source_ckpt_path is None:
            session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(session, source_ckpt_path)

        # NOTE: initialize the data iterator
        session.run(model['data_iterator'].initializer)

        while True:
            step = session.run(model['step'])

            if step % 10000 == 9999:
                saver.save(session, target_ckpt_path, global_step=step)

            # NOTE: train discriminator
            fetch = {'step': model['step'], 'trainer': model['d_trainer']}

            fetch['summary_losses'] = summaries['discriminator']

            fetched = session.run(fetch)

            if 'summary_losses' in fetched:
                reporter.add_summary(fetched['summary_losses'], step)

            # NOTE: train generator
            fetch = {'step': model['step'], 'trainer': model['g_trainer']}

            fetch['summary_losses'] = summaries['generator']

            if step % 1000 == 0:
                fetch['summary_images'] = summaries['images']

            fetched = session.run(fetch)

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

    tf.app.flags.DEFINE_integer('batch_size', 64, '')

    tf.app.run()

