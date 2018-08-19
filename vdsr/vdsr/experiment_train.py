"""
"""
import os

import tensorflow as tf

import vdsr.dataset as dataset
import vdsr.model_vdsr as model_vdsr


def build_model():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: placeholder as source sd images
    sd_images = tf.placeholder(
        shape=[None, None, None, 3],
        dtype=tf.float32,
        name='sd_images')

    # NOTE: placeholder as target hd images (ground-truth)
    hd_images = tf.placeholder(
        shape=[None, None, None, 3],
        dtype=tf.float32,
        name='hd_images')

    model = model_vdsr.build_model(
        sd_images,
        hd_images,
        num_layers=FLAGS.num_layers,
        use_adam=FLAGS.use_adam)

    return model


def build_dataset():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: transform scaling_factors from a string to a list of floats
    if isinstance(FLAGS.scaling_factors, str):
        FLAGS.scaling_factors = \
            [float(x) for x in FLAGS.scaling_factors.split('_')]

    image_batches = dataset.image_batches(
        FLAGS.data_path,
        FLAGS.scaling_factors,
        FLAGS.image_size,
        FLAGS.batch_size)

    return image_batches


def build_summaries(model):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    # NOTE: loss summaries
    summary_step = tf.summary.scalar('loss', model['loss'])

    # NOTE: build image summary
    sd_images = model['sd_images']
    sr_images = model['sr_images']
    hd_images = model['hd_images']

    images = tf.concat([sd_images, sr_images, hd_images], axis=2)

    images = tf.reshape(
        images,
        [1, FLAGS.batch_size * FLAGS.image_size, 3 * FLAGS.image_size, 3])

    images = tf.saturate_cast(images * 127.5 + 127.5, tf.uint8)

    summary_image = tf.summary.image('sd_sr_hd', images, max_outputs=4)

    # NOTE: build psnr summary
    psnrs = tf.image.psnr(sr_images, hd_images, max_val=2.0)

    psnrs = tf.reduce_mean(psnrs)

    summary_psnr = tf.summary.scalar('psnr', psnrs)

    summary_epoch = tf.summary.merge(
        [summary_step, summary_image, summary_psnr])

    return {
        'step': summary_step,
        'epoch': summary_epoch,
    }


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    image_batches = build_dataset()

    model = build_model()

    summaries = build_summaries(model)

    reporter = tf.summary.FileWriter(FLAGS.logs_path)

    source_ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    target_ckpt_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')

    learning_rate = FLAGS.initial_learning_rate
    decay_steps = FLAGS.learning_rate_decay_steps
    decay_factor = FLAGS.learning_rate_decay_factor

    with tf.Session() as session:
        saver = tf.train.Saver()

        if source_ckpt_path is None:
            session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(session, source_ckpt_path)

        while True:
            step = session.run(model['step'])

            if step == FLAGS.stop_training_at_k_step:
                saver.save(session, target_ckpt_path, global_step=step)
                break

            lr = learning_rate * (decay_factor ** (step // decay_steps))

            sd_images, hd_images = next(image_batches)

            feeds = {
                model['sd_images']: sd_images,
                model['hd_images']: hd_images,
                model['learning_rate']: lr,
            }

            fetch = {
                'step': model['step'],
                'loss': model['loss'],
                'trainer': model['trainer'],
            }

            if (step + 1) % 100 == 0:
                fetch['summary'] = summaries['epoch']
            else:
                fetch['summary'] = summaries['step']

            fetched = session.run(fetch, feed_dict=feeds)

            reporter.add_summary(fetched['summary'], fetched['step'])


if __name__ == '__main__':
    tf.app.flags.DEFINE_string(
        'data_path',
        None,
        'path to a directory which contains all image training data')

    tf.app.flags.DEFINE_string(
        'ckpt_path', None, 'path to a directory for keeping the checkpoint')

    tf.app.flags.DEFINE_string(
        'logs_path', None, 'path to a directory for keeping log')

    # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
    #       deep convolutional networks
    #
    #       2, 3 or 4
    tf.app.flags.DEFINE_string(
        'scaling_factors',
        '2_3_4',
        'different scaling factors for training, separated by _')

    # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
    #       deep convolutional networks
    #
    #       2.1 convolutional network for image super-resolution
    #       our network is very deep (20 vs. 3) and information used for
    #       reconstruction (receptive field) is much larger (41x41 vs. 13x13).
    tf.app.flags.DEFINE_integer('image_size', 41, 'size of training images')

    # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
    #       deep convolutional networks
    #
    #       5.2 training parameters
    #       training uses batches of size 64.
    tf.app.flags.DEFINE_integer(
        'batch_size', 64, 'size of each batch during training')

    # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
    #       deep convolutional networks
    #
    #       5.2 training parameters
    #       we use a network of depth 20.
    tf.app.flags.DEFINE_integer('num_layers', 20, 'number of hidden layers')

    # NOTE: arXiv:1511.04587v2, accurate image super-resolution using very
    #       deep convolutional networks
    #
    #       we train all experiments over 80 epochs (9960 iterations with batch
    #       size 64). learning rate was initially set to 0.1 and then decreased
    #       by a factor of 10 every 20 epochs.
    tf.app.flags.DEFINE_integer('learning_rate_decay_steps', 2560, '')

    tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1, '')

    tf.app.flags.DEFINE_float('initial_learning_rate', 0.1, '')

    tf.app.flags.DEFINE_integer(
        'stop_training_at_k_step',
        12800,
        'stop training at k step, default is stop after 80 epochs')

    # NOTE: use newer optimizer
    tf.app.flags.DEFINE_boolean(
        'use_adam', True, 'use adam instead of momentum optimizer')

    tf.app.run()
