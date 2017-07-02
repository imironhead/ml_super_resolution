"""
"""
import glob
import os
import tensorflow as tf


tf.app.flags.DEFINE_string('ckpt-dir-path', './ckpts/', '')
tf.app.flags.DEFINE_string('logs-dir-path', './logs/', '')
tf.app.flags.DEFINE_string('training-images-path', None, '')
tf.app.flags.DEFINE_string('sr-source-path', None, '')
tf.app.flags.DEFINE_string('sr-target-path', None, '')
tf.app.flags.DEFINE_boolean('train', False, '')
tf.app.flags.DEFINE_integer('batch-size', 64, '')
tf.app.flags.DEFINE_integer('upscaling-factor', 3, '')
tf.app.flags.DEFINE_integer('crop-image-size', 256, '')
tf.app.flags.DEFINE_integer('crop-image-side', 6, '')
tf.app.flags.DEFINE_integer('srcnn-fsub', 33, '')
tf.app.flags.DEFINE_integer('srcnn-f1', 9, '')
tf.app.flags.DEFINE_integer('srcnn-f2', 1, '')
tf.app.flags.DEFINE_integer('srcnn-f3', 5, '')
tf.app.flags.DEFINE_integer('srcnn-n1', 64, '')
tf.app.flags.DEFINE_integer('srcnn-n2', 32, '')

FLAGS = tf.app.flags.FLAGS


def sanity_check():
    """
    """
    smaller_output_size = \
        FLAGS.srcnn_fsub - FLAGS.srcnn_f1 - FLAGS.srcnn_f2 - FLAGS.srcnn_f3 + 3

    boundary = (FLAGS.srcnn_fsub - smaller_output_size) / 2

    crop_size = FLAGS.crop_image_size
    crop_size = ((crop_size - boundary * 2) / smaller_output_size)

    FLAGS.crop_image_side = boundary
    FLAGS.crop_image_size = crop_size * smaller_output_size + boundary * 2


def build_dataset_reader():
    """
    """
    if FLAGS.train:
        paths_jpg_wildcards = os.path.join(FLAGS.training_images_path, '*.jpg')

        paths_images = glob.glob(paths_jpg_wildcards)
    else:
        paths_images = [FLAGS.sr_source_path]

    file_name_queue = tf.train.string_input_producer(paths_images)

    reader = tf.WholeFileReader()

    reader_key, reader_val = reader.read(file_name_queue)

    image = tf.image.decode_jpeg(reader_val, channels=3)

    image = tf.random_crop(image, size=[FLAGS.crop_image_size] * 2 + [3])

    image = tf.image.random_flip_left_right(image)

    image = tf.cast(image, dtype=tf.float32) / 127.5 - 1.0

    return tf.train.batch(
        tensors=[image],
        batch_size=FLAGS.batch_size,
        capacity=FLAGS.batch_size)


def build_srcnn():
    """
    """
    weights_initializer = tf.truncated_normal_initializer(stddev=0.001)

    hi_images = build_dataset_reader()

    # scale to lo res
    lo_images = tf.image.resize_bicubic(
        hi_images, [FLAGS.crop_image_size / FLAGS.upscaling_factor] * 2)

    lo_images = tf.image.resize_bicubic(
        lo_images, [FLAGS.crop_image_size] * 2)

    # arXiv:1501.00092v3,
    # to avoid border effects during training, all convolutional layers have no
    # padding.

    # patch extraction and representation
    flow = tf.contrib.layers.convolution2d(
        inputs=lo_images,
        num_outputs=FLAGS.srcnn_n1,
        kernel_size=FLAGS.srcnn_f1,
        stride=1,
        padding='VALID',
        activation_fn=tf.nn.relu,
        weights_initializer=weights_initializer,
        scope='patch_extraction')

    # non-linear mapping
    flow = tf.contrib.layers.convolution2d(
        inputs=flow,
        num_outputs=FLAGS.srcnn_n2,
        kernel_size=FLAGS.srcnn_f2,
        stride=1,
        padding='VALID',
        activation_fn=tf.nn.relu,
        weights_initializer=weights_initializer,
        scope='non_linear_mapping')

    # reconstruction
    sr_images = tf.contrib.layers.convolution2d(
        inputs=flow,
        num_outputs=3,
        kernel_size=FLAGS.srcnn_f3,
        stride=1,
        padding='VALID',
        activation_fn=tf.nn.tanh,
        weights_initializer=weights_initializer,
        scope='reconstruction')

    bb_side = FLAGS.crop_image_side
    bb_size = FLAGS.crop_image_size - 2 * bb_side

    hi_images = tf.image.crop_to_bounding_box(
        hi_images, bb_side, bb_side, bb_size, bb_size)

    lo_images = tf.image.crop_to_bounding_box(
        lo_images, bb_side, bb_side, bb_size, bb_size)

    # mean squared error
    loss = tf.reshape(sr_images - hi_images, [-1, bb_size ** 2])
    loss = tf.norm(loss, 2, axis=1)
    loss = tf.reduce_mean(loss)

    # global step
    step = tf.get_variable(
        'global_step',
        [],
        trainable=False,
        initializer=tf.constant_initializer(0, dtype=tf.int64),
        dtype=tf.int64)

    #
    trainer = tf.train.AdamOptimizer(
        learning_rate=0.001, beta1=0.5, beta2=0.9)
    trainer = trainer.minimize(loss, global_step=step)

    return {
        'step': step,
        'loss': loss,
        'trainer': trainer,
        'hd_images': hi_images,
        'sd_images': lo_images,
        'sr_images': sr_images,
    }


def build_summaries(srcnn):
    """
    """
    batch = FLAGS.batch_size
    width = FLAGS.crop_image_size - 2 * FLAGS.crop_image_side

    hd_images = tf.image.pad_to_bounding_box(
        srcnn['hd_images'], 0, 0, width, width)
    sr_images = tf.image.pad_to_bounding_box(
        srcnn['sr_images'], 0, 0, width, width)

    hd_image = tf.reshape(hd_images, [1, batch * width, width, 3])
    sr_image = tf.reshape(sr_images, [1, batch * width, width, 3])
    sd_image = tf.reshape(srcnn['sd_images'], [1, batch * width, width, 3])

    image = tf.concat([hd_image, sd_image, sr_image], axis=2)

    summary_image = tf.summary.image('image', image)

    summary_part = tf.summary.scalar('loss', srcnn['loss'])
    summary_plus = tf.summary.merge([summary_part, summary_image])

    return {
        'summary_part': summary_part,
        'summary_plus': summary_plus,
    }


def train():
    """
    """
    # tensorflow
    ckpt_source_path = tf.train.latest_checkpoint(FLAGS.ckpt_dir_path)
    ckpt_target_path = os.path.join(FLAGS.ckpt_dir_path, 'model.ckpt')

    srcnn = build_srcnn()
    summaries = build_summaries(srcnn)

    reporter = tf.summary.FileWriter(FLAGS.logs_dir_path)

    with tf.Session() as session:
        if ckpt_source_path is None:
            session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(session, ckpt_source_path)

        # give up overlapped old data
        step = session.run(srcnn['step'])

        reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START), global_step=step)

        # make dataset reader work
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        while True:
            # discriminator
            fetches = {
                'loss': srcnn['loss'],
                'step': srcnn['step'],
                'trainer': srcnn['trainer'],
            }

            if step % 500 == 0:
                fetches['summary'] = summaries['summary_plus']
            else:
                fetches['summary'] = summaries['summary_part']

            fetched = session.run(fetches)

            step = fetched['step']

            reporter.add_summary(fetched['summary'], step)

            if step % 100 == 0:
                print('loss[{}]: {}'.format(step, fetched['loss']))

            if step % 5000 == 0:
                tf.train.Saver().save(
                    session,
                    ckpt_target_path,
                    global_step=srcnn['step'])

        coord.request_stop()
        coord.join(threads)


def super_resolution():
    """
    """


def main(_):
    """
    """
    sanity_check()

    if FLAGS.train:
        train()
    else:
        super_resolution()


if __name__ == '__main__':
    tf.app.run()
