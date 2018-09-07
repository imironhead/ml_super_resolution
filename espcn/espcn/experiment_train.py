"""
"""
import os

import tensorflow as tf

import espcn.dataset as dataset
import espcn.model_espcn as model_espcn


def build_dataset_iterator():
    """
    """
    FLAGS = tf.app.flags.FLAGS

    return dataset.build_image_batch_iterator(
        FLAGS.data_path, FLAGS.batch_size)


def build_model(dataset_iterator):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    lr_patch, hr_patch = dataset_iterator.get_next()

    model = model_espcn.build_model(lr_patch, FLAGS.scaling_factor, hr_patch)

    return model


def build_summaries(model):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    summary_loss = tf.summary.scalar('loss', model['loss'])

    # NOTE: build sr_patch + hr_patch summary
    batch_size = FLAGS.batch_size
    lr_patch_size = FLAGS.lr_patch_size
    scaling_factor = FLAGS.scaling_factor

    sr_patches = model['sr_result']
    hr_patches = model['hr_target']

    cmp_images = tf.concat([sr_patches, hr_patches], axis=2)

    shape = [
        1, batch_size * lr_patch_size * scaling_factor, scaling_factor, 3]

    sub_images = tf.split(cmp_images, 2 * lr_patch_size, axis=2)

    sub_images = [tf.reshape(img, shape) for img in sub_images]

    cmp_images = tf.concat(sub_images, axis=2)

    summary_patches = tf.summary.image('patches', cmp_images, max_outputs=1)

    return {
        'summary_loss': summary_loss,
        'summary_patches': summary_patches,
    }


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    dataset_iterator = build_dataset_iterator()

    model = build_model(dataset_iterator)

    summaries = build_summaries(model)

    source_ckpt_path = tf.train.latest_checkpoint(FLAGS.ckpt_path)
    target_ckpt_path = os.path.join(FLAGS.ckpt_path, 'model.ckpt')

    reporter = tf.summary.FileWriter(FLAGS.logs_path)

    saver = tf.train.Saver()

    with tf.Session() as session:
        if source_ckpt_path is None:
            session.run(tf.global_variables_initializer())
        else:
            tf.train.Saver().restore(session, source_ckpt_path)

        step = session.run(model['step'])

        # NOTE: exclude log which does not happend yet :)
        reporter.add_session_log(
            tf.SessionLog(status=tf.SessionLog.START), global_step=step)

        session.run(dataset_iterator.initializer)

        while step < FLAGS.stop_training_at_k_step:
            lr = FLAGS.initial_learning_rate
            lr_factor = FLAGS.learning_rate_decay_factor
            lr_level = step // FLAGS.learning_rate_decay_steps

            feeds = {
                model['learning_rate']: lr * (lr_factor ** lr_level),
            }

            fetch = {
                'step': model['step'],
                'optimizer': model['optimizer'],
                'summary_loss': summaries['summary_loss'],
            }

            if step % 1000 == 0:
                fetch['summary_patches'] = summaries['summary_patches']

            fetched = session.run(fetch, feed_dict=feeds)

            step = fetched['step']

            if 'summary_loss' in fetched:
                reporter.add_summary(fetched['summary_loss'], step)

            if 'summary_patches' in fetched:
                reporter.add_summary(fetched['summary_patches'], step)

        reporter.flush()

        saver.save(session, target_ckpt_path, global_step=model['step'])


if __name__ == '__main__':
    tf.app.flags.DEFINE_string(
        'data_path', None, 'path to training data (tfrecord) directory')

    tf.app.flags.DEFINE_string(
        'ckpt_path', None, 'path to a directory for keeping the checkpoint')

    tf.app.flags.DEFINE_string(
        'logs_path', None, 'path to a directory for keeping log')

    tf.app.flags.DEFINE_integer(
        'batch_size', 64, 'size of each batch during training')

    tf.app.flags.DEFINE_integer(
        'scaling_factor', 3, 'scaling factor for training')

    # NOTE: to reconstruct the super resolved version of image from sub-pixel
    #       convolution result (to build summaries in this experiment), we need
    #       the size of the training patches (to split patches along horizontal
    #       axes).
    tf.app.flags.DEFINE_integer(
        'lr_patch_size', 17, 'size of lr_patch as training data')

    tf.app.flags.DEFINE_float('initial_learning_rate', 0.1, '')

    tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.1, '')

    tf.app.flags.DEFINE_integer('learning_rate_decay_steps', 2560, '')

    tf.app.flags.DEFINE_integer(
        'stop_training_at_k_step', 10000, 'stop training at k step')

    tf.app.run()

