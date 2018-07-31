"""
"""
import os

import numpy as np
import scipy.misc
import tensorflow as tf

import enet.model_enet as model_enet


def extract_model():
    """
    to remove unnecessary noded from the trained model weights.

    then freeze the graph to reduce model memory usage so that we can super-
    resolve larger images.

    command example for freezing the graph:

    freeze_graph \
        --input_graph=./ckpt/extracted/graph.pbtxt \
        --input_checkpoint=./ckpt/extracted/model.ckpt \
        --output_graph=./ckpt/frozen/frozen.pb \
        --output_node_names=sd_images,bq_images,sr_images
    """
    FLAGS = tf.app.flags.FLAGS

    ckpt_path = FLAGS.source_ckpt_path

    # NOTE: build generator network and load weights from the checkpoint.
    sd_images = tf.placeholder(
        dtype=tf.float32, shape=[1, None, None, 3], name='sd_images')
    bq_images = tf.placeholder(
        dtype=tf.float32, shape=[1, None, None, 3], name='bq_images')

    model = model_enet.build_enet(sd_images, bq_images, None, None, None)

    with tf.Session() as session:
        # NOTE: restore model weights
        saver = tf.train.Saver()

        saver.restore(session, ckpt_path)

        graph = tf.get_default_graph()

        # NOTE: this is not necessary as long as it has been done in
        #       model_enet.build_enet
        #
        #       I forgot to do so and did not want to train the model again.
        model['sr_images'] = tf.identity(model['sr_images'], 'sr_images')

        # NOTE: save model weights again, everything but generator is removed.
        saver.save(session, os.path.join(FLAGS.target_ckpt_path, 'model.ckpt'))

        # NOTE: export the graph to freeze
        tf.train.write_graph(
            session.graph_def, FLAGS.target_ckpt_path, 'graph.pbtxt')


def source_images():
    """
    list all images that will be super-resolved
    """
    FLAGS = tf.app.flags.FLAGS

    names = tf.gfile.ListDirectory(FLAGS.source_dir_path)

    for name in names:
        name, ext = os.path.splitext(name)

        if ext.lower() not in ['.png', '.jpg', '.jpeg']:
            continue

        source_sd_path = os.path.join(FLAGS.source_dir_path, name + ext)
        target_bq_path = os.path.join(FLAGS.target_dir_path, name + '_bq.png')
        target_sr_path = os.path.join(FLAGS.target_dir_path, name + '_sr.png')

        # NOTE: read the sd image and bicubic upscale it to 4x
        sd_image = scipy.misc.imread(tf.gfile.GFile(source_sd_path, 'rb'))
        bq_image = scipy.misc.imresize(sd_image, 400, 'bicubic')

        # NOTE: re-map pixels range from -1.0 to +1.0
        sd_image = sd_image.astype(np.float32) / 127.5 - 1.0
        bq_image = bq_image.astype(np.float32) / 127.5 - 1.0

        # NOTE: expand batch dimension
        sd_image = np.expand_dims(sd_image, axis=0)
        bq_image = np.expand_dims(bq_image, axis=0)

        yield {
            'sd_image': {'image': sd_image},
            'bq_image': {'image': bq_image, 'path': target_bq_path},
            'sr_image': {'path': target_sr_path},
        }


def super_resolve():
    """
    """
    # NOTE: load the frozen graph
    FLAGS = tf.app.flags.FLAGS

    with tf.gfile.GFile(FLAGS.graph_define_path, 'rb') as gf:
        graph_def = tf.GraphDef()

        graph_def.ParseFromString(gf.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def)

    # NOTE: print tensor names if necessary
    # print([n.name for n in graph.as_graph_def().node])

    sd_images_tensor = graph.get_tensor_by_name('import/sd_images:0')
    bq_images_tensor = graph.get_tensor_by_name('import/bq_images:0')
    sr_images_tensor = graph.get_tensor_by_name('import/sr_images:0')

    # NOTE: extend to graph to encode the results as png
    bq_image_png_tensor = tf.squeeze(bq_images_tensor, [0])
    sr_image_png_tensor = tf.squeeze(sr_images_tensor, [0])

    bq_image_png_tensor = tf.saturate_cast(
        bq_image_png_tensor * 127.5 + 127.5, tf.uint8)
    sr_image_png_tensor = tf.saturate_cast(
        sr_image_png_tensor * 127.5 + 127.5, tf.uint8)

    bq_image_png_tensor = tf.image.encode_png(bq_image_png_tensor)
    sr_image_png_tensor = tf.image.encode_png(sr_image_png_tensor)

    # NOTE: do super-resolving
    with tf.Session(graph=graph) as session:
        fetch = [bq_image_png_tensor, sr_image_png_tensor]

        for images in source_images():
            feeds = {
                sd_images_tensor: images['sd_image']['image'],
                bq_images_tensor: images['bq_image']['image'],
            }

            bq_image_png, sr_image_png = session.run(fetch, feed_dict=feeds)

            # NOTE: output results
            with tf.gfile.GFile(images['sr_image']['path'], 'wb') as f:
                f.write(sr_image_png)

            with tf.gfile.GFile(images['bq_image']['path'], 'wb') as f:
                f.write(bq_image_png)


def main(_):
    """
    """
    FLAGS = tf.app.flags.FLAGS

    if FLAGS.extract_model:
        extract_model()
    else:
        super_resolve()


if __name__ == '__main__':
    # NOTE: for model extraction (remove discriminator & VGG).
    #       to freeze graph for larger input images.
    # NOTE: to do model extraction, set this falg to true
    #       refer to extract_model() for more imformation
    tf.app.flags.DEFINE_boolean('extract_model', False, '')

    # NOTE: path to the trained checkpoint
    tf.app.flags.DEFINE_string('source_ckpt_path', None, '')

    # NOTE: path for keeping the extraced graph
    tf.app.flags.DEFINE_string('target_ckpt_path', None, '')

    # NOTE: for super resolving
    # NOTE: path to the graph definition file
    tf.app.flags.DEFINE_string('graph_define_path', None, '')

    # NOTE: path to the directory that containes all images to be super-resolved
    tf.app.flags.DEFINE_string('source_dir_path', None, '')

    # NOTE: path to the directory that will be keep all bicubic up-sacaled
    #       and super-resolved results.
    tf.app.flags.DEFINE_string('target_dir_path', None, '')

    tf.app.run()

