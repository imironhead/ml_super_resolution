"""
"""
import os
import tensorflow as tf


def build_path_generator(dir_path):
    """
    """
    # NOTE: check if the extension is of an image
    def is_image_name(name):
        name, ext = os.path.splitext(name)

        return ext.lower() in ['.png', '.jpg', '.jpeg']

    names = os.listdir(dir_path)
    names = [n for n in names if is_image_name(n)]

    def paths_generator():
        while True:
            # NOTE: shuffle names
            np.random.shuffle(names)

            for name in names:
                yield os.path.join(dir_path, name)

    return paths_generator


def build_image_batch_iterator(dir_path, scale_factor, batch_size=32):
    """
    """
    # NOTE: build path list dataset, the '*' is must for google cloud storage
    path_pattern = os.path.join(dir_path, '*')

    data = tf.data.Dataset.list_files(path_pattern, shuffle=True)

    # NOTE: the path generator never ends
    data = data.repeat()

    # NOTE: the path generator shuffled path list in each epoch

    # NOTE: read file, transform data from file path to byte data
    data = data.map(tf.read_file)

    # NOTE: decode image, transform data from byte data to pixels
    data = data.map(lambda x: tf.image.decode_image(x, channels=3))

    # NOTE: to float32 images
    data = data.map(lambda x: tf.image.convert_image_dtype(x, tf.float32))

    # NOTE: to -1.0 ~ +1.0
    data = data.map(lambda x: x * 2.0 - 1.0)

    # NOTE: random crop to (32*alpha)x(32*alpha)x3
    # NOTE: arXiv: 1612.07919v2, 5.3
    #       we downsample the 256x256 images by alpha and then crop these to
    #       patches of size 32x32
    size = 32 * scale_factor

    data = data.map(lambda x: tf.random_crop(x, size=[size, size, 3]))

    # NOTE: random flip
    data = data.map(tf.image.random_flip_left_right)
    data = data.map(tf.image.random_flip_up_down)

    # NOTE: combine images to batch
    data = data.batch(batch_size=batch_size)

    # NOTE: create the final iterator
    iterator = data.make_initializable_iterator()

    return iterator

