"""
Define generators that read small image batches.

MUST: No resizing within the dataset generator (should be preprocessed).
"""
import functools
import itertools
import os

import imageio
import numpy as np
import tensorflow as tf


def sanity_check(image_streams):
    """
    Find the smallest desired patch size. Also see if we have to force random
    cropping on even position. And see if the sizes sample images match their
    desired patch sizes.

    Arguments:
        subsets: Information of image subsets in a list. For example,
            [{'path': dir_path, 'size': patch_size}, ...]

    Return:
        Tuple of (ref_patch_size, even_anchor). ref_patch_size represents the
        smalles patch size in subsets. even_anchor represents if we have to
        crop small patches on even coordinates.
    """
    if not image_streams:
        raise ValueError('Empty image_streams')

    for index, info in enumerate(image_streams):
        if not isinstance(info.get('size', None), int):
            raise ValueError(f'Invalid size for stream {index}')

        if not isinstance(info.get('path', None), str):
            raise ValueError(f'Invalid path for stream {index}')

        if not os.path.isdir(info['path']):
            raise ValueError(f'Invalid path for stream {index}')

    # NOTE: Select one image for sanity check.
    path = [info['path'] for info in image_streams][0]
    name = os.listdir(path)[0]

    all_sizes = []

    ref_image_height = None
    ref_image_width = None
    ref_patch_size = None

    for info in image_streams:
        path = os.path.join(info['path'], name)

        image = imageio.imread(path)

        all_sizes.append((image.shape[0], image.shape[1], info['size']))

        if ref_patch_size is None or ref_patch_size > info['size']:
            ref_patch_size = info['size']
            ref_image_width = image.shape[1]
            ref_image_height = image.shape[0]

    even_anchor = False

    for image_h, image_w, patch_size in all_sizes:
        # NOTE: For this image, we can do it by force the cropping position
        #       adjusted to even integers.
        if (patch_size * 2) % ref_patch_size == 0:
            even_anchor = True

        if image_h * ref_patch_size != ref_image_height * patch_size:
            raise ValueError('Invalid image height v.s. patch size.')

        if image_w * ref_patch_size != ref_image_width * patch_size:
            raise ValueError('Invalid image width v.s. patch size.')

        if patch_size % ref_patch_size != 0 and \
                (patch_size * 2) % ref_patch_size != 0:
            raise ValueError(
                'patch_size / reference_patch_size must be N or N + 0.5')

    return ref_patch_size, even_anchor


def infinite_image_streams(subsets):
    """
    Yield image information from subsets infinitely and randomly.

    Arguments:
        subsets: Information of image subsets in a list. For example,
            [{'path': dir_path, 'size': patch_size}, ...]

    Yield:
        Associated image information from one or multiple source in a list. For
        example, ((path_0, size_0), (path_1, size_1), ...)
    """
    # NOTE: Collect names.
    paths = [subset['path'] for subset in subsets]
    names = os.listdir(paths[0])

    # NOTE: Accept only jpg or png
    names = [name for name in names if name.lower()[-4:] in ['.jpg', '.png']]

    # NOTE: All names must exist within all streams' directory.
    for subset, name in itertools.product(subsets, names):
        path = os.path.join(subset['path'], name)

        if not os.path.isfile(path):
            raise ValueError('All names within each dir must be matched.')

    # NOTE: Infinite generator.
    while True:
        name = np.random.choice(names)

        new_images = []

        for subset in subsets:
            new_images.append((
                os.path.join(subset['path'], name), subset['size']))

        yield tuple(new_images)


def read_images(*image_descriptors):
    """
    Read images from one or multiple source.

    Arguments:
        image_descriptors: Each descriptor is a tuple, (image_path,
            patch_size).

    Return
        The same descriptors, but paths are replaced with associated image
        tensors.
    """
    new_descriptors = []

    for path, size in image_descriptors:
        image = tf.io.read_file(path)
        image = tf.io.decode_image(image, channels=3)

        new_descriptors.append((image, size))

    return tuple(new_descriptors)


def sample_image_patches(
        reference_patch_size,
        even_anchor,
        *image_descriptors):
    """
    Sample patches from images.

    Arguments:
        reference_patch_size: The smallest desired patch size.
        even_anchor: If ture, force the cropping offset aligned to even
            positions.

    Return
        Cropped patches.
    """
    image, patch_size = image_descriptors[0]

    # NOTE: Rebuild the reference image size.
    image_shape = tf.shape(image, out_type=tf.int64)

    image_h, image_w = image_shape[0], image_shape[1]

    reference_image_h = image_h * reference_patch_size // patch_size
    reference_image_w = image_w * reference_patch_size // patch_size

    # NOTE: Select the reference cropping area randomly
    reference_offset_y = tf.random.uniform(
        [],
        maxval=reference_image_h - reference_patch_size,
        dtype=tf.dtypes.int64)
    reference_offset_x = tf.random.uniform(
        [],
        maxval=reference_image_w - reference_patch_size,
        dtype=tf.dtypes.int64)

    # NOTE: Adjust the selected area for non integer scaling factor.
    #       target_patch_size / source_patch_size = 1.5, 2.5, 3.5, etc.
    #
    #       For example, assume target_patch_size is 15 and source_patch_size
    #       is 10, then the scaling factor is 1.5. If source_offset_x is 3
    #       (odd), then target_offset_x must be 4.5 which is impossible. If
    #       source_offset_x is 4(even), then the target_offset_x is 6 which
    #       is fine. So in this case, we will force the source_offset to be
    #       even.
    if even_anchor:
        reference_offset_y -= reference_offset_y % 2
        reference_offset_x -= reference_offset_x % 2

    patches = []

    for image, size in image_descriptors:
        target_offset_y = size * reference_offset_y // reference_patch_size
        target_offset_x = size * reference_offset_x // reference_patch_size

        patch = image[
            target_offset_y : target_offset_y + size,
            target_offset_x : target_offset_x + size]

        patches.append(patch)

    return tuple(patches)


def random_augment(*patches):
    """
    Random augment all pathces. All associated patches must be augmented in the
    same way. Currently support 3 types of augmentation: Vertical flipping,
    horizontal flipping and transpose.

    Arguments:
        patches: Tensors of image patches from one or multiple source.

    Return:
        Randomly augmented image patches.
    """
    augmented_patches = patches

    # NOTE: Augmentation, flip vertically
    if tf.random.uniform([], maxval=1, dtype=tf.dtypes.int32) == 0:
        augmented_patches = [patch[::-1, ::] for patch in augmented_patches]

    # NOTE: augmentation, flip horizontally
    if tf.random.uniform([], maxval=1, dtype=tf.dtypes.int32) == 0:
        augmented_patches = [patch[::, ::-1] for patch in augmented_patches]

    # NOTE: augmentation, transpose x-y axis
    if tf.random.uniform([], maxval=1, dtype=tf.dtypes.int32) == 0:
        augmented_patches = \
            [tf.transpose(patch, [1, 0, 2]) for patch in augmented_patches]

    return tuple(augmented_patches)


def cast_pixels(*patches):
    """
    Cast all patches to float point values and map to [-1.0, +1.0].

    Arguments:
        patches: Tensors of image patches from one or multiple source.

    Return:
        A new Tensorflow dataset with all image pixels casted to floating point
        between -1.0 to +1.0f.
    """
    patches = [tf.cast(patch, tf.float32) / 127.5 - 1.0 for patch in patches]

    return tuple(patches)


def finalize_batches(subsets, *batches):
    """
    Reshape each patch batch explicitly so that the graph can initialize layers
    which need concreate shape (e.g. dense layer need input shape to build its
    weights).

    Arguments:
        subsets: List of image information. Each element has information of
            patch size ('size').
        batches: A Tensorflow dataset consists of image tensorw defined by
            subsets.

    Return:
        A Tensorflow dataset and all its subset images have explicit shapes.
    """
    new_batches = []

    for index, subset in enumerate(subsets):
        new_shape = [-1, subset['size'], subset['size'], 3]
        new_batch = tf.reshape(batches[index], new_shape)
        new_batches.append(new_batch)

    return new_batches


def build_dataset(
        subsets,
        batch_size,
        sample_rate,
        augment,
        **_):
    """
    Build a tensorflow dataset which yields image batches. The source should be
    images from different directories, each of them is connected with file
    names. For example,
        ./image_1x/
            000.png
            001.png
        ./image_2x/
            000.png
            001.png
    Take super resolution as example, we expected the yielded patches from
    ./image_1x/000.png can be upscaled to 200% size and match the associated
    patch from ./image_2x/000.png

    Arguments:
        subsets: List of directory path and their associated patch size. For
            example,
            [
                {
                    'path': '/project/datasets/div2k_train_sd/',
                    'size': 32
                }, {
                    'path': '/project/datasets/div2k_train_sdx4/',
                    'size': 128
                }, {
                    'path': '/project/datasets/div2k_train_hd/',
                    'size': 128
                }
            ]

            Represent images of 3 different scales from 3 different
            directories. Each yielded image patch should match the required
            size.

        batch_size: Number of patches for each yielding. A yielded result
            consists severl image batches (from different source) with shape
            (batch_size, patch_size, patch_size, 3).

        sample_rate: Number of small patches from a large image. Decreasing
            sample_rate will increase loading of image reading but also
            increase randomness of data.

        augment: Do basic image augmentation (vertical & horizontal flip,
            transpose).

    Return:
        A Tensorflow dataset which yield image batches from one or multiple
        source.
    """
    # NOTE: Sanity check.
    #       1. Find the smallest desired patch size.
    #       2. See if we have to force random cropping on even coordinates.
    #       3. See if the sizes sample images match their desired patch sizes.
    reference_patch_size, even_anchor = sanity_check(subsets)

    # NOTE: Represent each steam with it's image path and patch size.
    output_types = tuple([(tf.string, tf.int64) for _ in subsets])

    # NOTE: Read images in random order and infinitely.
    dataset = tf.data.Dataset.from_generator(
        functools.partial(infinite_image_streams, subsets), output_types)

    # NOTE: Read images. After this node, each stream becomes a tuple of image
    #       and patch size.
    dataset = dataset.map(read_images, tf.data.experimental.AUTOTUNE)

    # NOTE: Sample specified number (sample_rate) of patches from each image.
    #       Incresing sample_rate will reduce I/O loading and decrease
    #       randomness (number of patches from one image).
    dataset = dataset.repeat(sample_rate)

    # NOTE: Crop small patches from large images.
    # NOTE: All the 4 parameters can not be **kwargs. Don't know why.
    referenced_patch_sampling_function = functools.partial(
        sample_image_patches,
        reference_patch_size,
        even_anchor)

    dataset = dataset.map(
        referenced_patch_sampling_function, tf.data.experimental.AUTOTUNE)

    if augment:
        dataset = dataset.map(random_augment)

    dataset = dataset.map(cast_pixels, tf.data.experimental.AUTOTUNE)

    # NOTE: Batch the patches.
    dataset = dataset.batch(batch_size)

    # NOTE: Prefetch batches for optimization.
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # NOTE: Remap each batch stream to it's original key.
    dataset = dataset.map(functools.partial(finalize_batches, subsets))

    return dataset
