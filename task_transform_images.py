"""
To load, transform and save all images withing a specific directory.
"""
import argparse
import multiprocessing
import os
import pathlib

import skimage.io
import skimage.transform


def process_image(task):
    """
    Load, transform and save one image.

    Arguments:
        task: A dictionary contains information of how to transform one image.
    """
    _, ext = os.path.splitext(task['file_name'])

    if ext.lower() not in ['.png', '.jpg', '.jpeg']:
        return

    source_image_path = \
        os.path.join(task['source_dir_path'], task['file_name'])
    target_image_path = \
        os.path.join(task['target_dir_path'], task['file_name'])

    # NOTE: we have the file already
    if os.path.isfile(target_image_path):
        return

    image = skimage.io.imread(source_image_path)

    # NOTE: We are interested in RGB images.
    if len(image.shape) != 3 or image.shape[-1] != 3:
        return

    size_h, size_w, _ = image.shape

    # NOTE: Ignore small images
    if task['drop']:
        if task['drop_size_lower_bound'] > min(size_h, size_w):
            return

    # TODO: Split large images to smaller ones.

    # NOTE: Do cropping.
    if task['crop']:
        if task['crop_mode'] == 'square':
            crop_size_h = crop_size_w = min(size_h, size_w)
        elif task['crop_mode'] == 'mod':
            crop_mod_size = task['crop_size_mod']

            crop_size_w = size_w - size_w % crop_mod_size
            crop_size_h = size_h - size_h % crop_mod_size
        else:
            crop_size_w = size_w
            crop_size_h = size_h

        offset_x = (size_w - crop_size_w) // 2
        offset_y = (size_h - crop_size_h) // 2

        size_h = crop_size_h
        size_w = crop_size_w

        image = image[offset_y : offset_y+size_h, offset_x : offset_x+size_w]

    # NOTE: Do scaling.
    if task['scale']:
        scale_n = task['scale_numerator']
        scale_d = task['scale_denominator']

        # NOTE: Sanity check. Softly fallback to prevent interupping the other
        #       tasks.
        if size_h * scale_n % scale_d != 0:
            print(f'Invalid scaling: {size_h} * {scale_n} % {scale_d}')
            return

        if size_w * scale_n % scale_d != 0:
            print(f'Invalid scaling: {size_w} * {scale_n} % {scale_d}')
            return

        size_h = size_h * scale_n // scale_d
        size_w = size_w * scale_n // scale_d

        # TODO: Consider scipy.misc.imresize.
        image = skimage.transform.resize(image, (size_h, size_w))

    skimage.io.imsave(target_image_path, image)


def prepare_images(arguments):
    """
    Dispatch images to multi-processes for transformation.

    Arguments:
        arguments: All information on how to transform each image.
    """
    # NOTE: Make arguments a dict.
    arguments = vars(arguments)

    # NOTE: Collect all images under source_dir_path.
    names = os.listdir(arguments['source_dir_path'])
    tasks = [{'file_name': name, **arguments} for name in names]

    # NOTE: dispatch the tasks
    with multiprocessing.Pool(arguments['num_processes']) as pool:
        pool.map(process_image, tasks)


def main():
    """
    Main function to do transforming jobs.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num-processes',
        default=8,
        type=int,
        help='Number of processes for the tasks')

    parser.add_argument(
        '--source-dir-path',
        type=str,
        help='The dir which contains source image files')

    parser.add_argument(
        '--target-dir-path',
        type=str,
        help='The dir for keeping the processed images')

    parser.add_argument(
        '--drop',
        action='store_true',
        help='Drop specific images')

    parser.add_argument(
        '--drop-size-lower-bound',
        default=256,
        type=int,
        help='Drop images that min(h, w) is less than this value')

    # TODO: Split large images to smaller ones.

    parser.add_argument(
        '--crop',
        action='store_true',
        help='Crop images')

    parser.add_argument(
        '--crop-mode',
        default='mod',
        type=str,
        help='Mode of cropping image [mod/square]')

    parser.add_argument(
        '--crop-size-mod',
        default=2,
        type=int,
        help='Crop the images so their size are multiple of crop_size_mod')

    parser.add_argument(
        '--scale',
        action='store_true',
        help='Scale the images')

    parser.add_argument(
        '--scale-numerator',
        default=64,
        type=int,
        help='Combined with denominator as scaling factor')

    parser.add_argument(
        '--scale-denominator',
        default=64,
        type=int,
        help='Combined with numerator as scaling factor')

    args = parser.parse_args()

    # NOTE: Sanity check.
    if not os.path.isdir(args.source_dir_path):
        raise ValueError(f'invalid source dir: {args.source_dir_path}')

    os.makedirs(args.target_dir_path, exist_ok=True)

    if not os.path.isdir(args.target_dir_path):
        raise ValueError(f'invalid target dir: {args.target_dir_path}')

    args.source_dir_path = pathlib.Path(args.source_dir_path).expanduser()
    args.target_dir_path = pathlib.Path(args.target_dir_path).expanduser()

    if args.source_dir_path == args.target_dir_path:
        raise ValueError('source_dir_path & target_dir_path must be different')

    prepare_images(args)


if __name__ == '__main__':
    main()
