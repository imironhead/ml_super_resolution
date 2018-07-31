"""
"""
import argparse
import hashlib
import multiprocessing
import os
import skimage.io
import skimage.transform


def process_images(params):
    """
    """
    for index, path in enumerate(params['source_image_paths']):
        if index % 100 == 0:
            num_paths = len(params['source_image_paths'])
            task_index = params['index']
            print('[{}] done {} / {}'.format(task_index, index, num_paths))

        if params['keep_name']:
            base_name = os.path.basename(path)
            target_name = os.path.splitext(base_name)[0]
        else:
            target_name = hashlib.md5(path.encode('utf-8')).hexdigest()

        target_name = target_name + '.' + params['extension']
        target_path = os.path.join(params['target_dir_path'], target_name)

        # NOTE: we have the file already
        if os.path.isfile(target_path):
            continue

        image = skimage.io.imread(path)

        # NOTE: we are not interested in mono images
        if len(image.shape) != 3:
            continue

        h, w, c = image.shape

        # NOTE: only interested in rgb images
        if c != 3:
            continue

        # NOTE: ignore small images
        if params['drop'] and min(h, w) < params['drop_size_threshold']:
            continue

        # NOTE: do central crop
        if params['crop']:
            s = min(h, w)
            x = (w - s) // 2
            y = (h - s) // 2

            w = h = s

            image = image[y:y+h, x:x+w]

        # NOTE: do scale
        if params['scale']:
            # NOTE: always scale to square image
            if w != h:
                continue

            image = skimage.transform.resize(
                image, (params['scale_size'], params['scale_size']))

        skimage.io.imsave(target_path, image)


def prepare_images(args):
    """
    """
    # NOTE: collect all images under source_dir_path
    source_image_paths = []

    for file_name in os.listdir(args.source_dir_path):
        root, ext = os.path.splitext(file_name)

        if ext.lower() not in ['.png', '.jpg', '.jpeg']:
            continue

        source_image_path = os.path.join(args.source_dir_path, file_name)

        source_image_paths.append(source_image_path)

    # NOTE: split all images into num_processes groups
    tasks = [{
        'index': index,
        'keep_name': args.keep_name,
        'extension': args.extension,
        'target_dir_path': args.target_dir_path,
        'drop': args.drop,
        'drop_size_threshold': args.drop_size_threshold,
        'crop': args.crop,
        'scale': args.scale,
        'scale_size': args.scale_size,
        'source_image_paths': []} for index in range(args.num_processes)]

    num_processes = args.num_processes
    num_paths = len(source_image_paths)
    num_paths_per_task = (num_paths + num_processes - 1) // num_processes

    for begin in range(0, num_paths, num_paths_per_task):
        index = begin // num_paths_per_task
        end = min(begin + num_paths_per_task, num_paths)

        tasks[index]['source_image_paths'] = source_image_paths[begin:end]

    # NOTE: dispatch the tasks
    with multiprocessing.Pool(num_processes) as pool:
        pool.map(process_images, tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--num-processes',
        default=8,
        type=int,
        help='num of processes for the tasks')

    parser.add_argument(
        '--extension',
        default='png',
        type=str,
        help='format of the target images, can be "png" or "jpg"')

    parser.add_argument(
        '--source-dir-path',
        type=str,
        help='the dir which contains source image files')

    parser.add_argument(
        '--target-dir-path',
        type=str,
        help='the dir for keeping the processed images')

    parser.add_argument(
        '--drop',
        action='store_true',
        help='drop small image')

    parser.add_argument(
        '--drop-size-threshold',
        default=256,
        type=int,
        help='drop images that min(h, w) is less than this value')

    parser.add_argument(
        '--crop',
        action='store_true',
        help='central crop images')

    parser.add_argument(
        '--scale',
        action='store_true',
        help='scale the images')

    parser.add_argument(
        '--scale-size',
        default=64,
        type=int,
        help='scale images to scale_size (e.g. 256 to 128/64/32)')

    parser.add_argument(
        '--keep-name',
        action='store_true',
        help='keep file name or use hash of source path')

    args = parser.parse_args()

    # NOTE: sanity check, only support jpg and png
    if args.extension != 'jpg' and args.extension != 'png':
        raise Exception('non-supported extension: {}'.format(args.extension))

    if not os.path.isdir(args.source_dir_path):
        raise Exception('invalid source dir: {}'.format(args.source_dir_path))

    if not os.path.isdir(args.target_dir_path):
        raise Exception('invalid target dir: {}'.format(args.target_dir_path))

    prepare_images(args)

