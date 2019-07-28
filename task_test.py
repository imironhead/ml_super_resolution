"""
Build and load a model for super resolution from a checkpoint. Then super
resolve images and save the results.
"""
import argparse
import importlib
import logging
import os
import pathlib

import imageio
import numpy as np
import ruamel.yaml as yaml


def resolve_path(path, experiment_name):
    """
    Resolve a path string.

    Arguments:
        path: A string which represents a path. It may be absolute, related to
            home (~) or current working path.
        experiment_name: A string to replace '{experiment_name}' with path.

    Return:
        A resolved path.
    """
    path = path.replace('{experiment_name}', experiment_name)

    if pathlib.PurePath(path).is_absolute():
        pass
    elif path.startswith('~'):
        path = str(pathlib.Path(path).expanduser())
    else:
        path = str(pathlib.Path(path).resolve())

    return path


def read_image_as_batch(path):
    """
    Read a image and expand its dimension to make it as a batch (then shape is
    [1, height, width, 3]).

    Arguments:
        path: Path to a image file.

    Return:
        A numpy array which represents an image. The image pixel values range
        from -1.0 to +1.0.
    """
    image = imageio.imread(path).astype(np.float32) / 127.5 - 1.0

    return np.expand_dims(image, 0)


def save_image(path, image):
    """
    Rescale the image and save it.

    Arguments:
        path: Target path to save the image.
        image: A numpy array which represents an image. Its pixel values range
            from -1.0 to 1.0.
    """
    image = np.clip(image * 127.5 + 127.5, 0.0, 255.0).astype(np.uint8)
    imageio.imwrite(path, image)


def load_experiment(checkpoint_path):
    """
    Load an experiment from a checkpoint. We need the information (e.g. the
    trained model weights) of the main super resolution model (which accepts
    low resolution images and return super-resolved images).

    Arguments:
        checkpoint_path: Path to a pre-trained checkpoint.

    Return:
        Context of the experiment which contains necessary information for this
        test.
    """
    # NOTE: If the checkpoint_path is a directory, assume it contains
    #       checkpoints and try to find the most uptodate one.
    if os.path.isdir(checkpoint_path):
        names = os.listdir(checkpoint_path)
        names = filter(lambda name: name.endswith('_checkpoint.yaml'), names)
        names = sorted(list(names))

        if not names:
            raise ValueError(f'Found no checkpoint with {checkpoint_path}')

        checkpoint_path = os.path.join(checkpoint_path, names[-1])

    if not os.path.isfile(checkpoint_path):
        raise ValueError(f'{checkpoint_path} is not a file for checkpoint')

    with open(checkpoint_path, 'r') as yaml_file:
        experiment = yaml.YAML(typ='safe').load(yaml_file)

    if 'name' not in experiment:
        raise ValueError('An experiment needs a name.')

    logger = logging.getLogger(experiment['name'])

    logger.setLevel(logging.INFO)
    logger.info('test on: %s', experiment['name'])

    return {
        'experiment': experiment,
        'logger': logger,
    }


def build_datasets(context, source_paths, result_path):
    """
    Build the datasets. A datasets consists of many image groups. Each group
    contains multiple low resolution image paths and one result image path. The
    test will try to super resolve each group with its low resolution images
    and save the result to result path.

    Arguments:
        context: The context which contains experiment information.
        source_paths: A list of paths. They should be all file paths or all
            directory paths. Note that the order of paths with source_paths
            must the same as the inputs order of the model.
        result_path: A path to a file if all source_paths are file paths. Or a
            path to a directory if all source_paths are directory paths.
    """
    experiment_name = context['experiment']['name']

    source_paths = map(
        lambda path: resolve_path(path, experiment_name), source_paths)
    source_paths = list(source_paths)

    result_path = resolve_path(result_path, experiment_name)

    if not source_paths:
        raise ValueError(f'Invalid source paths {source_paths}')

    if not (os.path.isfile(source_paths[0]) or os.path.isdir(source_paths[0])):
        raise ValueError(f'Invalid source paths {source_paths}')

    checksum = sum(map(
        lambda path: 1 if os.path.isdir(path) else 0, source_paths))

    if checksum != 0 and checksum != len(source_paths):
        raise ValueError(f'Invalid source paths {source_paths}')

    os.makedirs(result_path, exist_ok=True)

    # NOTE: file paths
    if checksum == 0:
        basename = os.path.basename(source_paths[0])

        context['datasets'] = [{
            'source': source_paths,
            'result': os.path.join(result_path, basename)}]

        return

    # NOTE: directory paths
    datasets = []

    for name in os.listdir(source_paths[0]):
        for path in source_paths:
            path = os.path.join(path, name)

            if not os.path.isfile(path):
                raise ValueError('Invalid source path {path}')

        datasets.append({
            'source': [os.path.join(path, name) for path in source_paths],
            'result': os.path.join(result_path, name)})

    context['datasets'] = datasets


def build_model(context):
    """
    Build the model for super resolution.

    Arguments:
        context: The context which contains experiment information.
    """
    models_config = context['experiment']['models']

    models = importlib \
        .import_module(models_config['name']) \
        .build_models(**models_config['parameters'])

    # NOTE: Use the principal model of the first validator to do super
    #       resolution. Princiapl models are the models which keep model
    #       weights under the hood. For example, a generator or a
    #       discriminator. A validator must use a principal model to do super
    #       resolution. That's what we want here.
    validator = context['experiment']['validators'][0]

    model_name = validator['principal_model']

    model = models['principals'][model_name]

    weights_path = resolve_path(
        models_config['principals'][model_name]['path'],
        context['experiment']['name'])

    # NOTE: Do super resolution once to build layer variables. Then we can load
    #       weights for them.
    source = context['datasets'][0]['source']
    source = [read_image_as_batch(path) for path in source]

    model(source)

    model.load_weights(weights_path)

    context['model'] = model


def super_resolve(context):
    """
    Do super resolution.

    Arguments:
        context: The context which contains experiment information.
    """
    model = context['model']
    datasets = context['datasets']

    for data in datasets:
        source = [read_image_as_batch(path) for path in data['source']]

        result = model(source)

        save_image(data['result'], result[0])

        context['logger'].info('done: %s', data['result'])


def test(checkpoint_path, source_paths, result_path):
    """
    Do the test experiment.

    Arguments:
        checkpoint_path: Path to a pre-trained model.
        source_paths: Paths to images (to do one super resolution) or paths to
            directories (to do multiple super resolutions).
        result_path: Path to an image path if all source_paths are file paths.
            Or Path to a directory path if all source_paths are directory
            paths.
    """
    context = load_experiment(checkpoint_path)

    build_datasets(context, source_paths, result_path)
    build_model(context)

    super_resolve(context)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Super resolving images.')

    parser.add_argument(
        '--checkpoint',
        type=str,
        help='Path to a checkpoint file or a directory contains checkpoints.')

    # NOTE: The order of multiple source must conform to the inputs of the
    #       principal model. For example, ENet need one low resolution image
    #       and one bicubic upscaled 4x image. Then the first --source must be
    #       for sd_images and the second --source for bq_images (check
    #       models/model_enet.Generator).
    parser.add_argument(
        '--source',
        action='append',
        type=str,
        help='Source image path or path to a directory which contains images.')

    parser.add_argument(
        '--result',
        type=str,
        help='Result image path or path to a directory to keep result images.')

    args = parser.parse_args()

    test(args.checkpoint, args.source, args.result)
