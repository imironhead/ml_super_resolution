"""
"""
import importlib
import logging
import os
import pathlib

import numpy as np
import ruamel.yaml as yaml
import tensorflow as tf

import dataset


def resolve_strings(data, experiment_name):
    """
    Replace "{experiment_name}" with experiment_name in each string except
    dictionary key. If a dictionary key contains 'path', resolve its value
    (e.g. expand ~/).

    Arguments:
        data: a dictionary or a list for resolving string values.
        experiment_name: a string to replace "{experiment_name}" within data.
    """
    # NOTE: We expect data is either a list or a dictionary.
    iterator = enumerate(data) if isinstance(data, list) else data.items()

    for key, val in iterator:
        if isinstance(val, list) or isinstance(val, dict):
            resolve_strings(val, experiment_name)
        elif isinstance(val, str):
            val = val.replace('{experiment_name}', experiment_name)

            if isinstance(key, str) and 'path' in key:
                if pathlib.PurePath(val).is_absolute():
                    pass
                elif val.startswith('~'):
                    val = str(pathlib.Path(val).expanduser())
                else:
                    val = str(pathlib.Path(val).resolve())

            data[key] = val


def global_step(context):
    """
    Count the global training step base on current iterations of the most
    frequent optimizer and the record from checkpoint. Return the global
    training step.

    Arguments:
        context: experiment information in a dictionary.
    """
    step = 0

    # NOTE: optimizer.iterations starts from 0 for each training session (
    #       both on fresh start and load from checkpoint)
    for optimizer in context['optimizers'].values():
        step = max(step, int(optimizer.iterations))

    return step + context['experiment']['global_step']


def load_experiment(path):
    """
    Read experiment information from a yaml file. Return a dictionary contains
    basic configuration of this experiment as the experiment context.

    Arguments:
    path -- path to a fresh experiment yaml or a checkpoint yaml.

    Raises:
        ValueError: if path is invalid.
    """
    if not os.path.isfile(path):
        raise ValueError(f'Invalid experiment path: {path}')

    with open(path, 'r') as yaml_file:
        experiment = yaml.YAML(typ='safe').load(yaml_file)

    resolve_strings(experiment, experiment['name'])

    #
    logger = logging.getLogger(f'{experiment["name"]}')

    logger.setLevel(logging.DEBUG)

    #
    scribe = tf.summary.create_file_writer(experiment['summary']['path'])

    # NOTE: Keep the loaded experiment. It can be then saved as checkpoint with
    #       updated information. This way the checkpoint would be in the same
    #       format as a fresh experiment yaml.
    return {
        'experiment': experiment,
        'logger': logger,
        'scribe': scribe,
    }


def build_datasets(context):
    """
    Build all datasets base on experiment information and keep them in the
    context.

    Arguments:
        context: experiment information in a dictionary.
    """
    data_streams = context['experiment']['data_streams']

    context['data_streams'] = {}

    for name, data_stream in data_streams.items():
        tf_dataset = dataset.build_dataset(
            image_streams=data_stream['image_streams'],
            batch_size=data_stream['batch_size'],
            sample_rate=data_stream['sample_rate'],
            augment=data_stream['augment'])

        context['data_streams'][name] = iter(tf_dataset)


def build_models(context):
    """
    Build all models base on experiment information and keep them in the
    context.

    Arguments:
        context: experiment information in a dictionary.

    Raises:
        ValueError: if data_streams have not been built yet.
    """
    if 'data_streams' not in context:
        raise ValueError('build data_streams before building models')

    models_config = context['experiment']['models']
    optimizers_config = context['experiment']['optimizers']

    # NOTE: Build models and train the entire model for 1 step to build the
    #       network graph explicitly so we can load weights later.
    context['models'] = importlib \
        .import_module(models_config['name']) \
        .build_models(**models_config['parameters'])

    for config in optimizers_config.values():
        data_stream = context['data_streams'][config['data_stream']]

        model = context['models']['extensions'][config['extension_model']]

        model(**next(data_stream))

        # NOTE: We do not have to do back propagation since we need only the
        #       concreate graph to load weights.

    # NOTE: Load weights.
    for name, config in models_config['principals'].items():
        if not os.path.isfile(config.get('path', '') or ''):
            continue

        context['models']['principals'][name].load_weights(config['path'])

    # NOTE: Build optimizers.
    context['optimizers'] = {}

    for name, config in optimizers_config.items():
        if config['optimizer'].lower() == 'adam':
            optimizer = tf.keras.optimizers.Adam

        default = {}

        if 'learning_rate' in config:
            default['learning_rate'] = config['learning_rate']

        context['optimizers'][name] = \
            optimizer.from_config(config.get('config', {}) or default)


def train(context):
    """
    Iterate through all optimizers and train their associated models for one
    step.

    Arguments:
        context: experiment information in a dictionary.
    """
    step = global_step(context)

    for name, optimizer in context['optimizers'].items():
        config = context['experiment']['optimizers'][name]

        if step % config['cycle'] != 0:
            continue

        data_stream = context['data_streams'][config['data_stream']]

        model = context['models']['extensions'][config['extension_model']]

        # TODO: batch_size_multiplier

        fetched = model(**next(data_stream))

        optimizer.apply_gradients(
            zip(fetched['gradients'], fetched['variables']))

        with context['scribe'].as_default():
            tf.summary.scalar(f'loss[{name}]', data=fetched['loss'], step=step)

        context['logger'].info(f'loss[{name}][{step}]: {fetched["loss"]}')


def validate(context):
    """
    Iterate through all validators and do basic validation for their associated
    models.

    Arguments:
        context: experiment information in a dictionary.
    """
    step = global_step(context)

    for name, config in context['experiment']['validators'].items():
        if step % config['cycle'] != 0:
            continue

        data_stream_name = config['data_stream']['name']

        data_stream = context['data_streams'][data_stream_name]

        data = next(data_stream)

        # NOTE: Inputs must conform the requirements of the principal model.
        inputs = [data[name] for name in config['data_stream']['parameters']]

        model = context['models']['principals'][config['principal_model']]

        sr_images = model(inputs)
        hd_images = data[config['hd_images']]

        psnr = tf.image.psnr(sr_images, hd_images, 2.0)
        psnr = np.mean(psnr)

        ssim = tf.image.ssim(sr_images, hd_images, 2.0)
        ssim = np.mean(ssim)

        hd_image = np.concatenate(hd_images, axis=1)
        sr_image = np.concatenate(sr_images, axis=1)

        summary_image = np.concatenate([hd_image, sr_image], axis=0)
        summary_image = [summary_image * 0.5 + 0.5]

        with context['scribe'].as_default():
            tf.summary.scalar(f'psnr[{name}]', data=psnr, step=step)
            tf.summary.scalar(f'ssim[{name}]', data=ssim, step=step)
            tf.summary.image(f'hd-sr[{name}]', data=summary_image, step=step)

        context['logger'].info(f'psnr[{name}][{step}]: {psnr}')
        context['logger'].info(f'ssim[{name}][{step}]: {ssim}')


def save(context):
    """
    Save the experiment.

    Arguments:
        context: experiment information in a dictionary.
    """
    step = global_step(context)

    experiment = context['experiment']

    if step % experiment['checkpoint']['cycle'] != 0:
        return

    experiment['global_step'] = step

    # NOTE: Save model weights.
    for model_name, config in experiment['models']['principals'].items():
        name = f'{str(step).rjust(16, "0")}_{model_name}.h5'
        path = os.path.join(experiment['checkpoint']['path'], name)

        config['path'] = path

        model = context['models']['principals'][model_name]

        model.save_weights(path)

    # NOTE: Update optimizers' configs.
    for name, config in experiment['optimizers'].items():
        config['config'] = context['optimizers'][name].get_config()

    # NOTE: Save checkpoint.
    name = f'{str(step).rjust(16, "0")}_checkpoint.yaml'
    path = os.path.join(experiment['checkpoint']['path'], name)

    with open(path, 'w') as yaml_file:
        # NOTE: Add a representer to handle numpy.float32 as float.
        def represent_numpy_float32(representer, data):
            return representer.represent_float(data)

        dumper = yaml.YAML()

        dumper.representer.add_representer(np.float32, represent_numpy_float32)
        dumper.dump(experiment, stream=yaml_file)


def train_validate_save(experiment_path):
    """
    Build the experiment and train the models.

    Arguments:
        experiment_path: path to a yaml which contains information of the
            experiment.
    """
    context = load_experiment(experiment_path)

    build_datasets(context)
    build_models(context)

    for _ in range(100):
        train(context)
        validate(context)
        save(context)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        raise ValueError('Usage: python task_train.py experiment_ooxx.json')

    train_validate_save(sys.argv[1])
