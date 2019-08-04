"""
Compare two images or images from two directories. Evaluate them with
pre-defined metrics and generate reports in pre-defined formats.
"""
import argparse
import logging
import os
import pathlib

import numpy as np
import ruamel.yaml as yaml
import skimage.io


def collect_images(hr_image_path, sr_image_path):
    """
    Collect paths of high resolution images and super-resolved images.

    Arguments:
        hr_image_path: A path to a high resolution image file. Or a path to a
            directory contains multiple high resolution image files.
        sr_image_path: A path to a super-resolved image file. Or a path to a
            directory contains multiple super-resolved image files.

    Raises:
        ValueError: The function works for two types of arguments:
            1. Both hr_image_path & sr_image_path are file paths.
            2. Both hr_image_path & sr_image_path are directory paths.
            If they are not, raise ValueError.

    Return:
        List of paired image paths. Each element is a dictionary of one path to
        a high resolution image and one path to a super-resolved image path.
        However, we assume both file paths are for images without checking
        them.
    """
    if hr_image_path is None or sr_image_path is None:
        raise ValueError('Both hr_image_path & sr_image_path can not be None.')

    if os.path.isfile(hr_image_path) != os.path.isfile(sr_image_path):
        raise ValueError(
            'One of hr_image_path and sr_image_path is a file while the'
            'other one is not.')

    if os.path.isdir(hr_image_path) != os.path.isdir(sr_image_path):
        raise ValueError(
            'One of hr_image_path and sr_image_path is a directory while the'
            'other one is not.')

    if os.path.isfile(hr_image_path):
        return [{
            'hr_image_path': hr_image_path,
            'sr_image_path': sr_image_path}]

    logger = logging.getLogger(__name__)

    names = os.listdir(hr_image_path)

    names = sorted(names)

    image_paths = []

    for name in names:
        _, ext = os.path.splitext(name)

        path = os.path.join(sr_image_path, name)

        if ext.lower() not in ['.png', '.jpg']:
            logger.debug('Found invalid file name: %s', name)
            continue

        if not os.path.isfile(path):
            logger.debug('Found invalid file path: %s', path)
            continue

        image_paths.append({
            'hr_image_path': os.path.join(hr_image_path, name),
            'sr_image_path': os.path.join(sr_image_path, name)})

    return image_paths


def report_to_yaml(report_path, reports):
    """
    Save the generated reports to a YAML file.

    Arguments:
        report_path: Target path for the reports. An extention of YAML (.yaml)
            will be appended.
        reports: A dictionary of generated reports.
    """
    logger = logging.getLogger(__name__)

    path = report_path + '.yaml'

    logger.info('Writing YAML: %s', path)

    with open(path, 'w') as yaml_file:
        dumper = yaml.YAML()

        dumper.dump(reports, stream=yaml_file)


def report_to_markdown(report_path, reports):
    """
    Save the generated reports to a markdown file.

    Arguments:
        report_path: Target path for the reports. An extention of markdown
            (.md) will be appended.
        reports: A dictionary of generated reports.
    """
    def value_to_str(value):
        if type(value) == float:
            return f'{value:.4f}'
        else:
            return str(value)

    if not reports or not reports.get('reports'):
        return

    columns = [key for key in reports['reports'][0].keys() if key != 'name']
    columns = ['name'] + columns

    logger = logging.getLogger(__name__)

    path = report_path + '.md'

    logger.info('Writing YAML: %s', path)

    with open(path, 'w') as md_file:
        md_file.write('## General Information\n')
        md_file.write(reports['description'] + '\n')

        md_file.write(f'|{"|".join(columns)}|\n')
        md_file.write(f'|{"---|" * len(columns)}\n')

        for report in reports['reports']:
            values = [value_to_str(report[column]) for column in columns]

            md_file.write(f'|{"|".join(values)}|\n')

        md_file.write('---\n')


def validate(description, hr_image_path, sr_image_path, report_path):
    """
    Validate image processing (super resolution) results and make reports.

    Arguments:
        description: A string to be added to the reports.
        hr_image_path: A path to a high resolution image file. Or a path to a
            directory which contains multiple high resolution image files.
        sr_image_path: A path to a super resolved image file. Or a path to a
            directory which contains multiple super resolved image files.
        report_path:
    """
    # TODO: Use skimage.metrics when 0.16 is released.
    def psnr(hr_image, sr_image):
        mse = np.mean((hr_image - sr_image) ** 2)

        return 10.0 * np.log10(1.0 / mse)

    logger = logging.getLogger(__name__)

    validators = [
#       {'name': 'psnr', 'eval': skimage.metrics.peak_signal_noise_ratio,},
#       {'name': 'ssim', 'eval': skimage.metrics.structural_similarity,},
        {'name': 'psnr', 'eval': psnr},
    ]

    reports = {
        'description': description,
        'hr_image_path': hr_image_path,
        'sr_image_path': sr_image_path,
        'reports': [],
    }

    image_info_collection = collect_images(hr_image_path, sr_image_path)

    for image_info in image_info_collection:
        report = {
            'name': os.path.basename(image_info['hr_image_path']),
        }

        hr_image = skimage.io.imread(image_info['hr_image_path'])
        sr_image = skimage.io.imread(image_info['sr_image_path'])

        hr_image = skimage.img_as_float(hr_image)
        sr_image = skimage.img_as_float(sr_image)

        for validator in validators:
            score = validator['eval'](hr_image, sr_image)

            report[validator['name']] = float(score)

            logger.info(
                '%s - %s: %f',
                os.path.basename(image_info['hr_image_path']),
                validator['name'],
                report[validator['name']])

        reports['reports'].append(report)

    reporters = [report_to_yaml, report_to_markdown]

    for reporter in reporters:
        reporter(report_path, reports)


def main():
    """
    Entry point of validation task.
    """
    parser = argparse.ArgumentParser(
        description='Validate super resolved images')

    parser.add_argument(
        '--hr_image_path',
        type=str,
        help='')

    parser.add_argument(
        '--sr_image_path',
        type=str,
        help='')

    parser.add_argument(
        '--report_path',
        type=str,
        help='')

    parser.add_argument(
        '--description',
        default='',
        type=str,
        help='')

    args = parser.parse_args()

    validate(
        args.description,
        args.hr_image_path,
        args.sr_image_path,
        args.report_path)


if __name__ == '__main__':
    main()
