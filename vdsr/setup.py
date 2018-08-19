from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['scikit-image>=0.14.0', 'pillow>=5.2.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='My training application package.'
    )
