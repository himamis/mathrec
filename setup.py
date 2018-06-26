from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.0.4',
                     'h5py==2.7.1',
                     'Pillow>=2.0.0',
                     'google-apitools==0.5.23',
                     'apache-beam==0.6.0']

setup(
    name='trainer',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    description='Keras trainer application'
)
