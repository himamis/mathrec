from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = ['Keras==2.1.6',
                     'h5py==2.7.1',
                     'Pillow>=2.0.0',
                     'google-apitools==0.5.23',
                     'google-cloud-storage>=1.10',
                     'numpy>=1.9.1',
                     'opencv-python>=3.4.2',
                     # 'tensorflow==1.9',
                     # 'tensorflow-gpu==1.12',
                     'gitpython==2.1.11',
                     'pypng==0.0.18',
                     'imgaug==0.2.6',
                     'parsy==1.2.0',
                     'google-cloud-logging==1.8.0']
                     #'sklearn==0.19.1'
                     #]

setup(
    name='trainer',
    version='0.3',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(exclude=("backgrounds",)),
    include_package_data=True,
    description='Keras trainer application'
)
