#!/usr/bin/env python
# -*- encoding: utf-8 -*-
import subprocess
from glob import glob
from os import environ
from os.path import basename
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


def get_version():
    version = environ.get('MLMODULE_BUILD_VERSION')
    if not version:
        try:
            version = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        except Exception:
            version = 'Unknown'
    return version


setup(
    name='mlmodule',
    version=get_version(),
    description='Model repository for the data platform',
    url='https://github.com/LSIR/mlmodule',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    package_data={'mlmodule': ['contrib/arcface/normalized_faces.npy']},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.4',
    install_requires=[
    ],
    extras_require={
        'torch': ['torch', 'torchvision', 'tqdm'],
        'clip': ['ftfy', 'regex', 'tqdm', 'clip @ git+https://github.com/openai/CLIP.git'],
        'mtcnn': ['facenet-pytorch']
    },
    setup_requires=[
        'pytest-runner',
    ],
)
