#!/usr/bin/env python
# -*- encoding: utf-8 -*-
from glob import glob
from os.path import basename
from os.path import splitext

from setuptools import find_packages
from setuptools import setup


setup(
    name='mlmodule',
    version='0.1.2',
    description='Model repository for the data platform',
    url='https://github.com/LSIR/mlmodule',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    py_modules=[splitext(basename(path))[0] for path in glob('src/*.py')],
    include_package_data=True,
    zip_safe=False,
    python_requires='>=3.4',
    install_requires=[
    ],
    extras_require={
        'torch': ['torch', 'torchvision'],
        'clip': ['ftfy', 'regex', 'tqdm', 'clip @ git+https://github.com/openai/CLIP.git']
    },
    setup_requires=[
        'pytest-runner',
    ],
)
