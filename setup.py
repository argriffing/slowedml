"""
The core algorithm has been deferred to the pyfelscore Cython module.

This package no longer defines extension modules.
"""

from distutils.core import setup

import os


script_directory = 'bin'
script_filenames = []
for name in os.listdir(script_dir):
    if name.endswith('.py'):
        filename = os.path.join(script_directory, name)
        script_filenames.append(filename)


setup(
        name = 'slowedml',
        version = '0.1',
        packages=[
            'slowedml',
            'slowedml.tests',
            ],
        scripts = script_filenames,
        )


