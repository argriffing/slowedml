"""
The core algorithm has been deferred to the pyfelscore Cython module.

So this package no longer includes extension modules.
"""

from distutils.core import setup

setup(
        name = 'slowedml',
        version = '0.1',
        packages=[
            'slowedml'
            'slowedml.tests',
            ],
        #scripts = ['bin/my-script.py'],
        )
