"""
The core algorithm has been deferred to the pyfelscore Cython module.

So this package no longer includes extension modules.
"""

from distutils.core import setup

my_package_name = 'slowedml'

setup(
        name = my_package_name,
        version = '0.1',
        packages=[my_package_name],
        #scripts = ['bin/my-script.py'],
        )
