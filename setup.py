"""
The core algorithm has been deferred to the pyfelscore Cython module.

This package no longer defines extension modules.
"""

from distutils.core import setup

setup(
        name = 'slowedml',
        version = '0.1',
        packages=[
            'slowedml',
            'slowedml.tests',
            ],
        scripts = [
            'bin/slowedml-expand-newick.py',
            'bin/slowedml-phylip-to-pattern.py',
            'bin/slowedml-reformat-yn-2008.py',
            'bin/slowedml-samp-pat-ind.py',
            'bin/slowedml-sly.py',
            'bin/slowedml-unique.py',
            'bin/slowedml-yn-1998.py',
            ],
        )


