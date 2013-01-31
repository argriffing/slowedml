#!/usr/bin/env python

"""
Given a shape (N, N) ndarray, compute a shape (N,) symmetric sum reduction.
"""

import argparse

import numpy as np

from slowedml import fileutil


def main(args):

    with fileutil.open_or_stdin(args.i) as fin:
        M = np.loadtxt(fin, delimiter='\t')
        v = np.sum(M, axis=0) + np.sum(M, axis=1)

    with fileutil.open_or_stdout(args.o, 'w') as fout:
        np.savetxt(fout, v, fmt='%g', delimiter='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', default='-',
            help='input text table of numbers (default is stdin)')
    parser.add_argument('-o', default='-',
            help='output sum is written here (default is stdout)')
    main(parser.parse_args())
