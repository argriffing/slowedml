#!/usr/bin/env python

import argparse

import numpy as np

from slowedml import fileutil


def main(args):

    with fileutil.open_or_stdin(args.i) as fin:
        total_sum = np.sum(np.loadtxt(fin))

    with fileutil.open_or_stdout(args.o, 'w') as fout:
        print >> fout, total_sum


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-i', default='-',
            help='input text table of numbers (default is stdin)')
    parser.add_argument('-o', default='-',
            help='output sum is written here (default is stdout)')
    main(parser.parse_args())
