#!/usr/bin/env python

"""
Use max likelihood estimation on a pair of sequences.

This script is meant to be somewhat temporary.
Please pillage it for useful parts and then delete it.
"""

import argparse
import csv

import numpy as np

from slowedml import design, fileutil



def main(args):

    # read the description of the genetic code
    with open(args.code) as fin_gcode:
        arr = list(csv.reader(fin_gcode, delimiter='\t'))
        indices, aminos, codons = zip(*arr)
        if [int(x) for x in indices] != range(len(indices)):
            raise ValueError

    # read the (nstates, nstates) array of counts
    counts = np.loadtxt(args.count_matrix)

    log_likelihood = 42

    # write the max log likelihood
    with fileutil.open_or_stdout(args.o, 'w') as fout:
        print >> fout, log_likelihood


if __name__ == '__main__':

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count-matrix', required=True,
            help='matrix of codon state change counts from human to chimp')
    parser.add_argument('--code', required=True,
            help='path to the human mitochondrial genetic code')
    parser.add_argument('-o', default='-',
            help='max log likelihood (default is stdout)')

    main(parser.parse_args())

