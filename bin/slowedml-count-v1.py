#!/usr/bin/env python

"""
"""

import argparse
import csv

import numpy as np

from slowedml import fileutil
from slowedml import moretypes


def main(args, fout):

    # read the ordered taxon subset
    with open(args.taxa) as fin:
        arr = list(csv.reader(fin, delimiter='\t'))
        indices, requested_taxa = zip(*arr)
        if [int(x) for x in indices] != range(len(indices)):
            raise ValueError
        taxon_to_index = dict((x, i) for i, x in enumerate(requested_taxa))

    # get the index of the initial taxon
    initial_index = taxon_to_index[args.initial_taxon_name]

    # get the index of the final taxon
    final_index = taxon_to_index[args.final_taxon_name]

    # get the number of states
    nstates = args.nstates

    # read the patterns
    patterns = np.loadtxt(args.patterns, dtype=int, ndmin=2)
    npatterns = patterns.shape[0]

    # extract the columns of the patterns that we care about
    initial_states = patterns[:, initial_index]
    final_states = patterns[:, final_index]

    # check for missing data or states that are outside of the state space
    for states in (initial_states, final_states):
        if np.any(np.less(states, 0)):
            raise NotImplementedError('missing data is not implemented')
        if np.any(np.greater_equal(states, nstates)):
            raise ValueError('some observed states are greater than expected')

    # read the pattern weights
    if args.pattern_weights:
        weights = np.loadtxt(args.pattern_weights, dtype=float)
    else:
        weights = np.ones(npatterns, dtype=float)

    # compute the weighted counts
    counts = np.zeros((nstates, nstates), dtype=float)
    for istate, jstate, weight in zip(initial_states, final_states, weights):
        counts[istate, jstate] += weight

    # write the count array
    np.savetxt(fout, counts, fmt='%g', delimiter='\t')


if __name__ == '__main__':

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--taxa', required=True,
            help='a tabular file that defines the taxon ordering')
    parser.add_argument('--patterns', required=True,
            help='each row of this integer array is a pattern'),
    parser.add_argument('--pattern-weights',
            help='a list of pattern weights (by default all weights are 1)')
    parser.add_argument('--initial-taxon-name', required=True,
            help='name of the initial taxon')
    parser.add_argument('--final-taxon-name', required=True,
            help='name of the final taxon')
    parser.add_argument('--nstates', type=moretypes.pos_int,
            help='size of the Markov state space')
    parser.add_argument('--counts-out', default='-',
            help='write the count matrix here (default is stdout)')

    args = parser.parse_args()

    with fileutil.open_or_stdout(args.counts_out, 'w') as fout:
        main(args, fout)

