#!/usr/bin/env python

"""
Sample a pattern file assuming everything is completely independent.

In the context of continuous time Markov processes on sequences
related by a tree structure whose branch lengths represent the
expected number of changes along the branch,
the independence assumption is like assuming that all branches
are infinitely long.
By default the pattern weights are each one.
"""

import sys
import argparse

import numpy as np
import numpy.random

import pedant


def main(args):
    if args.distn_in is None:
        if args.nstates is None:
            raise Exception(
                    'If the state distribution is not specified, '
                    'then the number of states must be provided.')
        distn = np.ones(distn.nstates, dtype=float)
        distn /= np.sum(distn)
    else:
        distn = np.loadtxt(args.distn_in)
    pedant.assert_valid_distn(distn)
    if args.nstates is None:
        nstates = len(distn)
    else:
        nstates = args.nstates
    if nstates != len(distn):
        raise Exception(
                'The specified number of states does not match '
                'the length of the provided state distribution.')
    ntaxa = args.nleaves + args.ninternal
    X_leaves = np.random.choice(
            nstates,
            (args.npatterns, args.nleaves),
            replace=True,
            p=distn,
            )
    X_internal = -np.ones(args.npatterns, args.ninternal)
    X = np.hstack((X_leaves, X_internal))
    np.savetxt(args.patterns_out, X, fmt='%d')
    if args.weights_out is not None:
        np.savetxt(args.weights_out, [1]*args.npatterns, fmt='%d')



if __name__ == '__main__':

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nstates', type=pedant.pos_int,
            help='the process has this many possible states')
    parser.add_argument('--npatterns', type=pedant.pos_int, required=True,
            help='sample this many patterns')
    parser.add_argument('--nleaves', type=pedant.pos_int, required=True,
            help='sample states for this many taxa')
    parser.add_argument('--ninternal', type=pedant.nonneg_int, default=0,
            help='append this many -1 states to each pattern')
    parser.add_argument('--distn-in',
            help='state distribution (default is uniform)')
    parser.add_argument('--patterns-out', required=True,
            help='the sampled site patterns')
    parser.add_argument('--weights-out',
            help='pattern weights')

    # sample the patterns
    main(parser.parse_args())

