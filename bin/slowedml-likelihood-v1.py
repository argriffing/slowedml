#!/usr/bin/env python

"""
Input matrices are formatted as tabular text files.

Use masked arrays to deal with counts that are equal to zero.
Allow the output to be redirected internally
for better compatibility with systems such as parallel 'make'
for which the stdin and stdout do not behave as expected.
"""

import argparse

import numpy as np

from slowedml import fileutil


def main(args, fout):

    counts = np.loadtxt(args.counts)

    log_distn = None
    if args.initial_distn:
        log_distn = ma.log(np.loadtxt(args.initial_distn))
    elif args.log_initial_distn:
        log_distn = np.loadtxt(args.log_initial_distn)
    
    log_trans = None
    if args.transition_matrix:
        log_trans = np.log(np.loadtxt(args.transition_matrix))
    elif args.log_transition_matrix:
        log_trans = np.loadtxt(args.log_transition_matrix)

    log_joint = None
    if args.joint_prob_matrix:
        log_joint = np.log(np.loadtxt(args.joint_prob_matrix))
    elif args.log_joint_prob_matrix:
        log_joint = np.loadtxt(args.log_joint_prob_matrix)
    else:
        log_joint = log_distn + log_trans

    mask = np.equal(counts, 0)
    log_likelihood = np.sum(counts * np.ma.array(log_joint, mask=mask))

    if args.report_log_likelihood:
        print >> fout, log_likelihood
    else:
        print >> fout, np.exp(log_likelihood)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--counts', required=True,
            help='state transition counts')
    
    likelihood = parser.add_mutually_exclusive_group(required=True)
    likelihood.add_argument('--transition-matrix',
            help='transition matrix')
    likelihood.add_argument('--log-transition-matrix',
            help='entrywise log of the transition matrix')
    likelihood.add_argument('--joint-prob-matrix',
            help='joint probability matrix')
    likelihood.add_argument('--log-joint-prob-matrix',
            help='entrywise log of the joint probability matrix')

    distn = parser.add_mutually_exclusive_group()
    distn.add_argument('--initial-distn',
            help='prior probability distribution over initial states')
    distn.add_argument('--log-initial-distn',
            help='entrywise log of prior distribution over initial states')

    parser.add_argument('--report-log-likelihood', action='store_true',
            help='report the log likelihood instead of the likelihood')

    parser.add_argument('-o', default='-',
            help='send output here (default is stdout)')

    # read the args
    args = parser.parse_args()

    with fileutil.open_or_stdout(args.o, 'w') as fout:
        main(args, fout)

