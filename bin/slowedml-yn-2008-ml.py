#!/usr/bin/env python

"""
Use max likelihood estimation on a pair of sequences.

This script is meant to be somewhat temporary.
Please pillage it for useful parts and then delete it.
"""

import functools
import argparse
import csv

import numpy as np
import algopy

from slowedml import design
from slowedml import fileutil
from slowedml import fmutsel
from slowedml.algopyboilerplate import eval_grad, eval_hess


##############################################################################
# These are related to two-taxon log likelihood calculations.

def get_two_taxon_neg_ll(
        subs_counts, log_counts, v,
        h,
        ts, tv, syn, nonsyn, compo, asym_compo,
        theta,
        ):
    """
    @param theta: unconstrained vector of free variables
    """

    # break theta into a branch length vs. other parameters
    branch_length = theta[0]
    reduced_theta = theta[1:]

    # get the unscaled rate matrix
    pre_Q = fmutsel.get_pre_Q(
            log_counts,
            h,
            ts, tv, syn, nonsyn, compo, asym_compo,
            reduced_theta)
    unscaled_Q = pre_Q - algopy.diag(algopy.sum(pre_Q, axis=1))

    # get the matrix exponential of the rescaled rate matrix
    curr_scale = -algopy.dot(algopy.diag(unscaled_Q), v)
    Q = unscaled_Q * (branch_length / curr_scale)
    P = algopy.expm(Q)

    # return the negative log likelihood
    log_likelihoods = algopy.log(P.T * v) * subs_counts
    neg_ll = -algopy.sum(log_likelihoods)
    return neg_ll


##############################################################################
# Try to minimize the neg log likelihood.


def main(args):

    # read the description of the genetic code
    with open(args.code) as fin_gcode:
        arr = list(csv.reader(fin_gcode, delimiter='\t'))
        indices, aminos, codons = zip(*arr)
        if [int(x) for x in indices] != range(len(indices)):
            raise ValueError

    # precompute some numpy ndarrays using the genetic code
    ts = design.get_nt_transitions(codons)
    tv = design.get_nt_transversions(codons)
    syn = design.get_syn(codons, aminos)
    nonsyn = design.get_nonsyn(codons, aminos)
    compo = design.get_compo(codons)
    nt_sinks = design.get_nt_sinks(codons)
    asym_compo = np.transpose(nt_sinks, (1, 2, 0))

    # read the (nstates, nstates) array of observed codon substitutions
    subs_counts = np.loadtxt(args.count_matrix)

    # compute some summaries of the observed codon substitutions
    counts = np.sum(subs_counts, axis=0) + np.sum(subs_counts, axis=1)
    log_counts = np.log(counts)
    v = counts / float(np.sum(counts))

    # guess the values of the free params
    guess = np.array([
        -3, # log branch length
        1,  # log kappa
        -3, # log omega
        0,  # log (pi_A / pi_T)
        0,  # log (pi_C / pi_T)
        0,  # log (pi_G / pi_T)
        ], dtype=float)

    # construct the neg log likelihood non-free params
    fmin_args = (
            subs_counts, log_counts, v,
            fmutsel.fixation_h,
            ts, tv, syn, nonsyn, compo, asym_compo,
            )

    # define the objective function and the gradient and hessian
    f = functools.partial(get_two_taxon_neg_ll, *fmin_args)
    g = functools.partial(eval_grad, f)
    h = functools.partial(eval_hess, f)

    # do the search, using information about the gradient and hessian
    results = scipy.optimize.fmin_ncg(
            f,
            guess,
            g,
            fhess_p=None,
            fhess=h,
            avextol=1e-05,
            epsilon=1.4901161193847656e-08,
            maxiter=100,
            full_output=True,
            disp=1,
            retall=0,
            callback=None,
            )

    # report a summary of the maximum likelihood search
    with fileutil.open_or_stdout(args.o, 'w') as fout:
        print >> fout, results
        x = results[0]
        print >> fout, np.exp(x)


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

