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
import scipy.optimize
import algopy

from slowedml import design
from slowedml import fileutil
from slowedml import fmutsel
from slowedml.algopyboilerplate import eval_grad, eval_hess


##############################################################################
# these are for checking for regressions

# http://en.wikipedia.org/wiki/File:Transitions-transversions-v3.png
g_ts = {'ag', 'ga', 'ct', 'tc'}
g_tv = {'ac', 'ca', 'gt', 'tg', 'at', 'ta', 'cg', 'gc'}

def get_hamming(codons):
    """
    Get the hamming distance between codons, in {0, 1, 2, 3}.
    @param codons: sequence of lower case codon strings
    @return: matrix of hamming distances
    """
    ncodons = len(codons)
    ham = np.zeros((ncodons, ncodons), dtype=int)
    for i, ci in enumerate(codons):
        for j, cj in enumerate(codons):
            ham[i, j] = sum(1 for a, b in zip(ci, cj) if a != b)
    return ham

def get_ts_tv(codons):
    """
    Get binary matrices defining codon pairs differing by single changes.
    @param codons: sequence of lower case codon strings
    @return: two binary numpy arrays
    """
    ncodons = len(codons)
    ts = np.zeros((ncodons, ncodons), dtype=int)
    tv = np.zeros((ncodons, ncodons), dtype=int)
    for i, ci in enumerate(codons):
        for j, cj in enumerate(codons):
            nts = sum(1 for p in zip(ci,cj) if ''.join(p) in g_ts)
            ntv = sum(1 for p in zip(ci,cj) if ''.join(p) in g_tv)
            if nts == 1 and ntv == 0:
                ts[i, j] = 1
            if nts == 0 and ntv == 1:
                tv[i, j] = 1
    return ts, tv

#def get_syn_nonsyn(code, codons):
def get_syn_nonsyn(inverse_table, codons):
    """
    Get binary matrices defining synonymous or nonynonymous codon pairs.
    @return: two binary matrices
    """
    ncodons = len(codons)
    #inverse_table = dict((c, i) for i, cs in enumerate(code) for c in cs)
    syn = np.zeros((ncodons, ncodons), dtype=int)
    for i, ci in enumerate(codons):
        for j, cj in enumerate(codons):
            if inverse_table[ci] == inverse_table[cj]:
                syn[i, j] = 1
    return syn, 1-syn

def get_compo(codons):
    """
    Get a matrix defining site-independent nucleotide composition of codons.
    @return: integer matrix
    """
    ncodons = len(codons)
    compo = np.zeros((ncodons, 4), dtype=int)
    for i, c in enumerate(codons):
        for j, nt in enumerate('acgt'):
            compo[i, j] = c.count(nt)
    return compo

def get_asym_compo(codons):
    """
    This is an ugly function.
    Its purpose is to help determine which is the mutant nucleotide type
    given an ordered pair of background and mutant codons.
    This determination is necessary if we want to follow
    the mutation model of Yang and Nielsen 2008.
    Entry [i, j, k] of the returned matrix gives the number of positions
    for which the nucleotides are different between codons i and j and
    the nucleotide type of codon j is 'acgt'[k].
    @return: a three dimensional matrix
    """
    ncodons = len(codons)
    asym_compo = np.zeros((ncodons, ncodons, 4), dtype=int)
    for i, ci in enumerate(codons):
        for j, cj in enumerate(codons):
            for k, nt in enumerate('acgt'):
                asym_compo[i, j, k] = sum(1 for a, b in zip(ci, cj) if (
                    a != b and b == nt))
    return asym_compo


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
    branch_length = algopy.exp(theta[0])
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
    print neg_ll, theta
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

    # trim the four mitochondrial stop codons
    aminos = aminos[:-4]
    codons = codons[:-4]


    # precompute some numpy ndarrays using the genetic code
    ts = design.get_nt_transitions(codons)
    tv = design.get_nt_transversions(codons)
    syn = design.get_syn(codons, aminos)
    nonsyn = design.get_nonsyn(codons, aminos)
    compo = design.get_compo(codons)
    nt_sinks = design.get_nt_sinks(codons)
    asym_compo = np.transpose(nt_sinks, (1, 2, 0))

    # check for regressions
    inverse_table = dict(zip(*(codons, aminos)))
    ts_old, tv_old = get_ts_tv(codons)
    syn_old, nonsyn_old = get_syn_nonsyn(inverse_table, codons)
    compo_old = get_compo(codons)
    asym_compo_old = get_asym_compo(codons)

    if not np.array_equal(ts, ts_old):
        raise Exception('transition regression')
    if not np.array_equal(tv, tv_old):
        raise Exception('transversion regression')

    # this is the same
    """
    if np.array_equal(compo, compo_old):
        raise Exception('compo is the same')
    else:
        raise Exception('compo is not the same')
    """

    """
    if asym_compo.shape == asym_compo_old.shape:
        raise Exception('same shape')
    else:
        raise Exception('different shape')
    """

    # not the same
    """
    if np.array_equal(asym_compo, asym_compo_old):
        raise Exception('asym compo is the same')
    else:
        raise Exception('asym compo is not the same')
    """


    # read the (nstates, nstates) array of observed codon substitutions
    subs_counts = np.loadtxt(args.count_matrix, dtype=float)

    # trim the four mitochondrial stop codons
    subs_counts = subs_counts[:-4, :-4]

    # compute some summaries of the observed codon substitutions
    counts = np.sum(subs_counts, axis=0) + np.sum(subs_counts, axis=1)
    log_counts = np.log(counts)
    v = counts / float(np.sum(counts))

    print 'counts:', counts
    print 'min counts:', min(counts)
    print
    print 'v:', v
    print 'min v:', min(v)

    total_count = np.sum(subs_counts)
    diag_count = np.sum(np.diag(subs_counts))
    mu_guess = (total_count - diag_count) / float(diag_count)
    log_mu_guess = np.log(mu_guess)
    print 'log mu guess:', log_mu_guess

    # guess the values of the free params
    guess = np.array([
        log_mu_guess, # log branch length
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

            # fail
            ts, tv, syn, nonsyn, compo, asym_compo,

            # ok
            #ts_old, tv_old, syn_old, nonsyn_old, compo_old, asym_compo_old,

            # fail
            #ts, tv, syn, nonsyn, compo, asym_compo_old,

            #ts, tv, syn, (1-syn), compo, asym_compo,
            )

    # define the objective function and the gradient and hessian
    f = functools.partial(get_two_taxon_neg_ll, *fmin_args)
    g = functools.partial(eval_grad, f)
    h = functools.partial(eval_hess, f)

    """
    results = scipy.optimize.fmin_bfgs(
            f,
            guess,
            g,
            )
    """

    #"""
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
    #"""

    """
    xopt = scipy.optimize.fmin(
            f,
            guess,
            )
    """

    # report a summary of the maximum likelihood search
    with fileutil.open_or_stdout(args.o, 'w') as fout:
        #print >> fout, xopt
        #"""
        print >> fout, results
        x = results[0]
        print >> fout, np.exp(x)
        #"""


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

