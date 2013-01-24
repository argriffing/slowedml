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

from slowedml import design, fileutil
from slowedml import fmutsel, codon1994, markovutil
from slowedml.algopyboilerplate import eval_grad, eval_hess


##############################################################################
# Two taxon F1 x 4.
# This model name uses a notation I saw in Yang-Nielsen 2008.

def get_two_taxon_f1x4_neg_ll(
        subs_counts,
        ts, tv, syn, nonsyn, compo, asym_compo,
        theta,
        ):
    """
    @param theta: unconstrained vector of free variables
    """

    # unpack theta
    branch_length = algopy.exp(theta[0])
    kappa = algopy.exp(theta[1])
    omega = algopy.exp(theta[2])
    nt_log_ratios = algopy.zeros(4, dtype=theta)
    nt_log_ratios[:-1] = theta[-3:]
    nt_unnormalized_distn = algopy.exp(nt_log_ratios)
    nt_distn = nt_unnormalized_distn / algopy.sum(nt_unnormalized_distn)

    # this might be a Goldman-Yang rather than Muse-Gaut approach
    print 'compo shape:', compo.shape
    print 'nt_distn shape:', nt_distn.shape
    codon_distn = codon1994.get_f1x4_codon_distn(compo, nt_distn)

    print 'ts shape:', ts.shape
    print 'tv shape:', tv.shape
    print 'syn shape:', syn.shape
    print 'nonsyn shape:', nonsyn.shape
    print 'codon distn shape:', codon_distn.shape
    print
    pre_Q = codon1994.get_pre_Q(
            ts, tv, syn, nonsyn,
            codon_distn, kappa, omega,
            )
    Q = markovutil.pre_Q_to_Q(pre_Q, codon_distn, branch_length)
    P = algopy.expm(Q)

    # check that the equilibrium distn was not falsely advertised
    """
    v_iter = algopy.dot(v, P)
    print 'v:'
    print v
    print
    print 'v_iter:'
    print v_iter
    print
    """

    # return the negative log likelihood
    log_likelihoods = algopy.log(P.T * codon_distn) * subs_counts
    neg_ll = -algopy.sum(log_likelihoods)
    print neg_ll, theta
    return neg_ll


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

    # compute the transition matrix
    pre_Q = fmutsel.get_pre_Q(
            log_counts,
            h,
            ts, tv, syn, nonsyn, compo, asym_compo,
            reduced_theta)
    Q = markovutil.pre_Q_to_Q(pre_Q, v, branch_length)
    P = algopy.expm(Q)

    # check that the equilibrium distn was not falsely advertised
    v_iter = algopy.dot(v, P)
    print 'v:'
    print v
    print
    print 'v_iter:'
    print v_iter
    print

    # return the negative log likelihood
    log_likelihoods = algopy.log(P.T * v) * subs_counts
    neg_ll = -algopy.sum(log_likelihoods)
    print neg_ll, theta
    return neg_ll


##############################################################################
# Try to minimize the neg log likelihood.

def neutral_h(x):
    """
    This is a fixation h function in the notation of Yang and Nielsen.
    """
    return algopy.ones_like(x)


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

    # read the (nstates, nstates) array of observed codon substitutions
    subs_counts = np.loadtxt(args.count_matrix, dtype=float)

    # trim the four mitochondrial stop codons
    subs_counts = subs_counts[:-4, :-4]

    # compute some summaries of the observed codon substitutions
    counts = np.sum(subs_counts, axis=0) + np.sum(subs_counts, axis=1)
    log_counts = np.log(counts)
    v = counts / float(np.sum(counts))

    """
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
    """

    # construct a guess based on paml
    """
    log_mu = np.log(0.43115)
    log_kappa = np.log(22.25603)
    log_omega = np.log(0.07232)
    pT = 0.24210
    pC = 0.32329
    pA = 0.31614
    pG = 0.11847
    """

    # construct a guess based a previous max likelihood estimate
    mu = 0.56371965
    kappa = 35.69335435
    omega = 0.04868303
    pA = 0.50462715
    pC = 0.30143984
    pG = 0.08668469
    pT = 0.10724831

    guess = np.array([
        np.log(mu),
        np.log(kappa),
        np.log(omega),
        np.log(pA / pT),
        np.log(pC / pT),
        np.log(pG / pT),
        ], dtype=float)

    nt_distn = np.array([pA, pC, pG, pT])
    v = codon1994.get_f1x4_codon_distn(compo, nt_distn)

    # construct the neg log likelihood non-free params
    fmin_args_fmutsel = (
            subs_counts, log_counts, v,
            #fmutsel.fixation_h,
            neutral_h,
            ts, tv, syn, nonsyn, compo, asym_compo,
            )

    fmin_args = (
        subs_counts,
        ts, tv, syn, nonsyn, compo, asym_compo,
        )

    # define the objective function and the gradient and hessian
    #f = functools.partial(get_two_taxon_neg_ll, *fmin_args)
    f = functools.partial(get_two_taxon_f1x4_neg_ll, *fmin_args)
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
            avextol=1e-06,
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
        print >> fout, 'probs assuming last three params are log nt probs:'
        kernel = np.exp(x[-3:].tolist() + [0])
        print kernel / np.sum(kernel)


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

