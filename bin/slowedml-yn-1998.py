#!/usr/bin/env python

"""
Reproduce the max likelihood for a codon model of Yang and Nielsen 1998.

I am confused about whether the 3*(4-1) nucleotide frequencies
are free parameters or whether they are empirically estimated.
"""

import argparse
import functools
import math
import csv

import numpy as np
import numpy
import algopy
import scipy
import scipy.optimize

from slowedml import design
from slowedml import alignll
from slowedml import codon1994, markovutil
from slowedml.algopyboilerplate import eval_grad, eval_hess

def get_neg_ll_model_B(
        patterns, pattern_weights,
        stationary_distn,
        ts, tv, syn, nonsyn,
        theta,
        ):
    """
    This model has multiple omega parameters.
    @param theta: vector of free variables with sensitivities
    """

    # unpack theta
    log_mus = theta[0:3]
    log_kappa = theta[3]
    log_omegas = theta[4:7]

    # construct the transition matrices
    transition_matrices = []
    for i in range(3):
        mu = algopy.exp(log_mus[i])
        kappa = algopy.exp(log_kappa)
        omega = algopy.exp(log_omegas[i])
        pre_Q = codon1994.get_pre_Q(
                ts, tv, syn, nonsyn,
                stationary_distn,
                kappa, omega)
        Q = markovutil.pre_Q_to_Q(pre_Q, stationary_distn, mu)
        P = algopy.expm(Q)
        transition_matrices.append(P)

    # return the neg log likelihood
    ov = range(4)
    v_to_children = {3 : [0, 1, 2]}
    de_to_P = {
            (3, 0) : transition_matrices[0],
            (3, 1) : transition_matrices[1],
            (3, 2) : transition_matrices[2],
            }
    root_prior = stationary_distn
    log_likelihood = alignll.fast_fels(
            ov, v_to_children, de_to_P, root_prior,
            patterns, pattern_weights,
            )
    neg_ll = -log_likelihood
    print neg_ll
    return neg_ll

def get_neg_ll_model_A(
        patterns, pattern_weights,
        stationary_distn,
        ts, tv, syn, nonsyn,
        theta,
        ):
    """
    This model has only a single omega parameter.
    @param theta: vector of free variables with sensitivities
    """

    # unpack theta
    log_mus = theta[0:3]
    log_kappa = theta[3]
    log_omega = theta[4]

    # construct the transition matrices
    transition_matrices = []
    for i in range(3):
        mu = algopy.exp(log_mus[i])
        kappa = algopy.exp(log_kappa)
        omega = algopy.exp(log_omega)
        pre_Q = codon1994.get_pre_Q(
                ts, tv, syn, nonsyn,
                stationary_distn,
                kappa, omega)
        Q = markovutil.pre_Q_to_Q(pre_Q, stationary_distn, mu)
        P = algopy.expm(Q)
        transition_matrices.append(P)

    # return the neg log likelihood
    ov = range(4)
    v_to_children = {3 : [0, 1, 2]}
    de_to_P = {
            (3, 0) : transition_matrices[0],
            (3, 1) : transition_matrices[1],
            (3, 2) : transition_matrices[2],
            }
    root_prior = stationary_distn
    log_likelihood = alignll.fast_fels(
    #log_likelihood = alignll.fels(
            ov, v_to_children, de_to_P, root_prior,
            patterns, pattern_weights,
            )
    neg_ll = -log_likelihood
    print neg_ll
    return neg_ll


def get_neg_ll_model_A_free(
        patterns, pattern_weights,
        ts, tv, syn, nonsyn, full_compo,
        theta,
        ):
    
    # pick the nt distn parameters from the end of the theta vector
    log_nt_distns = algopy.zeros((3, 4), dtype=theta)
    log_nt_distns_block = algopy.reshape(theta[-9:], (3, 3))
    log_nt_distns[:, :-1] = log_nt_distns_block
    reduced_theta = theta[:-9]
    unnormalized_nt_distns = algopy.exp(log_nt_distns)

    # normalize each of the three nucleotide distributions
    row_sums = algopy.sum(unnormalized_nt_distns, axis=1)
    nt_distns = (unnormalized_nt_distns.T / row_sums).T

    # get the implied codon distribution
    stationary_distn = codon1994.get_f3x4_codon_distn(
            full_compo,
            nt_distns,
            )

    return get_neg_ll_model_A(
        patterns, pattern_weights,
        stationary_distn,
        ts, tv, syn, nonsyn,
        reduced_theta,
        )


def get_guess_A_free():
    return np.array([
        -2, -2, -2, # logs of branch lengths
        1,          # log kappa
        -3,         # log omega
        0, 0, 0,    # \
        0, 0, 0,    #  logs of nucleotide probability ratios
        0, 0, 0,    # /
        ], dtype=float)

def get_guess_A():
    return np.array([
        -2, # log_mu_0
        -2, # log_mu_1
        -2, # log_mu_2
        1,  # log_kappa
        -3, # log_omega
        ], dtype=float)

def get_guess_B():
    return np.array([
        -2, # log_mu_0
        -2, # log_mu_1
        -2, # log_mu_2
        1,  # log_kappa
        -3, # log_omega_0
        -3, # log_omega_1
        -3, # log_omega_2
        ], dtype=float)



def main(args):

    # read the description of the genetic code
    with open(args.code_in) as fin_gcode:
        arr = list(csv.reader(fin_gcode, delimiter='\t'))
        indices, aminos, codons = zip(*arr)
        if [int(x) for x in indices] != range(len(indices)):
            raise ValueError

    # load the ordered directed edges
    DE = np.loadtxt(args.edges_in, delimiter='\t', dtype=int)

    # load the alignment pattern
    patterns = np.loadtxt(args.patterns_in, delimiter='\t', dtype=int)

    # load the alignment weights
    weights = np.loadtxt(args.weights_in, delimiter='\t', dtype=float)

    # get the empirical codon distribution
    ncodons = len(codons)
    nsites = patterns.shape[0]
    ntaxa = patterns.shape[1]
    v_emp = np.zeros(ncodons, dtype=float)
    for i in range(nsites):
        for j in range(ntaxa):
            state = patterns[i, j]
            if state != -1:
                v_emp[state] += weights[i]
    v_emp /= np.sum(v_emp)
    print 'empirical codon distribution:'
    print v_emp
    print

    # precompute some design matrices
    adj = design.get_adjacency(codons)
    ts = design.get_nt_transitions(codons)
    tv = design.get_nt_transversions(codons)
    syn = design.get_syn(codons, aminos)
    nonsyn = design.get_nonsyn(codons, aminos)
    full_compo = design.get_full_compo(codons)

    # For all of the data in the alignment,
    # compute the grand total nucleotide counts at each of the three
    # nucleotide positions within a codon.
    # The full_compo ndarray has shape (ncodons, 3, 4)
    # whereas the nucleotide distribution ndarray has shape (3, 4).
    position_specific_nt_distns = np.tensordot(full_compo, v_emp, axes=(0,0))

    # This is pre-computed if we want to use an empirical
    # stationary distribution, but it is postponed if we want to use
    # max likelihood parameters for the stationary distribution.
    stationary_distn = codon1994.get_f3x4_codon_distn(
            full_compo,
            position_specific_nt_distns,
            )

    # construct the args to the neg log likelihood function
    likelihood_args_empirical = (
            patterns, weights,
            stationary_distn,
            ts, tv, syn, nonsyn,
            )
    
    likelihood_args_free = (
        patterns, weights,
        ts, tv, syn, nonsyn, full_compo,
        )

    # get the model A estimates using plain fmin
    model_A_opt = scipy.optimize.fmin(
            functools.partial(get_neg_ll_model_A, *likelihood_args_empirical),
            get_guess_A(),
            )
    print 'optimal params for model A:'
    print np.exp(model_A_opt)
    print

    # reconstruct the matrix
    d = position_specific_nt_distns
    lastcol = d[:, -1]
    dratio = (d.T / lastcol).T
    log_nt_guess = np.log(dratio[:, :-1]).reshape(9)
    guess = np.hstack([model_A_opt, log_nt_guess])


    # choose the model
    likelihood_args = likelihood_args_free
    #guess = get_guess_A_free
    get_neg_ll = get_neg_ll_model_A_free
    #likelihood_args = likelihood_args_empirical
    #guess = theta_model_A
    #get_neg_ll = get_neg_ll_model_A

    # define the objective function and the gradient and hessian
    f = functools.partial(get_neg_ll, *likelihood_args)
    g = functools.partial(eval_grad, f)
    h = functools.partial(eval_hess, f)

    # do the search, using information about the gradient and hessian
    """
    results = scipy.optimize.fmin_ncg(
            f,
            theta_model_A,
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
    """

    #"""
    results = scipy.optimize.fmin(
            f,
            guess,
            )
    #"""

    # report a summary of the maximum likelihood search
    print results
    print numpy.exp(results)
    #x = results[0]
    #print numpy.exp(x)



if __name__ == '__main__':

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-in', required=True,
            help='an input file that defines the genetic code')
    parser.add_argument('--edges-in', required=True,
            help='ordered tree edges')
    parser.add_argument('--patterns-in', required=True,
            help='codon alignment pattern')
    parser.add_argument('--weights-in', required=True,
            help='codon alignment weights')

    # run the code with the args
    main(parser.parse_args())

