"""
Reproduce the max likelihood for a codon model of Yang and Nielsen 1998.
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

import design
import alignll

def get_Q(
        ts, tv, syn, nonsyn,
        stationary_distn,
        log_mu, log_kappa, log_omega,
        ):
    """
    @param ts: 2d binary transition nucleotide change design matrix
    @param tv: 2d binary transversion nucleotide change design matrix
    @param syn: 2d binary synonymous nucleotide change design matrix
    @param nonsyn: 2d binary nonsynonymous nucleotide change design matrix
    @param stationary_distn: a fixed codon stationary distribution
    @param log_mu: free param for expected number of substitutions per time
    @param log_kappa: free param for nucleotide ts/tv rate ratio
    @param log_omega: free param for nonsyn/syn rate ratio
    @return: rate matrix
    """

    # exponentiate the free parameters
    mu = algopy.exp(log_mu)
    kappa = algopy.exp(log_kappa)
    omega = algopy.exp(log_omega)

    # construct a matrix whose off-diagonals are proportional to the rates
    pre_Q = (kappa * ts + tv) * (omega * nonsyn + syn) * stationary_distn

    # use the row sums to compute the rescaled rate matrix
    r = algopy.sum(pre_Q, axis=1)
    old_rate = algopy.dot(r, stationary_distn)
    Q = (mu / old_rate) * (pre_Q - algopy.diag(r))
    return Q

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
        log_mu = log_mus[i]
        log_omega = log_omegas[i]
        Q = get_Q(
                ts, tv, syn, nonsyn,
                stationary_distn,
                log_mu, log_kappa, log_omega)
        P = algopy.expm(Q)
        transition_matrices.append(P)

    # return the neg log likelihood
    npatterns = patterns.shape[0]
    nstates = patterns.shape[1]
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
    log_mu_0 = theta[0]
    log_mu_1 = theta[1]
    log_mu_2 = theta[2]
    log_kappa = theta[3]
    log_omega = theta[4]

    # construct the transition matrices
    transition_matrices = []
    for log_mu in (
            log_mu_0,
            log_mu_1,
            log_mu_2,
            ):
        Q = get_Q(
                ts, tv, syn, nonsyn,
                stationary_distn,
                log_mu, log_kappa, log_omega)
        P = algopy.expm(Q)
        transition_matrices.append(P)

    # return the neg log likelihood
    npatterns = patterns.shape[0]
    nstates = patterns.shape[1]
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


def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))

def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))


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
    # The full_compo has shape (ncodons, 3, 4)
    # whereas the count matrix has shape (3, 4).
    position_specific_nt_counts = np.dot(
            np.transpose(full_compo, (1, 2, 0)),
            v_emp)
    v_smoothed = np.exp(np.tensordot(
        full_compo,
        np.log(position_specific_nt_counts),
        axes=((1,2), (0,1)),
        ))
    print 'smoothed empirical codon distribution before normalization:'
    print v_smoothed
    print
    v_smoothed /= np.sum(v_smoothed)
    print 'smoothed empirical codon distribution after normalization:'
    print v_smoothed
    print

    # define the stationary distribution of the rate matrix
    stationary_distn = v_smoothed

    # define the initial guess of the parameter values
    theta_model_A = np.array([
        -2, # log_mu_0
        -2, # log_mu_1
        -2, # log_mu_2
        1,  # log_kappa
        -3, # log_omega
        ], dtype=float)

    # define the initial guess of the parameter values
    theta_model_B = np.array([
        -2, # log_mu_0
        -2, # log_mu_1
        -2, # log_mu_2
        1,  # log_kappa
        -3, # log_omega_0
        -3, # log_omega_1
        -3, # log_omega_2
        ], dtype=float)

    # construct the args to the neg log likelihood function
    likelihood_args = (
            patterns, weights,
            stationary_distn,
            ts, tv, syn, nonsyn,
            )

    # define the objective function and the gradient and hessian
    f = functools.partial(get_neg_ll_model_B, *likelihood_args)
    g = functools.partial(eval_grad, f)
    h = functools.partial(eval_hess, f)

    # do the search, using information about the gradient and hessian
    """
    results = scipy.optimize.fmin_ncg(
            f,
            theta,
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

    results = scipy.optimize.fmin(
            f,
            theta_model_B,
            )

    # report a summary of the maximum likelihood search
    print results
    print numpy.exp(results)
    x = results[0]
    print numpy.exp(x)



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

