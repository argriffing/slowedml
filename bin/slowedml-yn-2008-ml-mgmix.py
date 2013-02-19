#!/usr/bin/env python

"""
Use max likelihood estimation on a pair of sequences.

This will be a bizarre site-class model.
In particular, it will be a mixture of two site-classes.
The first class will be F1x4 Muse-Gaut.
The second class will be an FMutSel model generalized
by adding a preferred codon recessivity parameter.
For the latter more richly parameterized site-class,
the equilibrium codon distribution will be estimated
by expectation maximization.
Expectation maximization will also be used to estimate
the mixing proportion.
The kappa parameter, the omega parameter, and the three degrees of freedom
among the four mutational process nucleotide frequencies
will all be shared between the two site-classes.
For parameterization we will use the term 'natural' to mean
natural for interpretation,
and we will use the term 'encoded' to mean
natural for optimization.
The distinction between encoded and natural forms may involve
transformations such as logit/expit or log/exp.
"""

import functools
import argparse
import csv

import numpy as np
import scipy.optimize
import scipy.linalg
import algopy

from slowedml import design, fileutil
from slowedml import fmutsel, codon1994, markovutil
from slowedml import codon1994models, yn2008models
from slowedml.algopyboilerplate import eval_grad, eval_hess


def get_two_taxon_neg_ll_encoded_theta(
        model,
        em_probs, em_distns,
        subs_counts,
        ts, tv, syn, nonsyn, compo, asym_compo,
        encoded_theta,
        ):
    """
    Get the negative log likelihood.
    This function uses the logarithms of the model parameters.
    The first param group is the model implementation.
    The second param group is expectation-maximization stuff.
    The third param group is the data.
    The next param group consists of design matrices related to genetic code.
    The next param group consist of free parameters of the model.
    """
    branch_length = algopy.exp(encoded_theta[0])
    encoded_model_theta = encoded_theta[1:]
    natural_model_theta = model.encoded_to_natural(
            encoded_model_theta)
    natural_theta = algopy.zeros_like(encoded_theta)
    natural_theta[0] = branch_length
    natural_theta[1:] = natural_model_theta
    return get_two_taxon_neg_ll(
        model,
        em_probs, em_distns,
        subs_counts,
        ts, tv, syn, nonsyn, compo, asym_compo,
        natural_theta,
        )

def get_two_taxon_neg_ll(
        model,
        em_probs, em_distns,
        subs_counts,
        ts, tv, syn, nonsyn, compo, asym_compo,
        natural_theta,
        ):
    """
    Get the negative log likelihood.
    This function does not use the logarithms.
    It is mostly for computing the hessian;
    otherwise the version with the logarithms would probably be better.
    The first param group is the model implementation.
    The second param group is expectation-maximization stuff.
    The third param group is the data.
    The next param group consists of design matrices related to genetic code.
    The next param group consist of free parameters of the model.
    """

    # unpack some parameters
    branch_length = natural_theta[0]
    natural_model_theta = natural_theta[1:]

    # compute the appropriately scaled transition matrices
    pre_Qs = model.get_pre_Qs(
            em_probs, em_distns,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_model_theta)
    eq_distns = model.get_distns(
            em_probs, em_distns,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_model_theta)
    Ps = markovutil.get_branch_mix(em_probs, pre_Qs, eq_distns, branch_length)

    # compute the mixture transition matrix
    P_mix = algopy.zeros_like(Ps[0])
    P_mix += em_probs[0] * (Ps[0].T * eq_distns[0]).T
    P_mix += em_probs[1] * (Ps[1].T * eq_distns[1]).T

    # compute the neg log likelihood
    neg_ll = -algopy.sum(algopy.log(P_mix) * subs_counts)
    print neg_ll
    return neg_ll


def get_posterior_expectations(subs_counts, Ps, prior_probs, prior_eq_distns):
    """
    Get posterior expectations for various quantities.
    The input transition matrices should be appropriately scaled
    so that their mixture gives the correct expected number of
    substitutions per time unit.
    This is pure numpy, not algopy, because it is not differentiated.
    @param subs_counts: observed substitution counts for the two-taxon data
    @param Ps: transition matrices
    @param prior_probs: prior distribution over the two site-classes
    @return: posterior site-class distn, posterior codon distns
    """
    nstates = subs_counts.shape[0]
    post_probs = np.zeros_like(prior_probs)
    nsites = np.sum(subs_counts)
    post_eq_weights = np.zeros_like(prior_eq_distns)
    for i in range(nstates):
        for j in range(nstates):
            nsubs = subs_counts[i, j]
            likelihood_0 = prior_eq_distns[0][i] * Ps[0][i, j]
            likelihood_1 = prior_eq_distns[1][i] * Ps[1][i, j]
            p0 = likelihood_0 / (likelihood_0 + likelihood_1)
            p1 = likelihood_1 / (likelihood_0 + likelihood_1)
            post_probs[0] += p0 * nsubs / float(nsites)
            post_probs[1] += p1 * nsubs / float(nsites)
            post_eq_weights[0][i] += p0 * nsubs
            post_eq_weights[0][j] += p0 * nsubs
            post_eq_weights[1][i] += p1 * nsubs
            post_eq_weights[1][j] += p1 * nsubs
    post_eq_distns = post_eq_weights
    post_eq_distns[0] /= float(np.sum(post_eq_distns[0]))
    post_eq_distns[1] /= float(np.sum(post_eq_distns[1]))
    return post_probs, post_eq_distns


class F1x3MG_FMutSel_F:

    @classmethod
    def get_natural_guess():
        return np.array([
            3.0, # kappa
            0.1, # omega
            1.0, # ratio A/T
            1.0, # ratio C/T
            1.0, # ratio G/T
            ])

class F1x3MG_FMutSel_F:

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 5:
            raise ValueError(len(theta))



class F1x3MG_FMutSelG_F:

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 6:
            raise ValueError(len(theta))

    @classmethod
    def natural_to_encoded(cls, natural_theta):
        cls.check_theta(natural_theta)
        encoded_theta = algopy.zeros_like(natural_theta)
        encoded_theta[0] = natural_theta[0]
        encoded_theta[1:] = algopy.exp(natural_theta[1:])
        return encoded_theta

    @classmethod
    def encoded_to_natural(cls, encoded_theta):
        natural_theta = algopy.zeros_like(encoded_theta)
        natural_theta[0] = encoded_theta[0]
        natural_theta[1:] = algopy.log(encoded_theta[1:])
        cls.check_theta(natural_theta)
        return natural_theta

    @classmethod
    def get_natural_guess(cls):
        return np.array([
            0.0, # kimura d
            3.0, # kappa
            0.1, # omega
            1.0, # ratio A/T
            1.0, # ratio C/T
            1.0, # ratio G/T
            ])

    @classmethod
    def get_initial_em_params(cls, natural_theta):
        f1x4mg_natural_theta = natural_theta[1:]
        em_probs = np.array([0.5, 0.5], dtype=float)
        em_distns = np.vstack((
            codon1994models.F1x4MG.get_distn(
                log_counts, empirical_codon_distn,
                ts, tv, syn, nonsyn, compo, asym_compo,
                f1x4mg_natural_theta),
            yn2008models.FMutSel_F.get_distn(
                log_counts, empirical_codon_distn,
                ts, tv, syn, nonsyn, compo, asym_compo,
                f1x4mg_natural_theta),
            ))
        return em_probs, em_distns

    @classmethod
    def get_distns(cls,
            em_probs, em_distns,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta):
        """
        The equilibrium distributions are quite distinct and weird.
        For the mg-like component, the equilibrium codon distribution
        is a function of the free parameters in the mutational process.
        For the fmutsel-like component, the equilibrium codon distribution
        is determined by an em-like component
        which is not purely empirical and which is not defined by
        free parameters.
        """
        f1x4mg_natural_theta = natural_theta[1:]
        d0 = codon1994models.F1x4MG.get_distn(
                None, None,
                ts, tv, syn, nonsyn, compo, asym_compo,
                f1x4mg_natural_theta)
        d1 = em_distns[1]
        return d0, d1

    @classmethod
    def get_pre_Qs(cls,
            em_probs, em_distns,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta):
        f1x4mg_natural_theta = natural_theta[1:]
        pre_Q_0 = codon1994models.F1x4MG.get_pre_Q(
            None, None,
            ts, tv, syn, nonsyn, compo, asym_compo,
            f1x4mg_natural_theta)
        pre_Q_1 = yn2008models.FMutSelG_F.get_pre_Q(
            algopy.log(em_distns[1]), em_distns[1],
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta)
        return pre_Q_0, pre_Q_1


def main(args):

    # read the description of the genetic code
    with open(args.code) as fin_gcode:
        arr = list(csv.reader(fin_gcode, delimiter='\t'))
        indices, aminos, codons = zip(*arr)
        if [int(x) for x in indices] != range(len(indices)):
            raise ValueError

    aminos = [x.lower() for x in aminos]
    nstop = aminos.count('stop')
    if nstop not in (2, 3, 4):
        raise Exception('expected 2 or 3 or 4 stop codons')
    if any(x == 'stop' for x in aminos[:-nstop]):
        raise Exception('expected stop codons at the end of the genetic code')

    # trim the stop codons
    aminos = aminos[:-nstop]
    codons = codons[:-nstop]

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

    # trim the stop codons
    subs_counts = subs_counts[:-nstop, :-nstop]

    # compute some summaries of the observed codon substitutions
    counts = np.sum(subs_counts, axis=0) + np.sum(subs_counts, axis=1)
    log_counts = np.log(counts)
    empirical_codon_distn = counts / float(np.sum(counts))

    # make crude guesses about parameter values
    blen = markovutil.guess_branch_length(subs_counts)
    theta = args.model.get_natural_guess()

    # Get the initial guesses for the EM parameters.
    prior_probs = np.array([0.5, 0.5], dtype=float)
    prior_em_distns = np.vstack((
        empirical_codon_distn,
        empirical_codon_distn,
        ))

    # iteratively compute parameter estimates
    for em_iteration_index in range(10):

        # given parameter guesses, compute the pre-rate matrices
        pre_Qs = args.model.get_pre_Qs(
                prior_probs, prior_em_distns,
                ts, tv, syn, nonsyn, compo, asym_compo,
                theta)

        # compute the appropriately scaled transition matrices
        eq_distns = args.model.get_distns(
                prior_probs, prior_em_distns,
                ts, tv, syn, nonsyn, compo, asym_compo,
                theta)
        Ps = markovutil.get_branch_mix(
                prior_probs, pre_Qs, eq_distns, blen)

        # given parameter guesses, compute posterior expectations
        post_probs, post_em_distns = get_posterior_expectations(
                subs_counts, Ps, prior_probs, eq_distns)

        # given posterior expectations, optimize the parameter guesses
        encoded_theta = np.empty(len(theta) + 1, dtype=float)
        encoded_theta[0] = np.log(blen)
        encoded_theta[1:] = args.model.natural_to_encoded(theta)

        # construct the neg log likelihood non-free params
        neg_ll_args = (
                args.model,
                post_probs, post_em_distns,
                subs_counts,
                ts, tv, syn, nonsyn, compo, asym_compo,
                )

        # define the objective function and the gradient and hessian
        f_encoded_theta = functools.partial(
                get_two_taxon_neg_ll_encoded_theta, *neg_ll_args)
        g_encoded_theta = functools.partial(eval_grad, f_encoded_theta)
        h_encoded_theta = functools.partial(eval_hess, f_encoded_theta)

        # do the search, using information about the gradient and hessian
        results = scipy.optimize.minimize(
                f_encoded_theta,
                encoded_theta,
                method=args.minimization_method,
                jac=g_encoded_theta,
                hess=h_encoded_theta,
                )

        # extract and decode the maximum likelihood estimates
        encoded_xopt = results.x
        mle_log_blen = encoded_xopt[0]
        mle_blen = np.exp(mle_log_blen)
        model_encoded_xopt = encoded_xopt[1:]
        model_xopt = args.model.encoded_to_natural(
                model_encoded_xopt)
        xopt = np.empty_like(encoded_xopt)
        xopt[0] = mle_blen
        xopt[1:] = model_xopt

        # report a summary of the maximum likelihood search
        print 'raw results from the minimization:'
        print results
        print
        print 'max likelihood branch length (expected number of substitutions):'
        print mle_blen
        print
        print 'max likelihood estimates of other model parameters:'
        print model_xopt
        print

        # get ready for the next iteration if we continue
        blen = mle_blen
        f1x4mg_natural_theta = model_xopt
        prior_probs = post_probs
        prior_em_distns = post_em_distns



if __name__ == '__main__':

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)

    # let the user define the model
    model_choice = parser.add_mutually_exclusive_group(required=True)
    model_choice.add_argument(
            '--FMutSel-F-mix',
            dest='model',
            action='store_const',
            const=F1x3MG_FMutSel_F,
            )
    model_choice.add_argument(
            '--FMutSelG-F-mix',
            dest='model',
            action='store_const',
            const=F1x3MG_FMutSelG_F,
            ) 

    solver_names = (
            'Nelder-Mead',
            'Powell', 
            'CG', 
            'BFGS',
            'Newton-CG',
            'Anneal',
            'L-BFGS-B',
            'TNC',
            'COBYLA',
            'SLSQP',
            )

    parser.add_argument('--count-matrix', required=True,
            help='matrix of codon state change counts on the branch')
    parser.add_argument('--code', required=True,
            help='path to the genetic code definition')
    parser.add_argument('--minimization-method',
            choices=solver_names,
            default='BFGS',
            help='use this scipy.optimize.minimize method')
    """
    parser.add_argument('--neg-log-likelihood-out',
            help='write the minimized neg log likelihood to this file')
    parser.add_argument('--parameter-estimates-out',
            help='write the maximum likelihood parameter estimates here')
    parser.add_argument('--parameter-errors-out',
            help='write the parameter estimate standard errors here')
    """
    args = parser.parse_args()
    main(args)

