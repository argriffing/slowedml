#!/usr/bin/env python

"""
Use max likelihood estimation on a pair of sequences.

Mixture models are sufficiently different from models defined
by a single rate matrix that I am going to implement them
in this separate module.
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
import algopy.special

from slowedml import design, fileutil
from slowedml import fmutsel, codon1994, markovutil
from slowedml.algopyboilerplate import eval_grad, eval_hess


#XXX this is the same as for non-mixture models
def guess_branch_length(subs_counts):
    """
    Make a very crude guess of expected number of changes along a branch.
    @param subs_counts: an (nstates, nstates) ndarray of observed substitutions
    @return: crude guess of expected number of changes along the branch
    """
    total_count = np.sum(subs_counts)
    diag_count = np.sum(np.diag(subs_counts))
    crude_estimate = (total_count - diag_count) / float(total_count)
    return crude_estimate

#XXX this is the same as for non-mixture models
def stationary_distn_check_helper(pre_Q, codon_distn, branch_length):
    Q = markovutil.pre_Q_to_Q(pre_Q, codon_distn, branch_length)
    P = scipy.linalg.expm(Q)
    next_distn = np.dot(codon_distn, P)
    if not np.allclose(next_distn, codon_distn):
        raise Exception(next_distn - codon_distn)
    print 'stationary distribution is ok'

def get_two_taxon_neg_ll_encoded_theta(
        model,
        subs_counts,
        log_counts, codon_distn,
        ts, tv, syn, nonsyn, compo, asym_compo,
        encoded_theta,
        ):
    """
    Get the negative log likelihood.
    This function uses the logarithms of the model parameters.
    The first param group is the model implementation.
    The second param group is the data.
    The third param group consists of data summaries.
    The fourth param group consists of design matrices related to genetic code.
    The fifth param group consist of free parameters of the model.
    """
    branch_length = algopy.exp(encoded_theta[0])
    encoded_model_theta = encoded_theta[1:]
    natural_model_theta = model.encoded_to_natural(encoded_model_theta)
    natural_theta = algopy.zeros_like(encoded_theta)
    natural_theta[0] = branch_length
    natural_theta[1:] = natural_model_theta
    return get_two_taxon_neg_ll(
        model,
        subs_counts,
        log_counts, codon_distn,
        ts, tv, syn, nonsyn, compo, asym_compo,
        natural_theta,
        )

def get_two_taxon_neg_ll(
        model,
        subs_counts,
        log_counts, codon_distn,
        ts, tv, syn, nonsyn, compo, asym_compo,
        natural_theta,
        ):
    """
    Get the negative log likelihood.
    This function does not use the logarithms.
    It is mostly for computing the hessian;
    otherwise the version with the logarithms would probably be better.
    The first param group is the model implementation.
    The second param group is the data.
    The third param group consists of data summaries.
    The fourth param group consists of design matrices related to genetic code.
    The fifth param group consist of free parameters of the model.
    """
    branch_length = natural_theta[0]
    natural_model_theta = natural_theta[1:]
    distn = model.get_distn(
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_model_theta,
            )
    probs, pre_Qs = model.get_mixture(
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_model_theta,
            )
    neg_ll = -markovutil.get_branch_mix_ll(
            subs_counts, probs, pre_Qs, distn, branch_length)
    print neg_ll
    return neg_ll



##############################################################################
# Do a little bit of object oriented programming for models.
# These classes should be thin wrappers around the vector of params.


class FMutSel_F:
    """
    A codon model used in Yang-Nielsen 2008.
    """

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 5:
            raise ValueError(len(theta))

    @classmethod
    def natural_to_encoded(cls, natural_theta):
        return algopy.log(natural_theta)

    @classmethod
    def encoded_to_natural(cls, encoded_theta):
        return algopy.exp(encoded_theta)

    @classmethod
    def get_natural_guess(cls):
        natural_theta = np.array([
            3.0, # kappa
            0.1, # omega
            1.0, # pi_A / pi_T
            1.0, # pi_C / pi_T
            1.0, # pi_G / pi_T
            ], dtype=float)
        cls.check_theta(natural_theta)
        return natural_theta

    @classmethod
    def get_distn(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        return codon_distn

    @classmethod
    def get_pre_Q(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        cls.check_theta(natural_theta)
        kappa = natural_theta[0]
        omega = natural_theta[1]
        nt_distn = markovutil.ratios_to_distn(natural_theta[2:5])
        pre_Q = fmutsel.get_pre_Q(
                log_counts,
                fmutsel.genic_fixation,
                ts, tv, syn, nonsyn, compo, asym_compo,
                nt_distn, kappa, omega,
                )
        return pre_Q


class FMutSelG_F:
    """
    A new model.
    This model is related to the model used by Yang and Nielsen in 2008.
    The difference is that this new model has an extra free parameter.
    This extra free parameter controls the recessivity/dominance.
    This model name is new,
    because I have not seen this model described elsewhere.
    Therefore I am giving it a short name so that I can refer to it.
    The name is supposed to be as inconspicuous as possible,
    differing from the standard name of the most closely related
    model in the literature by only one letter.
    This extra letter is the G at the end,
    which is supposed to mean 'generalized.'
    I realize that this is a horrible naming scheme,
    because there are multiple ways that any model can be generalized,
    and this name does not help to distinguish the particular
    way that I have chosen to generalize the model.
    """

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 6:
            raise ValueError(len(theta))

    @classmethod
    def natural_to_encoded(cls, natural_theta):
        encoded_theta = algopy.zeros_like(natural_theta)
        encoded_theta[0] = natural_theta[0]
        encoded_theta[1:] = algopy.log(natural_theta[1:])
        return encoded_theta

    @classmethod
    def encoded_to_natural(cls, encoded_theta):
        natural_theta = algopy.zeros_like(encoded_theta)
        natural_theta[0] = encoded_theta[0]
        natural_theta[1:] = algopy.exp(encoded_theta[1:])
        return natural_theta

    @classmethod
    def get_natural_guess(cls):
        natural_theta = np.array([
            0.0, # kimura d
            3.0, # kappa
            0.1, # omega
            1.0, # pi_A / pi_T
            1.0, # pi_C / pi_T
            1.0, # pi_G / pi_T
            ], dtype=float)
        cls.check_theta(natural_theta)
        return natural_theta

    @classmethod
    def get_distn(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        return codon_distn

    @classmethod
    def get_pre_Q(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        cls.check_theta(natural_theta)
        kimura_d = natural_theta[0]
        kappa = natural_theta[1]
        omega = natural_theta[2]
        nt_distn = markovutil.ratios_to_distn(natural_theta[3:6])
        pre_Q = fmutsel.get_pre_Q_unconstrained(
                log_counts,
                ts, tv, syn, nonsyn, compo, asym_compo,
                kimura_d, nt_distn, kappa, omega,
                )
        return pre_Q


class FMutSel_F_OmegaMix:
    """
    This is a mixture of two FMutSel_F models.
    The distinction between the two mixture components is that
    each component gets its own free omega parameter.
    The omega parameter controls exchangeability ratios of
    synonymous vs. nonsynonymous substitutions.
    Because the stationary distribution does not depend on exchangeability,
    and in particular does not depend on the omega parameter,
    both mixture components share the same codon stationary distribution
    and selection coefficients.
    """

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 5 + 2:
            raise ValueError(len(theta))

    @classmethod
    def natural_to_encoded(cls, natural_theta):
        """
        The first parameter is a proportion.
        """
        encoded = algopy.zeros_like(natural_theta)
        encoded[0] = algopy.special.logit(natural_theta[0])
        encoded[1:] = algopy.log(natural_theta[1:])
        return encoded

    @classmethod
    def encoded_to_natural(cls, encoded_theta):
        """
        The first parameter is a proportion.
        """
        natural = algopy.zeros_like(encoded_theta)
        natural[0] = algopy.special.expit(encoded_theta[0])
        natural[1:] = algopy.exp(encoded_theta[1:])
        return natural

    @classmethod
    def get_natural_guess(cls):
        natural_theta = np.array([
            0.98, # mixing proportion of first component
            0.01, # omega for first component
            2.20,  # omega for second component
            3.60,  # kappa
            1.00,  # pi_A / pi_T
            1.00,  # pi_C / pi_T
            1.00,  # pi_G / pi_T
            ], dtype=float)
        cls.check_theta(natural_theta)
        return natural_theta

    @classmethod
    def get_distn(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        return codon_distn

    @classmethod
    def get_mixture(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        """
        @return: finite_distn, pre_Q_matrices
        """
        cls.check_theta(natural_theta)
        p0 = natural_theta[0]
        p1 = 1 - p0
        first_omega = natural_theta[1]
        second_omega = natural_theta[2]
        kappa = natural_theta[3]
        nt_distn = markovutil.ratios_to_distn(natural_theta[4:4+3])
        first_pre_Q = fmutsel.get_pre_Q(
                log_counts,
                fmutsel.genic_fixation,
                ts, tv, syn, nonsyn, compo, asym_compo,
                nt_distn, kappa, first_omega,
                )
        second_pre_Q = fmutsel.get_pre_Q(
                log_counts,
                fmutsel.genic_fixation,
                ts, tv, syn, nonsyn, compo, asym_compo,
                nt_distn, kappa, second_omega,
                )
        return (p0, p1), (first_pre_Q, second_pre_Q)


class FMutSelG_F_OmegaMix:
    """
    This is a mixture of two FMutSelG models.
    """

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 6 + 2:
            raise ValueError(len(theta))

    @classmethod
    def natural_to_encoded(cls, natural_theta):
        """
        The first parameter is a proportion.
        The fourth parameter is unconstrained.
        """
        encoded = algopy.zeros_like(natural_theta)
        encoded[0] = algopy.special.logit(natural_theta[0])
        encoded[1] = algopy.log(natural_theta[1])
        encoded[2] = algopy.log(natural_theta[2])
        encoded[3] = natural_theta[3]
        encoded[4:] = algopy.log(natural_theta[4:])
        return encoded

    @classmethod
    def encoded_to_natural(cls, encoded_theta):
        """
        The first parameter is a proportion.
        The fourth parameter is unconstrained.
        """
        natural = algopy.zeros_like(encoded_theta)
        natural[0] = algopy.special.expit(encoded_theta[0])
        natural[1] = algopy.exp(encoded_theta[1])
        natural[2] = algopy.exp(encoded_theta[2])
        natural[3] = encoded_theta[3]
        natural[4:] = algopy.exp(encoded_theta[4:])
        return natural

    @classmethod
    def get_natural_guess(cls):
        natural_theta = np.array([
            0.98, # mixing proportion of first component
            0.01, # omega for first component
            2.20,  # omega for second component
            0.00,  # kimura D associated with fitter introduced allele
            3.60,  # kappa
            1.00,  # pi_A / pi_T
            1.00,  # pi_C / pi_T
            1.00,  # pi_G / pi_T
            ], dtype=float)
        cls.check_theta(natural_theta)
        return natural_theta

    @classmethod
    def get_distn(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        return codon_distn

    @classmethod
    def get_mixture(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        cls.check_theta(natural_theta)
        p0 = natural_theta[0]
        p1 = 1 - p0
        first_omega = natural_theta[1]
        second_omega = natural_theta[2]
        kimura_d = natural_theta[3]
        kappa = natural_theta[4]
        nt_distn = markovutil.ratios_to_distn(natural_theta[5:5+3])
        first_pre_Q = fmutsel.get_pre_Q_unconstrained(
                log_counts,
                ts, tv, syn, nonsyn, compo, asym_compo,
                kimura_d, nt_distn, kappa, first_omega,
                )
        second_pre_Q = fmutsel.get_pre_Q_unconstrained(
                log_counts,
                ts, tv, syn, nonsyn, compo, asym_compo,
                kimura_d, nt_distn, kappa, second_omega,
                )
        return (p0, p1), (first_pre_Q, second_pre_Q)



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

    # make a crude guess of the expected number of changes
    log_blen = np.log(guess_branch_length(subs_counts))

    # use the chosen model to construct an initial guess for max likelihood
    model_natural_guess = args.model.get_natural_guess()
    model_nparams = len(model_natural_guess)
    encoded_guess = np.empty(model_nparams + 1, dtype=float)
    encoded_guess[0] = log_blen
    encoded_guess[1:] = args.model.natural_to_encoded(model_natural_guess)

    # construct the neg log likelihood non-free params
    neg_ll_args = (
            args.model,
            subs_counts,
            log_counts, empirical_codon_distn,
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
            encoded_guess,
            method=args.minimization_method,
            jac=g_encoded_theta,
            hess=h_encoded_theta,
            )

    # extract and decode the maximum likelihood estimates
    encoded_xopt = results.x
    mle_log_blen = encoded_xopt[0]
    mle_blen = np.exp(mle_log_blen)
    model_encoded_xopt = encoded_xopt[1:]
    model_xopt = args.model.encoded_to_natural(model_encoded_xopt)
    xopt = np.empty_like(encoded_xopt)
    xopt[0] = mle_blen
    xopt[1:] = model_xopt

    """
    # check that the stationary distribution is ok
    mle_distn = args.model.get_distn(
            log_counts, empirical_codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            model_xopt,
            )
    (p0, p1), (first_pre_Q, second_pre_Q) = args.model.get_mixture(
            log_counts, empirical_codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            model_xopt,
            )
    stationary_distn_check_helper(mle_pre_Q, mle_distn, mle_blen)
    """

    # define functions for computing the hessian
    f = functools.partial(get_two_taxon_neg_ll, *neg_ll_args)
    g = functools.partial(eval_grad, f)
    h = functools.partial(eval_hess, f)

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

    # print the hessian matrix at the max likelihood parameter values
    fisher_info = h(xopt)
    cov = scipy.linalg.inv(fisher_info)
    errors = np.sqrt(np.diag(cov))
    print 'observed fisher information matrix:'
    print fisher_info
    print
    print 'inverse of fisher information matrix:'
    print cov
    print
    print 'standard error estimates (sqrt of diag of inv of fisher info)'
    print errors
    print

    # write the neg log likelihood into a separate file
    if args.neg_log_likelihood_out:
        with open(args.neg_log_likelihood_out, 'w') as fout:
            print >> fout, results.fun

    # write the parameter estimates into a separate file
    if args.parameter_estimates_out:
        with open(args.parameter_estimates_out, 'w') as fout:
            for value in xopt:
                print >> fout, value

    # write the parameter estimates into a separate file
    if args.parameter_errors_out:
        with open(args.parameter_errors_out, 'w') as fout:
            for value in errors:
                print >> fout, value



if __name__ == '__main__':

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)

    # let the user define the model
    model_choice = parser.add_mutually_exclusive_group(required=True)
    model_choice.add_argument(
            '--FMutSel-F-mix',
            dest='model',
            action='store_const',
            const=FMutSel_F_OmegaMix,
            )
    model_choice.add_argument(
            '--FMutSelG-F-mix',
            dest='model',
            action='store_const',
            const=FMutSelG_F_OmegaMix,
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
    parser.add_argument('--neg-log-likelihood-out',
            help='write the minimized neg log likelihood to this file')
    parser.add_argument('--parameter-estimates-out',
            help='write the maximum likelihood parameter estimates here')
    parser.add_argument('--parameter-errors-out',
            help='write the parameter estimate standard errors here')
    args = parser.parse_args()
    main(args)

