#!/usr/bin/env python

"""
For the linspace-prefixed argument names, see more details at
http://docs.scipy.org/doc/numpy/reference/generated/numpy.linspace.html
For more information about the models and about the max likelihood searches,
see the earlier project called yn-2008-nuclear.
"""

import functools
import argparse
import csv

import numpy as np
import scipy.optimize
import scipy.linalg
import algopy
import algopy.special

from slowedml import design, fileutil, moretypes
from slowedml import fmutsel, codon1994, markovutil
from slowedml.algopyboilerplate import eval_grad, eval_hess


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
    pre_Q = model.get_pre_Q(
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_model_theta,
            )
    neg_ll = -markovutil.get_branch_ll(
            subs_counts, pre_Q, distn, branch_length)
    print neg_ll
    return neg_ll



##############################################################################
# Do a little bit of object oriented programming for models.
# These classes should be thin wrappers around the vector of params.



class FMutSelG_F_partial:
    """
    This model uses an arbitrary predetermined parameter.
    This parameter quantifies the dominance of the preferred allele,
    using Kimura's D notation.
    """

    def __init__(self, kimura_d):
        self.kimura_d = kimura_d

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
            #0.0, # kimura d
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

    def get_pre_Q(self,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        self.check_theta(natural_theta)
        kimura_d = self.kimura_d
        kappa = natural_theta[0]
        omega = natural_theta[1]
        nt_distn = markovutil.ratios_to_distn(natural_theta[2:5])
        pre_Q = fmutsel.get_pre_Q_unconstrained(
                log_counts,
                ts, tv, syn, nonsyn, compo, asym_compo,
                kimura_d, nt_distn, kappa, omega,
                )
        return pre_Q


class FMutSelG_F:
    """
    This model uses a free parameter for dominance of preferred allele.
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


def get_min_neg_ll_and_slope(
        model,
        subs_counts,
        ts, tv, syn, nonsyn, compo, asym_compo,
        minimization_method,
        ):
    """
    return: the min_ll and its derivative with respect to kimura_d
    """

    # compute some summaries of the observed codon substitutions
    counts = np.sum(subs_counts, axis=0) + np.sum(subs_counts, axis=1)
    log_counts = np.log(counts)
    empirical_codon_distn = counts / float(np.sum(counts))

    # make a crude guess of the expected number of changes
    log_blen = np.log(guess_branch_length(subs_counts))

    # use the chosen model to construct an initial guess for max likelihood
    model_natural_guess = model.get_natural_guess()
    model_nparams = len(model_natural_guess)
    encoded_guess = np.empty(model_nparams + 1, dtype=float)
    encoded_guess[0] = log_blen
    encoded_guess[1:] = model.natural_to_encoded(model_natural_guess)

    # construct the neg log likelihood non-free params
    neg_ll_args = (
            model,
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
            method=minimization_method,
            jac=g_encoded_theta,
            hess=h_encoded_theta,
            )

    # compute the min neg log likelihood
    min_ll = results.fun

    # Compute the derivative of the neg log likelihood
    # with respect to the Kimura D parameter.
    # Begin by constructing the encoded parameter vector of the full model.
    enc_opt = results.x
    enc_full = np.zeros(len(enc_opt) + 1)
    enc_full[0] = enc_opt[0]
    enc_full[1] = model.kimura_d
    enc_full[2:] = enc_opt[1:]

    # Next transform the full encoded parameter vector into
    # its natural parameterization.
    full_model = FMutSelG_F
    nat_full = full_model.encoded_to_natural(enc_full)

    # args for gradient
    args_for_gradient = (
            full_model,
            subs_counts,
            log_counts, empirical_codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            )

    # define functions for computing the gradient
    f = functools.partial(get_two_taxon_neg_ll, *args_for_gradient)
    g = functools.partial(eval_grad, f)

    # compute the gradient
    mle_gradient = g(nat_full)

    # we care about the second entry of the gradient vector
    min_ll_slope = mle_gradient[1]

    # return the min_ll and its derivative with respect to kimura_d
    return min_ll, min_ll_slope


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

    # do the constrained log likelihood maximizations
    min_lls = []
    min_ll_slopes = []
    space = np.linspace(
            args.linspace_start,
            args.linspace_stop,
            num=args.linspace_num,
            )
    for kimura_d in space:

        # define the model
        model = FMutSelG_F_partial(kimura_d)

        # compute the constrained min negative log likelihood
        min_ll, min_ll_slope = get_min_neg_ll_and_slope(
                model,
                subs_counts,
                ts, tv, syn, nonsyn, compo, asym_compo,
                args.minimization_method,
                )

        # add the min log likelihood to the list
        min_lls.append(min_ll)
        min_ll_slopes.append(min_ll_slope)

    # write the R table
    with open(args.table_out, 'w') as fout:

        # write the R header
        print >> fout, '\t'.join((
            'Kimura.D',
            'min.neg.ll',
            'min.neg.ll.slope',
            ))

        # write each row of the R table,
        # where each row has
        # position, kimura_d, min_ll
        for i, v in enumerate(zip(space, min_lls, min_ll_slopes)):
            row = [i+1] + list(v)
            print >> fout, '\t'.join(str(x) for x in row)


if __name__ == '__main__':

    # these are the solver names hardcoded into scipy
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

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--count-matrix', required=True,
            help='matrix of codon state change counts on the branch')
    parser.add_argument('--code', required=True,
            help='path to the genetic code definition')
    parser.add_argument('--minimization-method',
            choices=solver_names,
            default='BFGS',
            help='use this scipy.optimize.minimize method')
    parser.add_argument('--table-out',
            required=True,
            help='write an R table here')
    parser.add_argument('--linspace-start', type=float,
            required=True,
            help='smallest value of the parameter D')
    parser.add_argument('--linspace-stop', type=float,
            required=True,
            help='biggest value of the parameter D')
    parser.add_argument('--linspace-num', type=moretypes.pos_int,
            required=True,
            help='check this many values of the parameter D')
    args = parser.parse_args()
    main(args)

