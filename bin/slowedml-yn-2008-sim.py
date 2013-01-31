#!/usr/bin/env python

"""
Use max likelihood estimation on a pair of sequences.

The model names are from Table (1) of Nielsen-Yang 2008.
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
from slowedml import moretypes
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

def get_two_taxon_neg_ll_log_theta(
        model,
        subs_counts,
        log_counts, codon_distn,
        ts, tv, syn, nonsyn, compo, asym_compo,
        theta,
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
    branch_length = algopy.exp(theta[0])
    model_theta = theta[1:]
    distn = model.get_distn(
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            model_theta,
            )
    pre_Q = model.get_pre_Q(
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            model_theta,
            )
    neg_ll = -markovutil.get_branch_ll(
            subs_counts, pre_Q, distn, branch_length)
    return neg_ll

def get_two_taxon_neg_ll(
        model,
        subs_counts,
        log_counts, codon_distn,
        ts, tv, syn, nonsyn, compo, asym_compo,
        theta,
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
    branch_length = theta[0]
    model_theta = theta[1:]
    model_log_theta = algopy.log(model_theta)
    distn = model.get_distn(
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            model_log_theta,
            )
    pre_Q = model.get_pre_Q(
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            model_log_theta,
            )
    neg_ll = -markovutil.get_branch_ll(
            subs_counts, pre_Q, distn, branch_length)
    return neg_ll



##############################################################################
# Do a little bit of object oriented programming for models.
# These classes should be thin wrappers around the vector of params.


class F1x4:
    """
    Goldman-Yang 1994 codon model.
    """

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 5:
            raise ValueError(len(theta))

    @classmethod
    def get_guess(cls):
        theta = np.array([
            1,  # log kappa
            -3, # log omega
            0,  # log (pi_A / pi_T)
            0,  # log (pi_C / pi_T)
            0,  # log (pi_G / pi_T)
            ], dtype=float)
        cls.check_theta(theta)
        return theta

    @classmethod
    def get_distn(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            theta,
            ):
        cls.check_theta(theta)
        nt_distn = markovutil.log_ratios_to_distn(theta[2:5])
        codon_distn = codon1994.get_f1x4_codon_distn(compo, nt_distn)
        return codon_distn

    @classmethod
    def get_pre_Q(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            theta,
            ):
        cls.check_theta(theta)
        kappa = algopy.exp(theta[0])
        omega = algopy.exp(theta[1])
        nt_distn = markovutil.log_ratios_to_distn(theta[2:5])
        codon_distn = codon1994.get_f1x4_codon_distn(compo, nt_distn)
        pre_Q = codon1994.get_pre_Q(
                ts, tv, syn, nonsyn,
                codon_distn, kappa, omega)
        return pre_Q


class F1x4MG:
    """
    Muse-Gaut 1994 codon model.
    """

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 5:
            raise ValueError(len(theta))

    @classmethod
    def get_guess(cls):
        theta = np.array([
            1,  # log kappa
            -3, # log omega
            0,  # log (pi_A / pi_T)
            0,  # log (pi_C / pi_T)
            0,  # log (pi_G / pi_T)
            ], dtype=float)
        cls.check_theta(theta)
        return theta

    @classmethod
    def get_distn(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            theta,
            ):
        cls.check_theta(theta)
        nt_distn = markovutil.log_ratios_to_distn(theta[2:5])
        codon_distn = codon1994.get_f1x4_codon_distn(compo, nt_distn)
        return codon_distn

    @classmethod
    def get_pre_Q(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            theta,
            ):
        cls.check_theta(theta)
        kappa = algopy.exp(theta[0])
        omega = algopy.exp(theta[1])
        nt_distn = markovutil.log_ratios_to_distn(theta[2:5])
        pre_Q = codon1994.get_MG_pre_Q(
                ts, tv, syn, nonsyn, asym_compo,
                nt_distn, kappa, omega)
        return pre_Q


class FMutSel_F:
    """
    A codon model used in Yang-Nielsen 2008.
    """

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 5:
            raise ValueError(len(theta))

    @classmethod
    def get_guess(cls):
        theta = np.array([
            1,  # log kappa
            -3, # log omega
            0,  # log (pi_A / pi_T)
            0,  # log (pi_C / pi_T)
            0,  # log (pi_G / pi_T)
            ], dtype=float)
        cls.check_theta(theta)
        return theta

    @classmethod
    def get_distn(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            theta,
            ):
        return codon_distn

    @classmethod
    def get_pre_Q(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            theta,
            ):
        cls.check_theta(theta)
        kappa = algopy.exp(theta[0])
        omega = algopy.exp(theta[1])
        nt_distn = markovutil.log_ratios_to_distn(theta[2:5])
        pre_Q = fmutsel.get_pre_Q(
                log_counts,
                fmutsel.genic_fixation,
                ts, tv, syn, nonsyn, compo, asym_compo,
                nt_distn, kappa, omega,
                )
        return pre_Q


class FMutSelPD_F:
    """
    A new model for which preferred alleles are purely dominant.
    """

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 5:
            raise ValueError(len(theta))

    @classmethod
    def get_guess(cls):
        theta = np.array([
            1,  # log kappa
            -3, # log omega
            0,  # log (pi_A / pi_T)
            0,  # log (pi_C / pi_T)
            0,  # log (pi_G / pi_T)
            ], dtype=float)
        cls.check_theta(theta)
        return theta

    @classmethod
    def get_distn(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            theta,
            ):
        return codon_distn

    @classmethod
    def get_pre_Q(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            theta,
            ):
        cls.check_theta(theta)
        kappa = algopy.exp(theta[0])
        omega = algopy.exp(theta[1])
        nt_distn = markovutil.log_ratios_to_distn(theta[2:5])
        pre_Q = fmutsel.get_pre_Q(
                log_counts,
                fmutsel.preferred_dominant_fixation,
                ts, tv, syn, nonsyn, compo, asym_compo,
                nt_distn, kappa, omega,
                )
        return pre_Q


class FMutSelPR_F:
    """
    A new model for which preferred alleles are purely recessive.
    """

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 5:
            raise ValueError(len(theta))

    @classmethod
    def get_guess(cls):
        theta = np.array([
            1,  # log kappa
            -3, # log omega
            0,  # log (pi_A / pi_T)
            0,  # log (pi_C / pi_T)
            0,  # log (pi_G / pi_T)
            ], dtype=float)
        cls.check_theta(theta)
        return theta

    @classmethod
    def get_distn(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            theta,
            ):
        return codon_distn

    @classmethod
    def get_pre_Q(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            theta,
            ):
        cls.check_theta(theta)
        kappa = algopy.exp(theta[0])
        omega = algopy.exp(theta[1])
        nt_distn = markovutil.log_ratios_to_distn(theta[2:5])
        pre_Q = fmutsel.get_pre_Q(
                log_counts,
                fmutsel.preferred_recessive_fixation,
                ts, tv, syn, nonsyn, compo, asym_compo,
                nt_distn, kappa, omega,
                )
        return pre_Q

g_model_name_to_class = {
        'FMutSel-F' : FMutSel_F,
        'FMutSelPD-F' : FMutSelPD_F,
        'FMutSelPR-F' : FMutSelPR_F,
        }


def get_min_neg_log_likelihood(
        model,
        subs_counts,
        ts, tv, syn, nonsyn, compo, asym_compo,
        minimization_method,
        ):

    # compute some summaries of the observed codon substitutions
    counts = np.sum(subs_counts, axis=0) + np.sum(subs_counts, axis=1)
    log_counts = np.log(counts)
    empirical_codon_distn = counts / float(np.sum(counts))

    # make a crude guess of the expected number of changes
    log_blen = np.log(guess_branch_length(subs_counts))

    # use the chosen model to construct an initial guess for max likelihood
    guess = np.array([log_blen] + model.get_guess().tolist(), dtype=float)

    # construct the neg log likelihood non-free params
    neg_ll_args = (
            model,
            subs_counts,
            log_counts, empirical_codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            )

    # define the objective function and the gradient and hessian
    f_log_theta = functools.partial(
            get_two_taxon_neg_ll_log_theta, *neg_ll_args)
    g_log_theta = functools.partial(eval_grad, f_log_theta)
    h_log_theta = functools.partial(eval_hess, f_log_theta)

    # do the search, using information about the gradient and hessian
    results = scipy.optimize.minimize(
            f_log_theta,
            guess,
            method=args.minimization_method,
            jac=g_log_theta,
            hess=h_log_theta,
            )

    # return the min neg log likelihood
    return results.fun


def main(args):

    # read the description of the genetic code
    with open(args.code) as fin_gcode:
        arr = list(csv.reader(fin_gcode, delimiter='\t'))
        indices, aminos, codons = zip(*arr)
        if [int(x) for x in indices] != range(len(indices)):
            raise ValueError

    # look for stop codons
    aminos = [x.lower() for x in aminos]
    nstop = aminos.count('stop')
    if nstop not in (2, 3, 4):
        raise Exception('expected 2 or 3 or 4 stop codons')
    if any(x == 'stop' for x in aminos[:-nstop]):
        raise Exception('expected stop codons at the end of the genetic code')

    # trim the stop codons
    aminos = aminos[:-nstop]
    codons = codons[:-nstop]
    ncodons = len(codons)

    # precompute some numpy ndarrays using the genetic code
    ts = design.get_nt_transitions(codons)
    tv = design.get_nt_transversions(codons)
    syn = design.get_syn(codons, aminos)
    nonsyn = design.get_nonsyn(codons, aminos)
    compo = design.get_compo(codons)
    nt_sinks = design.get_nt_sinks(codons)
    asym_compo = np.transpose(nt_sinks, (1, 2, 0))

    # read the empirical counts and detect and trim stop codon counts
    empirical_codon_counts = np.loadtxt(args.empirical_codon_counts)
    if len(empirical_codon_counts) < ncodons:
        raise Exception
    if any(empirical_codon_counts[ncodons:]):
        raise Exception
    empirical_codon_counts = empirical_codon_counts[:ncodons]
    log_empirical_codon_counts = np.log(empirical_codon_counts)
    empirical_codon_distn = empirical_codon_counts / float(
            np.sum(empirical_codon_counts))

    # Precompute the distribution over codon substitutions
    # using the stationary distribution and transition matrix
    # of the simulation model given the simulation parameters.
    sim_theta = np.loadtxt(args.simulation_parameter_values)
    log_sim_theta = np.log(sim_theta)
    log_sim_blen = log_sim_theta[0]
    log_sim_theta_model = log_sim_theta[1:]
    sim_model = g_model_name_to_class[args.simulation_model]
    pre_Q = sim_model.get_pre_Q(
            log_empirical_codon_counts,
            empirical_codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            log_sim_theta_model,
            )
    stationary_distn = sim_model.get_distn(
            log_empirical_codon_counts,
            empirical_codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            log_sim_theta_model,
            )
    # get the rate matrix
    Q = markovutil.pre_Q_to_Q(pre_Q, stationary_distn, np.exp(log_sim_blen))
    # get the conditional transition matrix
    P = scipy.linalg.expm(Q)
    # get the joint substitution probability matrix
    J = (P.T * stationary_distn).T

    # use an appropriate sample size
    if args.sample_size:
        sample_size = args.sample_size
    else:
        # If the sample size is unspecified,
        # use a sample size whose number of codon counts
        # matches the sum of the user-provided empirical codon counts.
        sample_size = int(np.sum(empirical_codon_counts) / 2)

    with open(args.table_out, 'w') as fout:

        # write the header of the R table file
        header_row = (
                args.first_inference_model.replace('-', '.'),
                args.second_inference_model.replace('-', '.'),
                )
        print >> fout, '\t'.join(header_row)

        for sample_index in range(args.nsamples):

            # sample an (ncodons, ncodons) ndarray of counts
            # using the precomputed multinomial distribution
            subs_counts = np.random.multinomial(
                    sample_size,
                    J.flat,
                    ).reshape(J.shape).astype(float)

            # compute the neg log likelihoods for the two models
            neg_log_likelihoods = []
            for model_name in (
                    args.first_inference_model,
                    args.second_inference_model,
                    ):
                model = g_model_name_to_class[model_name]
                min_neg_ll = get_min_neg_log_likelihood(
                        model,
                        subs_counts,
                        ts, tv, syn, nonsyn, compo, asym_compo,
                        args.minimization_method,
                        )
                neg_log_likelihoods.append(min_neg_ll)

            # write the data row
            data_row = (
                    sample_index+1,
                    -neg_log_likelihoods[0],
                    -neg_log_likelihoods[1],
                    )
            print >> fout, '\t'.join(str(x) for x in data_row)



    # read the (nstates, nstates) array of observed codon substitutions
    #subs_counts = np.loadtxt(args.count_matrix, dtype=float)

    # trim the stop codons
    #subs_counts = subs_counts[:-nstop, :-nstop]

    # compute some summaries of the observed codon substitutions
    #counts = np.sum(subs_counts, axis=0) + np.sum(subs_counts, axis=1)
    #log_counts = np.log(counts)
    #empirical_codon_distn = counts / float(np.sum(counts))

    # make a crude guess of the expected number of changes
    #log_blen = np.log(guess_branch_length(subs_counts))

    # use the chosen model to construct an initial guess for max likelihood
    #guess = np.array([log_blen] + args.model.get_guess().tolist(), dtype=float)

    # construct the neg log likelihood non-free params
    #neg_ll_args = (
            #args.model,
            #subs_counts,
            #log_counts, empirical_codon_distn,
            #ts, tv, syn, nonsyn, compo, asym_compo,
            #)

    # define the objective function and the gradient and hessian
    #f_log_theta = functools.partial(
            #get_two_taxon_neg_ll_log_theta, *neg_ll_args)
    #g_log_theta = functools.partial(eval_grad, f_log_theta)
    #h_log_theta = functools.partial(eval_hess, f_log_theta)

    # do the search, using information about the gradient and hessian
    #results = scipy.optimize.minimize(
            #f_log_theta,
            #guess,
            #method=args.minimization_method,
            #jac=g_log_theta,
            #hess=h_log_theta,
            #)

    #log_xopt = results.x
    #xopt = np.exp(log_xopt)

    ## check that the stationary distribution is ok
    #mle_blen = algopy.exp(xopt[0])
    #mle_distn = args.model.get_distn(
            #log_counts, empirical_codon_distn,
            #ts, tv, syn, nonsyn, compo, asym_compo,
            #log_xopt[1:],
            #)
    #mle_pre_Q = args.model.get_pre_Q(
            #log_counts, empirical_codon_distn,
            #ts, tv, syn, nonsyn, compo, asym_compo,
            #log_xopt[1:],
            #)
    #stationary_distn_check_helper(mle_pre_Q, mle_distn, mle_blen)

    ## define functions for computing the hessian
    #f = functools.partial(get_two_taxon_neg_ll, *neg_ll_args)
    #g = functools.partial(eval_grad, f)
    #h = functools.partial(eval_hess, f)

    # report a summary of the maximum likelihood search
    #print 'raw results from the minimization:'
    #print results
    #print
    #print 'max log likelihood estimates of log params:'
    #print log_xopt
    #print
    #print 'max log likelihood estimates of params:'
    #print xopt
    #print

    # print a thing for debugging
    #print 'nt distn ACGT:'
    #print markovutil.ratios_to_distn(xopt[-3:])
    #print

    # print the hessian matrix at the max likelihood parameter values
    #fisher_info = h(xopt)
    #cov = scipy.linalg.inv(fisher_info)
    #errors = np.sqrt(np.diag(cov))
    #print 'observed fisher information matrix:'
    #print fisher_info
    #print
    #print 'inverse of fisher information matrix:'
    #print cov
    #print
    #print 'standard error estimates (sqrt of diag of inv of fisher info)'
    #print errors
    #print

    # write the neg log likelihood into a separate file
    #if args.neg_log_likelihood_out:
        #with open(args.neg_log_likelihood_out, 'w') as fout:
            #print >> fout, results.fun

    # write the parameter estimates into a separate file
    #if args.parameter_estimates_out:
        #with open(args.parameter_estimates_out, 'w') as fout:
            #for value in xopt:
                #print >> fout, value

    # write the parameter estimates into a separate file
    #if args.parameter_errors_out:
        #with open(args.parameter_errors_out, 'w') as fout:
            #for value in errors:
                #print >> fout, value



if __name__ == '__main__':

    solver_choices = (
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

    model_choices = tuple(g_model_name_to_class)


    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--empirical-codon-counts', required=True,
            help='a vector of empirical codon counts')
    parser.add_argument('--simulation-parameter-values', required=True,
            help='a vector of simulation parameter values')
    parser.add_argument('--simulation-model',
            choices=model_choices,
            default='FMutSel-F',
            help='specify the model to be used for simulation')
    parser.add_argument('--first-inference-model',
            choices=model_choices,
            default='FMutSel-F',
            help='specify the first model to be used for inference')
    parser.add_argument('--second-inference-model',
            choices=model_choices,
            default='FMutSel-F',
            help='specify the second model to be used for inference')
    parser.add_argument('--code', required=True,
            help='path to the genetic code definition')
    parser.add_argument('--minimization-method',
            choices=solver_choices,
            default='BFGS',
            help='use this scipy.optimize.minimize method')
    parser.add_argument('--sample-size', type=moretypes.pos_int,
            help='sample size (e.g. number of codon sites)')
    parser.add_argument('--nsamples', type=moretypes.pos_int, required=True,
            help='get likelihood ratios for this many multinomial samples')
    parser.add_argument('--table-out', required=True,
            help='write the simulation output R table here')
    args = parser.parse_args()
    main(args)

