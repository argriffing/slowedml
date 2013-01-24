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
import algopy

from slowedml import design, fileutil
from slowedml import fmutsel, codon1994, markovutil
from slowedml.algopyboilerplate import eval_grad, eval_hess



def neutral_h(x):
    """
    This is a fixation h function in the notation of Yang and Nielsen.
    """
    return algopy.ones_like(x)


##############################################################################
# Two taxon F1 x 4 MG.
# This model name uses a notation I saw in Yang-Nielsen 2008.

def get_two_taxon_f1x4_MG_neg_ll(
        subs_counts,
        ts, tv, syn, nonsyn, compo, asym_compo,
        theta,
        ):
    """
    According to the Yang-Nielsen 2008 docs this is nested in FMutSel-F.
    So I am checking this nestedness.
    @param theta: unconstrained vector of free variables
    """

    # break theta into a branch length vs. other parameters
    branch_length = algopy.exp(theta[0])
    kappa = algopy.exp(theta[1])
    omega = algopy.exp(theta[2])
    nt_distn = markovutil.expand_distn(theta[3:])

    if nt_distn.shape != (4,):
        raise Exception(nt_distn.shape)

    # XXX not sure if this is right
    # compute the stationary distribution
    codon_distn = codon1994.get_f1x4_codon_distn(compo, nt_distn)

    # get the rate matrix
    pre_Q = codon1994.get_MG_pre_Q(
            ts, tv, syn, nonsyn, asym_compo,
            kappa, omega, nt_distn)
    neg_ll = -markovutil.get_branch_ll(
            subs_counts, pre_Q, codon_distn, branch_length)
    print neg_ll, theta
    return neg_ll


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
    nt_distn = markovutil.expand_distn(theta[3:])

    # this uses the Goldman-Yang rather than Muse-Gaut approach
    codon_distn = codon1994.get_f1x4_codon_distn(compo, nt_distn)

    pre_Q = codon1994.get_pre_Q(
            ts, tv, syn, nonsyn,
            codon_distn, kappa, omega)
    neg_ll = -markovutil.get_branch_ll(
            subs_counts, pre_Q, codon_distn, branch_length)
    print neg_ll, theta
    return neg_ll


##############################################################################
# Two taxon FMutSel-F.

def get_two_taxon_fmutsel_neg_ll(
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

    # compute the neg log likelihood using the fmutsel model
    pre_Q = fmutsel.get_pre_Q(
            log_counts,
            h,
            ts, tv, syn, nonsyn, compo, asym_compo,
            reduced_theta)
    neg_ll = -markovutil.get_branch_ll(
            subs_counts, pre_Q, v, branch_length)
    print neg_ll, theta
    return neg_ll




##############################################################################
# Try to minimize the neg log likelihood.

def do_FMutSel_F(
        subs_counts, log_counts, v,
        ts, tv, syn, nonsyn, compo, asym_compo,
        ):

    # guess the branch length
    total_count = np.sum(subs_counts)
    diag_count = np.sum(np.diag(subs_counts))
    branch_length_guess = (total_count - diag_count) / float(total_count)
    log_branch_length_guess = np.log(branch_length_guess)

    # guess the values of the free params
    guess = np.array([
        log_branch_length_guess, # log branch length
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
    f = functools.partial(get_two_taxon_fmutsel_neg_ll, *fmin_args)
    g = functools.partial(eval_grad, f)
    h = functools.partial(eval_hess, f)

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

    xopt = results[0]

    # print a thing for debugging
    print 'nt distn ACGT:'
    print markovutil.expand_distn(xopt[-3:])

    return results, xopt


def do_F1x4(
        subs_counts, log_counts, v,
        ts, tv, syn, nonsyn, compo, asym_compo,
        ):

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

    fmin_args = (
        subs_counts,
        ts, tv, syn, nonsyn, compo, asym_compo,
        )

    # define the objective function and the gradient and hessian
    f = functools.partial(get_two_taxon_f1x4_neg_ll, *fmin_args)
    g = functools.partial(eval_grad, f)
    h = functools.partial(eval_hess, f)

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

    xopt = results[0]

    # print a thing for debugging
    kernel = np.exp(xopt[-3:].tolist() + [0])
    print kernel / np.sum(kernel)

    return results, xopt


def do_F1x4MG(
        subs_counts, log_counts, empirical_codon_distn,
        ts, tv, syn, nonsyn, compo, asym_compo,
        ):


    # XXX not sure if this is right
    #v = codon1994.get_f1x4_codon_distn(compo, nt_distn)

    #FIXME: this is mostly copypasted from FMutSel

    # guess the branch length
    total_count = np.sum(subs_counts)
    diag_count = np.sum(np.diag(subs_counts))
    branch_length_guess = (total_count - diag_count) / float(total_count)
    log_branch_length_guess = np.log(branch_length_guess)

    # guess the values of the free params
    guess = np.array([
        log_branch_length_guess, # log branch length
        1,  # log kappa
        -3, # log omega
        0,  # log (pi_A / pi_T)
        0,  # log (pi_C / pi_T)
        0,  # log (pi_G / pi_T)
        ], dtype=float)

    # construct the neg log likelihood non-free params
    fmin_args = (
            subs_counts,
            ts, tv, syn, nonsyn, compo, asym_compo,
            )

    # define the objective function and the gradient and hessian
    f = functools.partial(get_two_taxon_f1x4_MG_neg_ll, *fmin_args)
    g = functools.partial(eval_grad, f)
    h = functools.partial(eval_hess, f)

    # do the search, using information about the gradient and hessian
    """
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
    xopt = results[0]
    """
    results = scipy.optimize.fmin_bfgs(
            f,
            guess,
            g,
            )
    xopt = results

    # print a thing for debugging
    print 'nt distn ACGT:'
    print markovutil.expand_distn(xopt[-3:])

    return results, xopt


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

    # estimate the parameters of the model using log likelihood
    results, xopt = args.model(
        subs_counts, log_counts, v,
        ts, tv, syn, nonsyn, compo, asym_compo,
        )

    # report a summary of the maximum likelihood search
    with fileutil.open_or_stdout(args.o, 'w') as fout:
        print >> fout, 'raw results from the minimization:'
        print >> fout, results
        print >> fout
        print >> fout, 'max log likelihood params:'
        print >> fout, xopt
        print >> fout
        print >> fout, 'exp of max log likelihood params:'
        print >> fout, np.exp(xopt)



if __name__ == '__main__':

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)

    # let the user define the model
    model_choice = parser.add_mutually_exclusive_group(required=True)
    model_choice.add_argument(
            '--FMutSel-F',
            dest='model',
            action='store_const',
            const=do_FMutSel_F,
            )
    model_choice.add_argument(
            '--F1x4',
            dest='model',
            action='store_const',
            const=do_F1x4,
            )
    model_choice.add_argument(
            '--F1x4MG',
            dest='model',
            action='store_const',
            const=do_F1x4MG,
            )

    parser.add_argument('--count-matrix', required=True,
            help='matrix of codon state change counts from human to chimp')
    parser.add_argument('--code', required=True,
            help='path to the human mitochondrial genetic code')
    parser.add_argument('-o', default='-',
            help='max log likelihood (default is stdout)')

    main(parser.parse_args())

