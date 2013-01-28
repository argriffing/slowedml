"""
This module is related to the FMutSel model of Yang and Nielsen 2008.

The point of this module is to construct a FMutSel rate matrix
whose expected rate is normalized to one expected change per time unit.
"""

import numpy as np
import algopy
import algopy.special

from slowedml import design, ntmodel


##############################################################################
# These functions are directly related to the log likelihood calculation.

def fixation_h(x):
    """
    This is a fixation h function in the notation of Yang and Nielsen.
    """
    return 1. / algopy.special.hyp1f1(1., 2., -x)

def get_selection_F(log_counts, compo, log_nt_weights):
    """
    The F and S notation is from Yang and Nielsen 2008.
    Note that three of the four log nt weights are free parameters.
    One of the four log weights is zero and the other three
    are free parameters to be estimated jointly in the
    maximimum likelihood search,
    so this function is inside the optimization loop.
    @param log_counts: logs of empirical codon counts
    @param compo: codon composition as defined in the get_compo function
    @param log_nt_weights: un-normalized log mutation process probabilities
    @return: a log selection for each codon, up to an additive constant
    """
    return log_counts - algopy.dot(compo, log_nt_weights)

def get_selection_S(F):
    """
    The F and S notation is from Yang and Nielsen 2008.
    @param F: a selection value for each codon, up to an additive constant
    @return: selection differences F_j - F_i, also known as S_ij
    """
    e = algopy.ones_like(F)
    return algopy.outer(e, F) - algopy.outer(F, e)

def get_pre_Q(
        log_counts,
        h,
        ts, tv, syn, nonsyn, compo, asym_compo,
        log_nt_weights, kappa, omega,
        ):
    """
    Notation is from Yang and Nielsen 2008.
    The first group consists of empirically (non-free) estimated parameters.
    The second group is only the fixation function.
    The third group of args consists of precomputed ndarrays.
    The fourth group depends only on free parameters.
    @param log_counts: logs of empirically counted codons in the data set
    @param h: fixation function
    @param ts: indicator for transition
    @param tv: indicator for transversion
    @param syn: indicator for synonymous codons
    @param nonsyn: indicator for nonsynonymous codons
    @param compo: site independent nucleotide composition per codon
    @param asym_compo: tensor from get_asym_compo function
    @return: rate matrix
    """
    F = get_selection_F(log_counts, compo, log_nt_weights)
    S = get_selection_S(F)
    pre_Q = (kappa * ts + tv) * (omega * nonsyn + syn) * algopy.exp(
            algopy.dot(asym_compo, log_nt_weights)) * h(S)
    return pre_Q


##############################################################################
# The following functions follow the Yang-Nielsen notation more closely.

def get_pre_Q_expanded(
        log_counts,
        h,
        ts, tv, syn, nonsyn, compo, asym_compo,
        nt_distn, kappa, omega,
        ):

    """
    # define the symmetric factor of the hky nucleotide mutation matrix
    nt_hky = ntmodel.ts * kappa + ntmodel.tv

    # define the codon hky mutation process
    codon_hky = algopy.dot(asym_compo, nt_hky * nt_distn)

    # fmutsel actually treats nonsyn/syn as a mutational parameter
    codon_mutation = (nonsyn * omega + syn) * codon_hky

    # compute the fixation 
    F = get_selection_F(log_counts, compo, log_nt_weights)
    S = get_selection_S(F)
    codon_fixation = h(S)

    # return the unscaled rates corresponding to the mutation-selection process
    pre_Q = codon_mutation * codon_fixation
    return pre_Q
    """

    # compute the fixation 
    F = get_selection_F(log_counts, compo, algopy.log(nt_distn))
    S = get_selection_S(F)
    codon_fixation = h(S)

    # compute the mutation and fixation components
    A = (kappa * ts + tv) * (omega * nonsyn + syn)
    B = algopy.dot(asym_compo, nt_distn) * h(S)

    # construct the pre rate matrix
    pre_Q = A * B
    return pre_Q


