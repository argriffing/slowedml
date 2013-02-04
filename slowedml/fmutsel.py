"""
This module is related to the FMutSel model of Yang and Nielsen 2008.

The point of this module is to construct a pre-rate matrix
for which the scaling is undefined and the diagonal entries are undefined.
Ratios between off-diagonal rates should be correct.
"""

import numpy as np
import algopy
import algopy.special

from slowedml import design, ntmodel
import pykimuracore


##############################################################################
# These functions are directly related to the log likelihood calculation.
# These are fixation h functions in the notation of Yang and Nielsen.

def genic_fixation(x):
    """
    This fixation function corresponds to pure additivity with no dominance.
    """
    return 1. / algopy.special.hyp1f1(1., 2., -x)

def preferred_dominant_fixation(x):
    """
    Preferred alleles are purely dominant.
    """
    a = algopy.exp(algopy.special.botched_clip(0, np.inf, x))
    b = algopy.special.hyp1f1(0.5, 1.5, abs(x))
    return a / b

def preferred_recessive_fixation(x):
    """
    Preferred alleles are purely recessive.
    """
    a = algopy.exp(algopy.special.botched_clip(-np.inf, 0, x))
    b = algopy.special.hyp1f1(0.5, 1.5, -abs(x))
    return a / b

def unconstrained_recessivity_fixation(kimura_d, S):
    """
    Compute the fixation rates with unconstrained recessivity.
    This function is not compatible with algopy.
    The dominance parameter d uses the notation of Kimura 1957.
    @param kimura_d: an unconstrained dominance vs. recessivity parameter
    @param S: an ndarray of selection differences
    @return: an ndarray of fixation rates
    """
    if S.ndim != 2:
        raise Exception(S.ndim)
    if S.shape[0] != S.shape[1]:
        raise Exception(S.shape)
    nstates = S.shape[0]
    mask = np.ones((nstates, nstates), dtype=int)
    D = mask * kimura_d
    out = np.empty((nstates, nstates), dtype=float)
    if S.ndim != 2:
        raise Exception(S.shape)
    if D.ndim != 2:
        raise Exception(D.shape)
    if mask.ndim != 2:
        raise Exception(mask.shape)
    if out.ndim != 2:
        raise Exception(out.shape)
    pykimuracore.kimura_integral_2d_masked_inplace(S, D, mask, out)
    return out


##############################################################################
# These functions are also directly related to the log likelihood calculation.

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


##############################################################################
# The following functions follow the Yang-Nielsen notation more closely.

def get_pre_Q(
        log_counts,
        h,
        ts, tv, syn, nonsyn, compo, asym_compo,
        nt_distn, kappa, omega,
        ):

    # compute the selection differences
    F = get_selection_F(log_counts, compo, algopy.log(nt_distn))
    S = get_selection_S(F)

    # compute the term that corresponds to conditional fixation rate of codons
    codon_fixation = h(S)

    # compute the mutation and fixation components
    A = (kappa * ts + tv) * (omega * nonsyn + syn)
    B = algopy.dot(asym_compo, nt_distn) * codon_fixation

    # construct the pre rate matrix
    pre_Q = A * B
    return pre_Q


def get_pre_Q_unconstrained(
        log_counts,
        ts, tv, syn, nonsyn, compo, asym_compo,
        kimura_d, nt_distn, kappa, omega,
        ):
    """
    The inputs are divided into groups.
    The first group is an empirical summary of the data.
    The second group has design matrices computed from the genetic code.
    The third group consists of free parameters of the model.
    """

    # compute the selection differences
    F = get_selection_F(log_counts, compo, algopy.log(nt_distn))
    S = get_selection_S(F)

    # compute the term that corresponds to conditional fixation rate of codons
    codon_fixation = unconstrained_recessivity_fixation(kimura_d, S)

    # compute the mutation and fixation components
    A = (kappa * ts + tv) * (omega * nonsyn + syn)
    B = algopy.dot(asym_compo, nt_distn) * codon_fixation

    # construct the pre rate matrix
    pre_Q = A * B
    return pre_Q

