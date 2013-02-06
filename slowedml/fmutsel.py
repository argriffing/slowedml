"""
This module is related to the FMutSel model of Yang and Nielsen 2008.

The point of this module is to construct a pre-rate matrix
for which the scaling is undefined and the diagonal entries are undefined.
Ratios between off-diagonal rates should be correct.
"""

import numpy as np
import scipy.special
import algopy
import algopy.special
import warnings

from slowedml import design, ntmodel
import pykimuracore


#xdummy, wdummy = scipy.special.orthogonal.p_roots(101)
#print 'numpy version:', np.__version__
#print 'scipy version:', scipy.__version__
#print 'xdummy.dtype:', xdummy.dtype
#print
#raise Exception

##############################################################################
# We are giving up and computing integrals by fixed-order gaussian quadrature.

def _precompute_quadrature(a, b, npoints):
    """
    This is for Gaussian quadrature.
    This function exists because computers are not as fast as we would like.
    Two arrays are returned.
    For the recessivity kimura integral,
    the arguments should be something like a=0, b=1, npoints=101.
    @param a: definite integral lower bound
    @param b: definite integral upper bound
    @param npoints: during quadrature evaluate the function at this many points
    @return: roots, weights
    """
    x_raw, w_raw = scipy.special.orthogonal.p_roots(npoints)
    #FIXME: p_roots gives complex numbers in a very annoying way.
    #FIXME: The thing that is so annoying
    #FIXME: is that I cannot seem to reproduce it in another project.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', np.ComplexWarning)
        x_raw = x_raw.astype(np.float64)
        w_raw = w_raw.astype(np.float64)
    c = (b - a) / 2.
    x = c * (x_raw + 1) + a
    w = c * w_raw
    return x, w

# Precompute some ndarrays for quadrature.
g_quad_x, g_quad_w = _precompute_quadrature(0.0, 1.0, 101)


##############################################################################
# These functions are directly related to the log likelihood calculation.
# These are fixation h functions in the notation of Yang and Nielsen.

def genic_fixation(S):
    """
    This fixation function corresponds to pure additivity with no dominance.
    """
    return 1. / algopy.special.hyp1f1(1., 2., -S)

def preferred_dominant_fixation(S):
    """
    Preferred alleles are purely dominant.
    """
    a = algopy.exp(algopy.special.botched_clip(0, np.inf, S))
    b = algopy.special.hyp1f1(0.5, 1.5, abs(S))
    return a / b

def preferred_recessive_fixation(S):
    """
    Preferred alleles are purely recessive.
    """
    a = algopy.exp(algopy.special.botched_clip(-np.inf, 0, S))
    b = algopy.special.hyp1f1(0.5, 1.5, -abs(S))
    return a / b

def fast_unconstrained_recessivity_fixation(kimura_d, S):
    """
    Compute the fixation rates with unconstrained recessivity.
    This function is not compatible with algopy.
    The dominance parameter d uses the notation of Kimura 1957.
    @param kimura_d: an unconstrained dominance vs. recessivity parameter
    @param S: an ndarray of selection differences
    @return: an ndarray of fixation rates
    """
    nstates = S.shape[0]
    mask = np.ones((nstates, nstates), dtype=int)
    D = np.sign(S) * kimura_d
    out = np.empty((nstates, nstates), dtype=float)
    pykimuracore.kimura_integral_2d_masked_inplace(0.5 * S, D, mask, out)
    return 1.0 / out

def algopy_unconstrained_recessivity_fixation(
        kimura_d,
        S,
        ):
    """
    This is only compatible with algopy and is not compatible with numpy.
    It takes ridiculous measures to compute higher order derivatives.
    @param adjacency: a binary design matrix to reduce unnecessary computation
    @param kimura_d: a parameter that might carry Taylor information
    @param S: an ndarray of selection differences with Taylor information
    return: an ndarray of fixation probabilities with Taylor information
    """
    nstates = S.shape[0]
    D = algopy.sign(S) * kimura_d
    H = algopy.zeros_like(S)
    ncoeffs = S.data.shape[0]
    shp = (ncoeffs, -1)
    S_data_reshaped = S.data.reshape(shp)
    D_data_reshaped = D.data.reshape(shp)
    H_data_reshaped = H.data.reshape(shp)
    tmp_a = algopy.zeros_like(H)
    tmp_b = algopy.zeros_like(H)
    tmp_c = algopy.zeros_like(H)
    tmp_a_data_reshaped = tmp_a.data.reshape(shp)
    tmp_b_data_reshaped = tmp_b.data.reshape(shp)
    tmp_c_data_reshaped = tmp_c.data.reshape(shp)
    pykimuracore.kimura_algopy(
            g_quad_x,
            g_quad_w,
            S_data_reshaped,
            D_data_reshaped,
            tmp_a_data_reshaped,
            tmp_b_data_reshaped,
            tmp_c_data_reshaped,
            H_data_reshaped,
            )
    return H


def unconstrained_recessivity_fixation(
        adjacency,
        kimura_d,
        S,
        ):
    """
    This should be compatible with algopy.
    But it may be very slow.
    @param adjacency: a binary design matrix to reduce unnecessary computation
    @param kimura_d: a parameter that might carry Taylor information
    @param S: an ndarray of selection differences with Taylor information
    return: an ndarray of fixation probabilities with Taylor information
    """
    x = g_quad_x
    w = g_quad_w
    nstates = S.shape[0]
    D = algopy.sign(S) * kimura_d
    H = algopy.zeros_like(S)
    for i in range(nstates):
        for j in range(nstates):
            if not adjacency[i, j]:
                continue
            tmp_a = - S[i, j] * x
            tmp_b = algopy.exp(tmp_a * (D[i, j] * (1-x) + 1))
            tmp_c = algopy.dot(tmp_b, w)
            H[i, j] = algopy.reciprocal(tmp_c)
    return H

#XXX this is much slower
def unrolled_unconstrained_recessivity_fixation(
        adjacency,
        kimura_d,
        S,
        ):
    """
    This should be compatible with algopy.
    But it may be very slow.
    The unrolling is with respect to a dot product.
    @param adjacency: a binary design matrix to reduce unnecessary computation
    @param kimura_d: a parameter that might carry Taylor information
    @param S: an ndarray of selection differences with Taylor information
    return: an ndarray of fixation probabilities with Taylor information
    """
    nknots = len(g_quad_x)
    nstates = S.shape[0]
    D = algopy.sign(S) * kimura_d
    H = algopy.zeros_like(S)
    for i in range(nstates):
        for j in range(nstates):
            if not adjacency[i, j]:
                continue
            for x, w in zip(g_quad_x, g_quad_w):
                tmp_a = - S[i, j] * x
                tmp_b = algopy.exp(tmp_a * (D[i, j] * (1-x) + 1))
                H[i, j] += tmp_b * w
            H[i, j] = algopy.reciprocal(H[i, j])
    return H


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

    # compute an adjacency matrix to use as a mask
    adjacency = ts + tv

    # compute the selection differences
    F = get_selection_F(log_counts, compo, algopy.log(nt_distn))
    S = get_selection_S(F)

    # compute the term that corresponds to conditional fixation rate of codons
    codon_fixation = unconstrained_recessivity_fixation(adjacency, kimura_d, S)
    """
    if type(S) == np.ndarray:
        codon_fixation = unconstrained_recessivity_fixation(
                adjacency, kimura_d, S)
    else:
        codon_fixation = algopy_unconstrained_recessivity_fixation(
                kimura_d, S)
    """

    # compute the mutation and fixation components
    A = (kappa * ts + tv) * (omega * nonsyn + syn)
    B = algopy.dot(asym_compo, nt_distn) * codon_fixation

    # construct the pre rate matrix
    pre_Q = A * B
    return pre_Q

