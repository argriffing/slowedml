"""
Miscellaneous functions related to continuous time Markov processes.
"""

import algopy


def expand_distn(log_ratios):
    """
    This expands log probability ratios into a normalized distribution.
    @param log_ratios: logs of probability ratios
    @return: a probability distribution
    """
    nstates = log_ratios.shape[0] + 1
    unnormalized_distn = algopy.ones(nstates, dtype=log_ratios)
    unnormalized_distn[:-1] = algopy.exp(log_ratios)
    return unnormalized_distn / algopy.sum(unnormalized_distn)

def pre_Q_to_Q(pre_Q, stationary_distn, target_expected_rate):
    """
    Return a matrix with a different diagonal and a different scaling.
    """
    unscaled_Q = pre_Q - algopy.diag(algopy.sum(pre_Q, axis=1))
    r = -algopy.dot(algopy.diag(unscaled_Q), stationary_distn)
    Q = (target_expected_rate / r) * unscaled_Q
    return Q

def get_branch_ll(subs_counts, pre_Q, distn, branch_length):
    """
    This log likelihood calculation function is compatible with algopy.
    @param subs_counts: substitution counts
    @param pre_Q: rates with arbitrary scaling and arbitrary diagonals
    @param distn: initial distribution
    @param branch_length: expected number of changes
    @return: log likelihood
    """
    Q = pre_Q_to_Q(pre_Q, distn, branch_length)
    P = algopy.expm(Q)

    # Scale the rows of the transition matrix by the initial distribution.
    # This scaled matrix will be symmetric if the process is reversible.
    P_scaled = (P.T * distn).T

    # Use the transition matrix and the substitution counts
    # to compute the log likelihood.
    return algopy.sum(algopy.log(P_scaled) * subs_counts)

