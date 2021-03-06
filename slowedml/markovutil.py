"""
Miscellaneous functions related to continuous time Markov processes.

Clarify whether we are working with logarithms or with non-logarithm values.
Logarithms are nice for unconstrained minimization so that we can
easily incorporate non-negativity,
but the non-logarithm transformation is better when we want to compute
things like standard errors of parameters in a way that can be compared
to the outputs of other software such as paml.
"""

import algopy

def ratios_to_distn(ratios):
    """
    @param ratios: n-1 ratios of leading prob to the trailing prob
    @return: a finite distribution over n states
    """
    n = ratios.shape[0] + 1
    expanded_ratios = algopy.ones(n, dtype=ratios)
    expanded_ratios[:-1] = ratios
    distn = expanded_ratios / algopy.sum(expanded_ratios)
    return distn

def log_ratios_to_distn(log_ratios):
    """
    @param ratios: n-1 ratios of leading prob to the trailing prob
    @return: a finite distribution over n states
    """
    return ratios_to_distn(algopy.exp(log_ratios))

def pre_Q_to_Q(pre_Q, stationary_distn, target_expected_rate):
    """
    Return a matrix with a different diagonal and a different scaling.
    """
    unscaled_Q = pre_Q - algopy.diag(algopy.sum(pre_Q, axis=1))
    r = -algopy.dot(algopy.diag(unscaled_Q), stationary_distn)
    Q = (target_expected_rate / r) * unscaled_Q
    return Q

def guess_branch_length(subs_counts):
    """
    Make a very crude guess of expected number of changes along a branch.
    @param subs_counts: an (nstates, nstates) ndarray of observed substitutions
    @return: crude guess of expected number of changes along the branch
    """
    total_count = algopy.sum(subs_counts)
    diag_count = algopy.sum(algopy.diag(subs_counts))
    crude_estimate = (total_count - diag_count) / float(total_count)
    return crude_estimate

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


def get_branch_mix(probs, pre_Qs, eq_distns, branch_length):
    """
    This log likelihood calculation function is compatible with algopy.
    Note that the word 'mix' in the function name
    does not refer to a mix of branch lengths,
    but rather to a mixture of unscaled parameterized rate matrices.
    @param probs: discrete distribution of mixture probabilities
    @param pre_Qs: rates with arbitrary scaling and arbitrary diagonals
    @param eq_distns: equilibrium distributions
    @param branch_length: expected number of changes
    @return: transition matrices
    """

    # Subtract diagonals to give the unscaled rate matrices.
    # Also compute the expected rates of the unscaled rate matrices.
    # Use an unnecessarily explicit-looking calculation,
    # because the entries inside the probs list
    # and the entries inside the observed expected rates list
    # each have taylor information,
    # but the lists themselves are not taylor-aware.
    # The code could be re-orgainized later so that we are using
    # more explicitly taylor-aware lists.
    unscaled_Qs = []
    r = 0
    for p, pre_Q, eq_distn in zip(probs, pre_Qs, eq_distns):
        unscaled_Q = pre_Q - algopy.diag(algopy.sum(pre_Q, axis=1))
        unscaled_Qs.append(unscaled_Q)
        observed_r = -algopy.dot(algopy.diag(unscaled_Q), eq_distn)
        r = r + p * observed_r

    # Compute the correctly scaled rate matrices
    # so that the expected rate of the mixture is equal
    # to the branch length that has been passed as an argument
    # to this function.
    Qs = []
    for unscaled_Q in unscaled_Qs:
        Q = (branch_length / r) * unscaled_Q
        Qs.append(Q)

    # Return the appropriately time-scaled transition matrices.
    return [algopy.expm(Q) for Q in Qs]


def get_branch_mix_ll(subs_counts, probs, pre_Qs, distn, branch_length):
    """
    This log likelihood calculation function is compatible with algopy.
    Note that the word 'mix' in the function name
    does not refer to a mix of branch lengths,
    but rather to a mixture of unscaled parameterized rate matrices.
    @param subs_counts: substitution counts
    @param probs: discrete distribution of mixture probabilities
    @param pre_Qs: rates with arbitrary scaling and arbitrary diagonals
    @param distn: initial distribution common to both component processes
    @param branch_length: expected number of changes
    @return: log likelihood
    """

    # Get the appropriately time-scaled transition matrices.
    Ps = get_branch_mix(probs, pre_Qs, [distn, distn], branch_length)

    # The probability associated with each count is
    # a convex combination of the probabilities computed with site classes.
    P_mix = probs[0] * Ps[0] + probs[1] * Ps[1]

    # Scale the rows of the transition matrix by the initial distribution.
    # This scaled matrix will be symmetric if the process is reversible.
    P_mix_scaled = (P_mix.T * distn).T

    # Use the probability transition matrix and the substitution counts
    # to compute the log likelihood.
    return algopy.sum(algopy.log(P_mix_scaled) * subs_counts)
