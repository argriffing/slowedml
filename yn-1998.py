"""
Reproduce the max likelihood for a codon model of Yang and Nielsen 1998.
"""

import math

import numpy as np
import numpy
import algopy
import scipy
import scipy.optimize

import design
import alignll

def get_Q(
        ts, tv, syn, nonsyn,
        stationary_distn,
        log_mu, log_kappa, log_omega,
        ):
    """
    @param ts: 2d binary transition nucleotide change design matrix
    @param tv: 2d binary transversion nucleotide change design matrix
    @param syn: 2d binary synonymous nucleotide change design matrix
    @param nonsyn: 2d binary nonsynonymous nucleotide change design matrix
    @param stationary_distn: a fixed codon stationary distribution
    @param log_mu: free param for expected number of substitutions per time
    @param log_kappa: free param for nucleotide ts/tv rate ratio
    @param log_omega: free param for nonsyn/syn rate ratio
    @return: rate matrix
    """

    # exponentiate the free parameters
    mu = algopy.exp(log_mu)
    kappa = algopy.exp(log_kappa)
    omega = algopy.exp(log_omega)

    # construct a matrix whose off-diagonals are proportional to the rates
    pre_Q = (kappa * ts + tv) * (omega * nonsyn + syn) * stationary_distn

    # use the row sums to compute the rescaled rate matrix
    r = algopy.sum(pre_Q, axis=1)
    old_rate = algopy.dot(r, stationary_distn)
    Q = (mu / old_rate) * (pre_Q - algopy.diag(r))
    return Q


def get_pattern_based_log_likelihood(
        ov, v_to_children, de_to_P, root_prior,
        patterns, pattern_weights,
        ):
    """
    def fels(
            ov, v_to_children, de_to_P, root_prior,
            patterns, pattern_weights,
            ):
        @param ov: ordered vertices with child vertices before parent vertices
        @param v_to_children: map from a vertex to a sequence of child vertices
        @param de_to_P: map from a directed edge to a transition matrix
        @param root_prior: equilibrium distribution at the root
        @param patterns: each pattern assigns a state to each leaf
        @param pattern_weights: a multiplicity for each pattern
        @return: log likelihood
    """
    pass

def eval_f(
        theta,
        patterns, pattern_weights,
        stationary_distn,
        ts, tv, syn, nonsyn,
        ):
    """
    @param theta: vector of free variables with sensitivities
    """

    # unpack theta
    log_mu_0 = theta[0]
    log_mu_1 = theta[1]
    log_mu_2 = theta[2]
    log_kappa = theta[3]
    log_omega = theta[4]

    # construct the transition matrices
    transition_matrices = []
    for log_mu in (
            log_mu_0,
            log_mu_1,
            log_mu_2,
            ):
        Q = get_Q(
                ts, tv, syn, nonsyn
                stationary_distn,
                log_mu, log_kappa, log_omega)
        P = algopy.expm(Q)
        transition_matrices.append(P)

    # return the neg log likelihood
    npatterns = patterns.shape[0]
    nstates = patterns.shape[1]
    ov = range(4)
    v_to_children = {3 : [0, 1, 2]}
    de_to_P = {
            (3, 0) : transition_matrices[0],
            (3, 1) : transition_matrices[1],
            (3, 2) : transition_matrices[2],
            }
    root_prior = stationary_distn
    log_likelihood = alignll.fels(
            ov, v_to_children, de_to_P, root_prior,
            patterns, pattern_weights,
            )
    neg_ll = -log_likelihood
    print neg_ll
    return neg_ll


def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))

def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))


def subs_counts_to_pattern_and_weights(subs_counts):
    """
    This is a new function which tries to generalize to tree likelihoods.
    """
    nstates = subs_counts.shape[0]
    npatterns = nstates * nstates
    patterns = np.zeros((npatterns, 2), dtype=int)
    pattern_weights = np.zeros(npatterns, dtype=float)
    for i in range(nstates):
        for j in range(nstates):
            pattern_index = i*nstates + j
            patterns[pattern_index][0] = i
            patterns[pattern_index][1] = j
            pattern_weights[pattern_index] = subs_counts[i, j]
    return patterns, pattern_weights


def main(args):

    # read the description of the genetic code
    with open(args.code) as fin_gcode:
        arr = list(csv.reader(fin_gcode, delimiter='\t'))
        indices, aminos, codons = zip(*arr)
        if [int(x) for x in indices] != range(len(indices)):
            raise ValueError

    # load the ordered directed edges
    DE = np.loadtxt(args.edges_in, delimiter='\t', dtype=int)

    # load the alignment pattern
    pattern = np.loadtxt(args.pattern_in, delimiter='\t', dtype=int)

    # load the alignment weights
    weights = np.loadtxt(args.weights_in, delimiter='\t', dtype=float)

    # precompute some design matrices
    adj = design.get_adjacency(codons)
    ts = design.get_nt_transitions(codons)
    tv = design.get_nt_transversion(codons)

    # define the initial guess of the parameter values
    theta = np.array([
        -2, # log_mu_0
        -2, # log_mu_1
        -2, # log_mu_2
        1,  # log_kappa
        -3, # log_omega
        ], dtype=float)

    # construct the args to the neg log likelihood function
    fmin_args = (
            #subs_counts,
            patterns, pattern_weights,
            log_counts, v,
            fixation_h,
            ts, tv, syn, nonsyn, compo, asym_compo,
            )

    # do the search, using information about the gradient and hessian
    """
    results = scipy.optimize.fmin_ncg(
            eval_f,
            theta,
            eval_grad_f,
            fhess_p=None,
            fhess=eval_hess_f,
            args=fmin_args,
            avextol=1e-05,
            epsilon=1.4901161193847656e-08,
            maxiter=100,
            full_output=True,
            disp=1,
            retall=0,
            callback=None,
            )
    """
    results = scipy.optimize.fmin(
            eval_f,
            theta,
            args=fmin_args,
            )

    # report a summary of the maximum likelihood search
    print results
    x = results[0]
    print numpy.exp(x)



if __name__ == '__main__':

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-in', required=True,
            help='an input file that defines the genetic code')
    parser.add_argument('--edges-in', required=True,
            help='ordered tree edges')
    parser.add_argument('--pattern-in', required=True,
            help='codon alignment pattern')
    parser.add_argument('--weights-in', required=True,
            help='codon alignment weights')

    # read the args
    args = parser.parse_args()

    # open files for reading and writing
    if args.i == '-':
        fin = sys.stdin
    else:
        fin = open(args.i)
    if args.o == '-':
        fout = sys.stdout
    else:
        fout = open(args.o, 'w')
    if args.taxa:
        fin_taxa = open(args.taxa)
    else:
        fin_taxa = None

    # read and write the data
    with open(args.code, 'r') as fin_gcode:
        main(fin, fin_gcode, fin_taxa, fout)

    # close the files
    if fin_taxa is not None:
        fin_taxa.close()
    if fin is not sys.stdin:
        fin.close()
    if fout is not sys.stdout:
        fout.close()

