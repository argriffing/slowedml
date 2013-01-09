"""
Check a figure in the Felsenstein 1981 paper.
"""

from collections import defaultdict
import functools

import numpy as np
import numpy.testing
import scipy.optimize
import algopy

import alignll


g_data = """\
xenopus
GCCUACGGCC ACACCACCCU GAAAGUGCCC GAUCUCGUCU GAUCUCGGAA GCCAAGCAGG
GUCGGGCCUG GUUAGUACUU GGAUGGGAGA CCGCCUGGGA AUACCAGGUG UCGUAGGCUU 
salmo
GCUUACGGCC AUACCAGCCU GAAUACGCCC GAUCUCGUCC GAUCUCGGAA GCUAAGCAGG
GUCGGGCCUG GUUAGUACUU GGAUGGGAGA CCGCCUGGGA AUACCAGGUG CUGUAAGCUU 
chicken
GCCUACGGCC AUCCCACCCU GGUAACGCCC GAUCUCGUCU GAUCUCGGAA GCUAAGCAGG
GUCGGGCCUG GUUAGUACUU GGAUGGGAGA CCUCCUGGGA AUACCGGGUG CUGUAGGCUU 
turtle
GUCUACGGCC AUACCACCCU GAACACGCCC GAUCUCGUCU GAUCUCGGAA GCUAAGCAGG
GUCGGGCCUG GUUAGUACUU GGAUGGGAGA CCUCCUGGGA AUACUGGGUG CUGUAGGCUU 
iguana
GCCUACGGCC AUACCACCCU GAACACGCCC GAUCUCGUCU GAUCUCGGAA GCUAAGCAGG
GUCGGGCCUG GUUAGUACUU GGAUGGGAGA CCGCCUGGGA AUACCGGGUG CUGUAGGCUU 
"""

########################################################################
# boilerplate functions for algopy

def eval_grad(f, theta):
    theta = algopy.UTPM.init_jacobian(theta)
    return algopy.UTPM.extract_jacobian(f(theta))

def eval_hess(f, theta):
    theta = algopy.UTPM.init_hessian(theta)
    return algopy.UTPM.extract_hessian(len(theta), f(theta))


def get_jc_rate_matrix():
    """
    This is only for testing.
    It returns a continuous-time Jukes-Cantor rate matrix
    normalized to one expected substitution per time unit.
    """
    nstates = 4
    pre_Q_jc = np.ones((nstates, nstates), dtype=float)
    Q_jc = pre_Q_jc - np.diag(np.sum(pre_Q_jc, axis=1))
    return Q_jc * (1.0 / 3.0)


def neg_log_likelihood(
        ov, v_to_children, root_prior, 
        patterns, pat_mults,
        des,
        log_blens,
        ):
    blens = algopy.exp(log_blens)
    Q = get_jc_rate_matrix()
    de_to_P = dict((de, algopy.expm(b*Q)) for de, b in zip(des, blens))
    log_likelihood = alignll.fels(
            ov, v_to_children, de_to_P, root_prior,
            patterns, pat_mults,
            )
    neg_ll = -log_likelihood
    #print 'branch lengths:'
    #print blens
    #print 'neg log likelihood:'
    #print neg_ll
    #print
    return neg_ll

def main():

    # Read the data in an ad hoc way.
    ntaxa = 5
    states = 'ACGU'
    lines = g_data.splitlines()
    taxon_names = []
    taxon_seqs = []
    for i in range(ntaxa):

        # append the taxon name
        taxon_name = lines[i*3].strip()
        taxon_names.append(taxon_name)

         # append the taxon sequence
        taxon_seq = lines[i*3+1] + lines[i*3+2]
        taxon_seq = ''.join(taxon_seq.split())
        taxon_seqs.append([states.index(x) for x in taxon_seq])

    # construct an alignment pattern
    raw_patterns = zip(*taxon_seqs)
    pattern_to_mult = defaultdict(int)
    for pat in raw_patterns:
        pattern_to_mult[pat] += 1
    patterns, pattern_mults = zip(*pattern_to_mult.items())
    npatterns = len(patterns)
    patterns = np.hstack([np.array(patterns), -np.ones((npatterns, 3))])
    pattern_mults = np.array(pattern_mults)

    # define the ordered vertices with child vertices before parent vertices
    ov = range(8)
    v_to_children = {
            7 : [2, 5, 6],
            6 : [0, 1],
            5 : [3, 4],
            }
    root_prior = 0.25 * np.ones(4)

    # Construct a list of directed edges on the tree.
    #des = [(p, c) for p, cs in v_to_children.items() for c in cs]
    de_blen_map = {
            (6, 0) : 0.0691,
            (6, 1) : 0.0561,
            (5, 3) : 0.0302,
            (5, 4) : 0.0483,
            (7, 2) : 0.0537,
            (7, 6) : 0.0144,
            (7, 5) : 0.0097,
            }
    des, blens = zip(*de_blen_map.items())

    # Precompute negative log likeilhood args.
    args = (ov, v_to_children, root_prior, patterns, pattern_mults, des)

    # Construct a partial functions using the args.
    f = functools.partial(neg_log_likelihood, *args)
    g = functools.partial(eval_grad, f)
    h = functools.partial(eval_hess, f)

    # Guess the branch lengths.

    #log_blens = np.log(0.1 * np.ones(len(des)))
    log_blens = np.log(blens)

    #result = scipy.optimize.fmin_bfgs(
    result = scipy.optimize.fmin_ncg(
        f,
        log_blens,
        fprime=g,
        fhess=h,
        avextol=1e-6,
        disp=True,
        full_output=True,
        )
    #result = scipy.optimize.fmin(
        #f,
        #log_blens,
        #)

    # report the result
    best_log_blens = result[0]
    best_blens = np.exp(best_log_blens)
    for de, blen in zip(des, best_blens):
        print de, ':', blen


if __name__ == '__main__':
    main()

