"""
Diagonalize a matrix with a particular form.

In matlab-like notation, the block matrix form of the input is
[S0 D0 - L, L ; 0, S1 D1]
which can be rewritten using the output block matrices as
[
D0^{-1/2} U0 lam0 U0' D0^{1/2} ,
D0^{-1/2} U0 lam0 U0' D0^{1/2} XQ - XQ D1^{-1/2} U1 lam1 U1' D1^{1/2} XQ ;
0,
D1^{-1/2} U1 lam1 U1' D1^{1/2}
]
which can be expm'd (cf. "matrix exponential")
by exponentiating the diagonals lam0 and lam1.
All inputs and outputs are filenames of arrays of numbers.
Some of the arrays are 1d and some are 2d.
"""

import argparse

import numpy as np
from numpy import testing
import scipy.linalg


def ndot(*args):
    M = args[0]
    for B in args[1:]:
        M = np.dot(M, B)
    return M

def build_block_2x2(M):
    return np.vstack([np.hstack(M[0]), np.hstack(M[1])])

def get_original(
        S0, S1, D0, D1, L,
        ):
    """
    Return the original matrix given its block form.
    """
    Q_original = build_block_2x2([
        [ndot(S0, np.diag(D0)) - np.diag(L), np.diag(L)],
        [np.zeros_like(np.diag(L)), ndot(S1, np.diag(D1))],
        ])
    return Q_original

def get_reconstructed(
        S0, S1, D0, D1, L,
        U0, U1, lam0, lam1, XQ,
        ):
    """
    Return the reconstructed matrix given a spectral form.
    """
    R11 = ndot(
            np.diag(np.reciprocal(np.sqrt(D0))),
            U0,
            np.diag(lam0),
            U0.T,
            np.diag(np.reciprocal(D0)),
            )
    R22 = ndot(
            np.diag(np.reciprocal(np.sqrt(D1))),
            U1,
            np.diag(lam1),
            U1.T,
            np.diag(np.reciprocal(D1)),
            )
    Q_reconstructed = build_block_2x2([
        [R11, ndot(R11, XQ) - ndot(XQ, R22)],
        [np.zeros_like(np.diag(L)), R22],
        ])
    return Q_reconstructed


def check_decomp(
        S0, S1, D0, D1, L,
        U0, U1, lam0, lam1, XQ,
        ):
    Q_original = get_original(S0, S1, D0, D1, L)
    Q_reconstructed = get_reconstructed(
            S0, S1, D0, D1, L,
            U0, U1, lam0, lam1, XQ)
    testing.assert_array_almost_equal(Q_original, Q_reconstructed)

def check_decomp_expm(
        S0, S1, D0, D1, L,
        U0, U1, lam0, lam1, XQ,
        ):
    t = 0.123
    Q_original = get_original(S0, S1, D0, D1, L)
    Q_original_expm = scipy.linalg.expm(t * Q_original)
    Q_spectral_expm = get_reconstructed(
            S0, S1, D0, D1, L,
            U0,
            U1,
            np.exp(t * lam0),
            np.exp(t * lam1),
            XQ,
            )
    testing.assert_array_almost_equal(
            Q_original_expm,
            Q_spectral_expm,
            )


def main(args):
    #FIXME: this code uses slow ways to multiply by diagonal matrices

    # load the input ndarrays
    S0 = np.loadtxt(args.S0_in)
    S1 = np.loadtxt(args.S1_in)
    D0 = np.loadtxt(args.D0_in)
    D1 = np.loadtxt(args.D1_in)
    L = np.loadtxt(args.L_in)

    # compute the first symmetric eigendecomposition
    D0_sqrt = np.sqrt(D0)
    H0 = ndot(np.diag(D0_sqrt), S0, np.diag(D0_sqrt)) - np.diag(L)
    lam0, U0 = scipy.linalg.eigh(H0)

    # compute the second symmetric eigendecomposition
    D1_sqrt = np.sqrt(D1)
    H1 = ndot(np.diag(D1_sqrt), S1, np.diag(D1_sqrt))
    lam1, U1 = scipy.linalg.eigh(H1)

    # solve_sylvester(A, B, Q) finds a solution of AX + XB = Q
    A = ndot(S0, np.diag(D0)) - np.diag(L)
    B = -ndot(S1, np.diag(D1))
    Q = np.diag(L)
    XQ = scipy.linalg.solve_sylvester(A, B, Q)

    # check some stuff if debugging
    if args.debug:
        check_decomp(
                S0, S1, D0, D1, L,
                U0, U1, lam0, lam1, XQ)
        check_decomp_expm(
                S0, S1, D0, D1, L,
                U0, U1, lam0, lam1, XQ)

    # write the output ndarrays
    fmt = '%.17g'
    np.savetxt(args.U0_out, U0, fmt)
    np.savetxt(args.U1_out, U1, fmt)
    np.savetxt(args.lam0_out, lam0, fmt)
    np.savetxt(args.lam1_out, lam1, fmt)
    np.savetxt(args.XQ_out, XQ, fmt)




if __name__ == '__main__':

    # define the command line usage
    parser = argparse.ArgumentParser(description=__doc__)

    # optional arg for testing
    parser.add_argument('--debug', action='store_true',
            help='make the code run slower')

    # input args
    parser.add_argument('--S0-in', required=True,
            help='2d symmetric')
    parser.add_argument('--S1-in', required=True,
            help='2d symmetric')
    parser.add_argument('--D0-in', required=True,
            help='1d invertible diagonal')
    parser.add_argument('--D1-in', required=True,
            help='1d invertible diagonal')
    parser.add_argument('--L-in', required=True,
            help='1d positive diagonal')

    # output args
    parser.add_argument('--U0-out', required=True,
            help='2d orthogonal')
    parser.add_argument('--U1-out', required=True,
            help='2d orthogonal')
    parser.add_argument('--lam0-out', required=True,
            help='1d diagonal')
    parser.add_argument('--lam1-out', required=True,
            help='1d diagonal')
    parser.add_argument('--XQ-out', required=True,
            help='2d')

    # read the inputs and write the outputs
    main(parser.parse_args())

