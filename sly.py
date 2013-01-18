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
import scipy


def ndot(*args):
    M = args[0]
    for B in args[1:]:
        M = np.dot(M, B)
    return M


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

