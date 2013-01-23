"""
Compute pre rate matrices for a family of codon models proposed in 1994.

I am using the term 'pre rate matrix' to describe a matrix
whose off-diagonal entries are are proportional to the rates
of the corresponding rate matrix.
The scale of the 'pre rate matrix' is considered to be meaningless.
The diagonal entries of the 'pre rate matrix' are also
considered to be meaningless.
The inputs and outputs of the functions of this module are defined in terms
of finite distributions, but it would probably be more efficient to use
logs of unnormalized logs of finite distributions.
"""

import algopy


def get_f1x4_codon_distn(
        compo,
        nt_distn,
        ):
    """
    The f1x4 notation is from e.g. Table (1) of Yang and Nielsen 1998.
    @param compo: a (ncodons, 4) design matrix defining codon compositions
    @param nt_distn: empirical or free nucleotide distribution
    @return: codon distribution
    """
    log_nt_distn = algopy.log(nt_distn)
    M = log_nt_distn * compo
    log_codon_distn = algopy.sum(M, axis=-1)
    codon_kernel = algopy.exp(log_codon_distn)
    codon_distn = codon_kernel / algopy.sum(codon_kernel)
    return codon_distn

def get_f3x4_codon_distn(
        full_compo,
        nt_distns,
        ):
    """
    The f3x4 notation is from e.g. Table (1) of Yang and Nielsen 1998.
    Although algopy implements most of the functions of numpy,
    it seems to not have an implementation of the tensordot function.
    @param full_compo: a (ncodons, 3, 4) binary matrix of codon compositions
    @param nt_distns: empirical or free nucleotide distributions
    @return: codon distribution
    """
    log_nt_distns = algopy.log(nt_distns)
    M = log_nt_distns * full_compo
    log_codon_distn = algopy.sum(algopy.sum(M, axis=-1), axis=-1)
    codon_kernel = algopy.exp(log_codon_distn)
    codon_distn = codon_kernel / algopy.sum(codon_kernel)
    return codon_distn

def get_pre_Q(
        ts, tv, syn, nonsyn,
        codon_distn, kappa, omega,
        ):
    """
    In this model family the stationary distn may be either free or empirical.
    By 'free' I mean that it is a function of parameters which are
    estimated inside the maximum likelihood estimation.
    By 'empirical' I mean that it is a function of only parameters which are
    estimated outside of the maximum likelihood.
    One example of an empirically estimated stationary distribution
    is the one that you get by just counting the codons in your data.
    Another example of a stationary distribution which I will call 'empirical'
    is the one that you get by just counting the number of each
    of the four different nucleotides in the data and then computing
    a codon probability by multiplying together the probabilities of its
    three constituent nucleotides.
    An intermediate empirical distribution counts the nucleotides
    separately at each of the three positions within codons.
    @param ts: 2d binary transition nucleotide change design matrix
    @param tv: 2d binary transversion nucleotide change design matrix
    @param syn: 2d binary synonymous nucleotide change design matrix
    @param nonsyn: 2d binary nonsynonymous nucleotide change design matrix
    @param codon_distn: a free or empirical codon stationary distribution
    @param kappa: free param for nucleotide ts/tv rate ratio
    @param omega: free param for nonsyn/syn rate ratio
    @return: pre rate matrix
    """
    return (kappa * ts + tv) * (omega * nonsyn + syn) * codon_distn

