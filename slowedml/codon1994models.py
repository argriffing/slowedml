"""
This module has utility classes for Markov model parameter management.
"""

import algopy

from slowedml import markovutil, codon1994

__all__ = [
        'F1x4',
        'F1x4MG',
        ]


class F1x4:
    """
    Goldman-Yang 1994 codon model.
    """

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 5:
            raise ValueError(len(theta))

    @classmethod
    def natural_to_encoded(cls, natural_theta):
        cls.check_theta(natural_theta)
        return algopy.log(natural_theta)

    @classmethod
    def encoded_to_natural(cls, encoded_theta):
        natural_theta = algopy.exp(encoded_theta)
        cls.check_theta(natural_theta)
        return natural_theta

    @classmethod
    def get_natural_guess(cls):
        natural_theta = np.array([
            3.0, # kappa
            0.1, # omega
            1.0, # pi_A / pi_T
            1.0, # pi_C / pi_T
            1.0, # pi_G / pi_T
            ], dtype=float)
        cls.check_theta(natural_theta)
        return natural_theta

    @classmethod
    def get_distn(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        cls.check_theta(natural_theta)
        nt_distn = markovutil.ratios_to_distn(natural_theta[2:5])
        codon_distn = codon1994.get_f1x4_codon_distn(compo, nt_distn)
        return codon_distn

    @classmethod
    def get_pre_Q(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        cls.check_theta(theta)
        kappa = natural_theta[0]
        omega = natural_theta[1]
        nt_distn = markovutil.ratios_to_distn(theta[2:5])
        codon_distn = codon1994.get_f1x4_codon_distn(compo, nt_distn)
        pre_Q = codon1994.get_pre_Q(
                ts, tv, syn, nonsyn,
                codon_distn, kappa, omega)
        return pre_Q


class F1x4MG:
    """
    Muse-Gaut 1994 codon model.
    """

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 5:
            raise ValueError(len(theta))

    @classmethod
    def natural_to_encoded(cls, natural_theta):
        cls.check_theta(natural_theta)
        return algopy.log(natural_theta)

    @classmethod
    def encoded_to_natural(cls, encoded_theta):
        natural_theta = algopy.exp(encoded_theta)
        cls.check_theta(natural_theta)
        return natural_theta

    @classmethod
    def get_natural_guess(cls):
        natural_theta = np.array([
            3.0, # kappa
            0.1, # omega
            1.0, # pi_A / pi_T
            1.0, # pi_C / pi_T
            1.0, # pi_G / pi_T
            ], dtype=float)
        cls.check_theta(natural_theta)
        return natural_theta

    @classmethod
    def get_distn(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        cls.check_theta(natural_theta)
        nt_distn = markovutil.ratios_to_distn(natural_theta[2:5])
        codon_distn = codon1994.get_f1x4_codon_distn(compo, nt_distn)
        return codon_distn

    @classmethod
    def get_pre_Q(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        cls.check_theta(natural_theta)
        kappa = natural_theta[0]
        omega = natural_theta[1]
        nt_distn = markovutil.ratios_to_distn(natural_theta[2:5])
        pre_Q = codon1994.get_MG_pre_Q(
                ts, tv, syn, nonsyn, asym_compo,
                nt_distn, kappa, omega)
        return pre_Q
