"""
This module has utility classes for Markov model parameter management.

"""

import algopy

from slowedml import markovutil, fmutsel

__all__ = [
        'FMutSel_F',
        'FMutSelPD_F',
        'FMutSelPR_F',
        'FMutSelG_F',
        ]


class FMutSel_F:
    """
    A codon model used in Yang-Nielsen 2008.
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
        pre_Q = fmutsel.get_pre_Q(
                log_counts,
                fmutsel.genic_fixation,
                ts, tv, syn, nonsyn, compo, asym_compo,
                nt_distn, kappa, omega,
                )
        return pre_Q


class FMutSelPD_F:
    """
    A new model for which preferred alleles are purely dominant.
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
        nt_distn = markovutil.ratios_to_distn(theta[2:5])
        pre_Q = fmutsel.get_pre_Q(
                log_counts,
                fmutsel.preferred_dominant_fixation,
                ts, tv, syn, nonsyn, compo, asym_compo,
                nt_distn, kappa, omega,
                )
        return pre_Q


class FMutSelPR_F:
    """
    A new model for which preferred alleles are purely recessive.
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
        pre_Q = fmutsel.get_pre_Q(
                log_counts,
                fmutsel.preferred_recessive_fixation,
                ts, tv, syn, nonsyn, compo, asym_compo,
                nt_distn, kappa, omega,
                )
        return pre_Q


class FMutSelG_F:
    """
    A new model.
    This model is related to the model used by Yang and Nielsen in 2008.
    The difference is that this new model has an extra free parameter.
    This extra free parameter controls the recessivity/dominance.
    This model name is new,
    because I have not seen this model described elsewhere.
    Therefore I am giving it a short name so that I can refer to it.
    The name is supposed to be as inconspicuous as possible,
    differing from the standard name of the most closely related
    model in the literature by only one letter.
    This extra letter is the G at the end,
    which is supposed to mean 'generalized.'
    I realize that this is a horrible naming scheme,
    because there are multiple ways that any model can be generalized,
    and this name does not help to distinguish the particular
    way that I have chosen to generalize the model.
    """

    @classmethod
    def check_theta(cls, theta):
        if len(theta) != 6:
            raise ValueError(len(theta))

    @classmethod
    def natural_to_encoded(cls, natural_theta):
        cls.check_theta(natural_theta)
        encoded_theta = algopy.zeros_like(natural_theta)
        encoded_theta[0] = natural_theta[0]
        encoded_theta[1:] = algopy.log(natural_theta[1:])
        return encoded_theta

    @classmethod
    def encoded_to_natural(cls, encoded_theta):
        natural_theta = algopy.zeros_like(encoded_theta)
        natural_theta[0] = encoded_theta[0]
        natural_theta[1:] = algopy.exp(encoded_theta[1:])
        cls.check_theta(natural_theta)
        return natural_theta

    @classmethod
    def get_natural_guess(cls):
        natural_theta = np.array([
            0.0, # kimura d
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
        return codon_distn

    @classmethod
    def get_pre_Q(cls,
            log_counts, codon_distn,
            ts, tv, syn, nonsyn, compo, asym_compo,
            natural_theta,
            ):
        cls.check_theta(natural_theta)
        kimura_d = natural_theta[0]
        kappa = natural_theta[1]
        omega = natural_theta[2]
        nt_distn = markovutil.ratios_to_distn(natural_theta[3:6])
        pre_Q = fmutsel.get_pre_Q_unconstrained(
                log_counts,
                ts, tv, syn, nonsyn, compo, asym_compo,
                kimura_d, nt_distn, kappa, omega,
                )
        return pre_Q

