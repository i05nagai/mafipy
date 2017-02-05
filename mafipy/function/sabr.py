#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import math
import numpy as np

import mafipy.function


# ----------------------------------------------------------------------------
# SABR model
# ----------------------------------------------------------------------------
def sabr_payers_swaption_value(
        init_swap_rate,
        option_strike,
        swap_annuity,
        option_maturity,
        alpha,
        beta,
        rho,
        nu):
    """sabr_payers_swaption_value
    calculate european payer's swaption value.
    See
    Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).
    Managing smile risk.
    Wilmott Magazine, m, 84–108.
    Retrieved from http://www.math.columbia.edu/~lrb/sabrAll.pdf

    :param float init_swap_rate:
    :param float option_strike:
    :param float swap_annuity:
    :param float option_maturity:
    :param float alpha: must be greater than 0.
    :param float beta: must be within :math:`[0, 1]`.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: must be positive.

    :return: value.
    :rtype: float.
    """
    vol = sabr_implied_vol_hagan(
        init_swap_rate, option_strike, option_maturity, alpha, beta, rho, nu)
    return mafipy.function.black_payers_swaption_value(
        init_swap_rate, option_strike, swap_annuity, option_maturity, vol)


def sabr_receivers_swaption_value(
        init_swap_rate,
        option_strike,
        swap_annuity,
        option_maturity,
        alpha,
        beta,
        rho,
        nu):
    """sabr_receivers_swaption_value
    calculate european reciever's swaption value.
    This value is calculated by put-call parity.
    See :py:func:`sabr_payers_swaption_value`.

    :param float init_swap_rate:
    :param float option_strike:
    :param float swap_annuity:
    :param float option_maturity:
    :param float alpha: must be greater than 0.
    :param float beta: must be within :math:`[0, 1]`.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: must be positive.

    :return: value.
    :rtype: float.
    """
    payers_value = sabr_payers_swaption_value(
        init_swap_rate,
        option_strike,
        swap_annuity,
        option_maturity,
        alpha,
        beta,
        rho,
        nu)
    forward_value = swap_annuity * (init_swap_rate - option_strike)
    return payers_value - forward_value


# ----------------------------------------------------------------------------
# SABR implied volatility
# ----------------------------------------------------------------------------
def sabr_implied_vol_hagan(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """sabr_implied_vol_hagan
    calculate implied volatility under SABR model.

    .. math::
        \\begin{eqnarray}
            \sigma_{B}(K, S; T)
                & \\approx &
                \\frac{
                    \\alpha
                }{
                (SK)^{(1-\\beta)/2}
                \left(
                    1
                    + \\frac{(1 - \\beta)^{2}}{24}\log^{2}
                        \\frac{S}{K}
                    + \\frac{(1 - \\beta)^{4}}{1920}
                        \log^{4}\\frac{S}{K}
                \\right)
                }
                \left(
                    \\frac{z}{x(z)}
                \\right)
                \\\\
                & &
                \left[
                    1
                    +
                    \left(
                        \\frac{(1 - \\beta)^{2}}{24}
                            \\frac{\\alpha^{2}}{(SK)^{1-\\beta}}
                        + \\frac{1}{4}
                            \\frac{\\rho\\beta\\nu\\alpha}{(SK)^{(1-\\beta)/2}}
                        + \\frac{2 - 3\\rho^{2}}{24}\\nu^{2}
                    \\right) T
                \\right],
                \\\\
            z
                & := &
                \\frac{\\nu}{\\alpha}
                    (SK)^{(1-\\beta)/2}
                    \log\left( \\frac{S}{K} \\right),
                \\\\
            x(z)
                & := &
                \log
                \left(
                    \\frac{
                        \sqrt{1 - 2\\rho z + z^{2}} + z - \\rho
                    }{
                        1 - \\rho
                    }
                \\right)
        \end{eqnarray}

    where
    :math:`S` is underlying,
    :math:`K` is strike,
    :math:`T` is maturity,
    :math:`\\alpha` is alpha,
    :math:`\\beta` is beta,
    :math:`\\rho` is rho,
    :math:`\\nu` is nu.

    See
    Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).
    Managing smile risk. Wilmott Magazine, m, 84–108.
    Retrieved from http://www.math.columbia.edu/~lrb/sabrAll.pdf

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: alpha is :math:`[0, 1]`.
    :param float beta:
    :param float rho: correlation of brownian motion.
        value is in :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be greater than 0.
    :return: implied volatility.
    :rtype: float.
    """
    if alpha <= 0.0:
        raise ValueError("alpha({0}) must be greater than 0.".format(alpha))
    if ((rho > 1.0 and not np.isclose(rho, 1.0))
            or (rho < -1.0 and not np.isclose(rho, -1.0))):
        raise ValueError("rho({0}) must be between -1 and 1.".format(rho))
    if nu <= 0.0:
        raise ValueError("nu must be greater than 0.".format(nu))
    if underlying <= 0.0:
        raise ValueError(
            "Approximation not defined for non-positive underlying({0})."
            .format(underlying))

    log_val = math.log(underlying / strike)
    log_val2 = log_val ** 2
    log_val4 = log_val ** 4
    one_minus_beta = 1.0 - beta
    one_minus_beta2 = one_minus_beta ** 2
    one_minus_beta4 = one_minus_beta ** 4
    common_factor1 = (underlying * strike) ** (one_minus_beta * 0.5)

    # factor1
    term11 = one_minus_beta2 * log_val2 / 24.0
    term12 = one_minus_beta4 * log_val4 / 1920.0
    denominator1 = common_factor1 * (1.0 + term11 + term12)
    factor1 = alpha / denominator1
    # factor2
    z = (nu * common_factor1 * log_val / alpha)
    numerator2 = math.sqrt(1.0 - 2 * rho * z + z * z) + z - rho
    if numerator2 <= 0.0:
        factor2 = 0.0
    elif np.isclose(abs(1.0 - rho), 0.0):
        x = 1.0
        factor2 = 1.0 if (abs(x - z) < 1E-10) else z / x
    else:
        x = math.log(numerator2 / (1.0 - rho))
        factor2 = 1.0 if (abs(x - z) < 1E-10) else z / x
    # factor3
    numerator31 = one_minus_beta2 * (alpha ** 2)
    denominator31 = 24.0 * ((underlying * strike) ** one_minus_beta)
    term31 = numerator31 / denominator31
    numerator32 = 0.25 * rho * beta * nu * alpha
    term32 = numerator32 / common_factor1
    numerator33 = (2.0 - 3.0 * rho * rho) * nu * nu
    term33 = numerator33 / 24.0
    factor3 = 1.0 + (term31 + term32 + term33) * maturity
    return factor1 * factor2 * factor3


def sabr_atm_implied_vol_hagan(
        underlying, maturity, alpha, beta, rho, nu):
    """sabr_atm_implied_vol_hagan
    calculate implied volatility under SABR model at the money.

    .. math::
        \sigma_{ATM}(S; T)
            := \sigma_{B}(S, S; T)
            \\approx
            \\frac{\\alpha}{S^{(1-\\beta)}}
            \left[
                1
                +
                \left(
                    \\frac{(1 - \\beta)^{2}}{24}
                        \\frac{\\alpha^{2}}{S^{2 - 2\\beta}}
                    + \\frac{1}{4}
                        \\frac{\\rho \\beta \\alpha \\nu}{S^{1-\\beta}}
                    + \\frac{2 - 3\\rho^{2}}{24} \\nu^{2}
                \\right) T
            \\right]

    See
    Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).
    Managing smile risk. Wilmott Magazine, m, 84–108.
    Retrieved from http://www.math.columbia.edu/~lrb/sabrAll.pdf

    :param float underlying:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.
    :return: implied volatility.
    :rtype: float.
    """
    oneMinusBeta = 1.0 - beta
    A = underlying ** oneMinusBeta
    term1 = ((oneMinusBeta * alpha) ** 2.0) / (24.0 * (A ** 2))
    term2 = 0.25 * rho * beta * alpha * nu / A
    term3 = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    vol = (alpha / A) * (1 + (term1 + term2 + term3) * maturity)
    return vol


# ----------------------------------------------------------------------------
# SABR implied vol derivative
# ----------------------------------------------------------------------------
def _sabr_implied_vol_hagan_A11(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A11
    One of factor in hagan formua.
    See :py:func:`sabr_implied_vol_hagan`.

    .. math::
        A_{11} := (SK)^{(1-\\beta)/2}

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    return (underlying * strike) ** ((1.0 - beta) / 2.0)


def _sabr_implied_vol_hagan_A11_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A11_fprime_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A11`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    one_minus_beta_half = (1.0 - beta) / 2.0
    one_plus_beta_half = (1.0 + beta) / 2.0

    factor = (one_minus_beta_half * (underlying ** one_minus_beta_half))
    return factor * (strike ** (-one_plus_beta_half))


def _sabr_implied_vol_hagan_A11_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A11_fhess_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A11`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    one_minus_beta_half = (1.0 - beta) / 2.0
    one_plus_beta_half = (1.0 + beta) / 2.0
    three_plus_beta_half = (3.0 + beta) / 2.0

    factor = -(one_minus_beta_half
               * (underlying ** one_minus_beta_half)
               * one_plus_beta_half)
    return factor * (strike ** (-three_plus_beta_half))


def _sabr_implied_vol_hagan_A11_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A11_fprime_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A11`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """

    return _sabr_implied_vol_hagan_A11_fprime_by_strike(
        strike, underlying, maturity, alpha, beta, rho, nu)


def _sabr_implied_vol_hagan_A11_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A11_fhess_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A11`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    return _sabr_implied_vol_hagan_A11_fhess_by_strike(
        strike, underlying, maturity, alpha, beta, rho, nu)


def _sabr_implied_vol_hagan_A12(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A12
    See :py:func:`_sabr_implied_vol_hagan`.

    .. math::
        A_{12}
            & := &
            \left(
                1
                + \\frac{(1 - \\beta)^{2}}{24}
                    \log^{2}\\frac{S}{K}
                + \\frac{(1 - \\beta)^{4}}{1920}
                    \log^{4}\\frac{S}{K}
            \\right)

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    assert(underlying / strike > 0.0)

    one_minus_beta = 1.0 - beta
    one_minus_beta2 = one_minus_beta ** 2
    one_minus_beta4 = one_minus_beta ** 4
    ln_moneyness = math.log(underlying / strike)
    ln_moneyness2 = ln_moneyness ** 2
    ln_moneyness4 = ln_moneyness ** 4

    term1 = one_minus_beta2 * ln_moneyness2 / 24.0
    term2 = one_minus_beta4 * ln_moneyness4 / 1920.0
    return 1.0 + term1 + term2


def _sabr_implied_vol_hagan_A12_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A12_fprime_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A12`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    assert(underlying / strike > 0.0)
    assert(not np.isclose(strike, 0.0))

    one_minus_beta = 1.0 - beta
    one_minus_beta2 = one_minus_beta ** 2
    one_minus_beta4 = one_minus_beta ** 4
    ln_moneyness = math.log(underlying / strike)
    ln_moneyness3 = ln_moneyness ** 3

    term1 = -one_minus_beta2 * ln_moneyness / 12.0
    term2 = -one_minus_beta4 * ln_moneyness3 / 480.0
    return (term1 + term2) / strike


def _sabr_implied_vol_hagan_A12_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A12_fhess_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A12`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    assert(underlying / strike > 0.0)
    assert(not np.isclose(strike, 0.0))

    one_minus_beta = 1.0 - beta
    one_minus_beta2 = one_minus_beta ** 2
    one_minus_beta4 = one_minus_beta ** 4
    ln_moneyness = math.log(underlying / strike)
    ln_moneyness2 = ln_moneyness ** 2
    ln_moneyness3 = ln_moneyness ** 3

    strike2 = strike ** 2
    factor1 = one_minus_beta2 / (12.0 * strike2)
    term1 = factor1 * (ln_moneyness + 1.0)
    # term2
    factor2 = one_minus_beta4 / (160.0 * strike2)
    term2 = factor2 * (ln_moneyness3 + 3.0 * ln_moneyness2)
    return term1 + term2


def _sabr_implied_vol_hagan_A12_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A12_fprime_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A12`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    assert(underlying / strike > 0.0)
    assert(not np.isclose(strike, 0.0))

    one_minus_beta = 1.0 - beta
    one_minus_beta2 = one_minus_beta ** 2
    one_minus_beta4 = one_minus_beta ** 4
    ln_moneyness = math.log(underlying / strike)
    ln_moneyness3 = ln_moneyness ** 3

    term1 = one_minus_beta2 * ln_moneyness / 12.0
    term2 = one_minus_beta4 * ln_moneyness3 / 480.0
    return (term1 + term2) / underlying


def _sabr_implied_vol_hagan_A12_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A12_fhess_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A12`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    assert(underlying / strike > 0.0)
    assert(not np.isclose(strike, 0.0))

    one_minus_beta = 1.0 - beta
    one_minus_beta2 = one_minus_beta ** 2
    one_minus_beta4 = one_minus_beta ** 4
    ln_moneyness = math.log(underlying / strike)
    ln_moneyness2 = ln_moneyness ** 2
    ln_moneyness3 = ln_moneyness ** 3

    underlying2 = underlying ** 2
    factor1 = one_minus_beta2 / (12.0 * underlying2)
    term1 = factor1 * (-ln_moneyness + 1.0)
    # term2
    factor2 = one_minus_beta4 / (160.0 * underlying2)
    term2 = factor2 * (-ln_moneyness3 + 3.0 * ln_moneyness2)
    return term1 + term2


def _sabr_implied_vol_hagan_A1(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A1
    One of factor in hagan formua.
    See :py:func:`sabr_implied_vol_hagan`.

    .. math::
        A_{1}(K, S; T)
            & := &
                (SK)^{(1-\\beta)/2}
                \left(
                    1
                    + \\frac{(1 - \\beta)^{2}}{24}
                        \log^{2}\\frac{S}{K}
                    + \\frac{(1 - \\beta)^{4}}{1920}
                        \log^{4}\\frac{S}{K}
                \\right)

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    assert(underlying / strike > 0.0)

    A11 = _sabr_implied_vol_hagan_A11(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A12 = _sabr_implied_vol_hagan_A12(
        underlying, strike, maturity, alpha, beta, rho, nu)

    return A11 * A12


def _sabr_implied_vol_hagan_A1_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A1_fprime_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A1`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A11 = _sabr_implied_vol_hagan_A11(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A11_fprime = _sabr_implied_vol_hagan_A11_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)

    A12 = _sabr_implied_vol_hagan_A12(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A12_fprime = _sabr_implied_vol_hagan_A12_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    return A11 * A12_fprime + A11_fprime * A12


def _sabr_implied_vol_hagan_A1_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A1_fhess_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A1`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A11 = _sabr_implied_vol_hagan_A11(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A11_fprime = _sabr_implied_vol_hagan_A11_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A11_fhess = _sabr_implied_vol_hagan_A11_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)

    A12 = _sabr_implied_vol_hagan_A12(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A12_fprime = _sabr_implied_vol_hagan_A12_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A12_fhess = _sabr_implied_vol_hagan_A12_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    return A11 * A12_fhess + 2.0 * A11_fprime * A12_fprime + A11_fhess * A12


def _sabr_implied_vol_hagan_A1_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A1_fprime_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A1`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A11 = _sabr_implied_vol_hagan_A11(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A11_fprime = _sabr_implied_vol_hagan_A11_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)

    A12 = _sabr_implied_vol_hagan_A12(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A12_fprime = _sabr_implied_vol_hagan_A12_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    return A11 * A12_fprime + A11_fprime * A12


def _sabr_implied_vol_hagan_A1_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A1_fhess_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A1`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within :math:`[0, 1]`.
    :param float beta: must be greater than 0.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A11 = _sabr_implied_vol_hagan_A11(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A11_fprime = _sabr_implied_vol_hagan_A11_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A11_fhess = _sabr_implied_vol_hagan_A11_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)

    A12 = _sabr_implied_vol_hagan_A12(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A12_fprime = _sabr_implied_vol_hagan_A12_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A12_fhess = _sabr_implied_vol_hagan_A12_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    return A11 * A12_fhess + 2.0 * A11_fprime * A12_fprime + A11_fhess * A12


def _sabr_implied_vol_hagan_A2(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A2
    One of factor in hagan formua.
    See :py:func:`sabr_implied_vol_hagan`.

    .. math::
        A_{2}(K, S; T)
            & := &
                z
        \\nonumber
        \\\\
            & = &
            \\frac{\\nu}{\alpha}
                (SK)^{(1-\\beta)/2}
                \log\left( \\frac{S}{K} \\right),

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    assert(underlying / strike > 0.0)

    one_minus_beta_half = (1.0 - beta) / 2.0
    ln_moneyness = math.log(underlying / strike)

    factor1 = nu / alpha
    factor2 = (underlying * strike) ** one_minus_beta_half
    return factor1 * factor2 * ln_moneyness


def _sabr_implied_vol_hagan_A2_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A2_fprime_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A2`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    assert(alpha > 0.0)
    assert(underlying / strike > 0.0)

    one_minus_beta_half = (1.0 - beta) / 2.0
    one_plus_beta_half = (1.0 + beta) / 2.0
    ln_moneyness = math.log(underlying / strike)

    factor1 = (underlying ** one_minus_beta_half) * nu / alpha
    factor2 = strike ** (-one_plus_beta_half)
    factor3 = one_minus_beta_half * ln_moneyness - 1.0
    return factor1 * factor2 * factor3


def _sabr_implied_vol_hagan_A2_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A2_fhess_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A2`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    assert(alpha > 0.0)
    assert(underlying / strike > 0.0)

    one_minus_beta_half = (1.0 - beta) / 2.0
    three_plus_beta_half = (3.0 + beta) / 2.0
    ln_moneyness = math.log(underlying / strike)

    factor1 = (underlying ** one_minus_beta_half) * nu / alpha
    factor2 = strike ** (-three_plus_beta_half)

    term1 = (beta ** 2 - 1.0) * ln_moneyness / 4.0
    return factor1 * factor2 * (term1 + beta)


def _sabr_implied_vol_hagan_A2_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A2_fprime_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A2`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    assert(alpha > 0.0)
    assert(underlying / strike > 0.0)

    one_minus_beta_half = (1.0 - beta) / 2.0
    one_plus_beta_half = (1.0 + beta) / 2.0
    ln_moneyness = math.log(underlying / strike)

    factor1 = (strike ** one_minus_beta_half) * nu / alpha
    factor2 = underlying ** (-one_plus_beta_half)
    factor3 = one_minus_beta_half * ln_moneyness + 1.0
    return factor1 * factor2 * factor3


def _sabr_implied_vol_hagan_A2_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A2_fhess_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A2`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    assert(alpha > 0.0)
    assert(underlying / strike > 0.0)

    one_minus_beta_half = (1.0 - beta) / 2.0
    three_plus_beta_half = (3.0 + beta) / 2.0
    ln_moneyness = math.log(underlying / strike)

    factor1 = (strike ** one_minus_beta_half) * nu / alpha
    factor2 = underlying ** (-three_plus_beta_half)

    term1 = (beta ** 2 - 1.0) * ln_moneyness / 4.0
    return factor1 * factor2 * (term1 - beta)


def _sabr_implied_vol_hagan_A31(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A31

    .. math::
        A_{31}
            & := &
                \sqrt{1 - 2\\rho z + z^{2}} + z - \\rho

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A2 = _sabr_implied_vol_hagan_A2(
        underlying, strike, maturity, alpha, beta, rho, nu)
    assert(1.0 - 2.0 * rho * A2 + A2 * A2 >= 0.0)
    return math.sqrt(1.0 - 2.0 * rho * A2 + A2 * A2) + A2 - rho


def _sabr_implied_vol_hagan_A31_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A31_fprime_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A31`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A2 = _sabr_implied_vol_hagan_A2(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2_fprime = _sabr_implied_vol_hagan_A2_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)

    assert(1.0 - 2.0 * rho * A2 + A2 * A2 > 0.0)

    numerator = (-rho + A2) * A2_fprime
    denominator = math.sqrt(1.0 - 2.0 * rho * A2 + A2 * A2)
    return numerator / denominator + A2_fprime


def _sabr_implied_vol_hagan_A31_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A31_fhess_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A31`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A2 = _sabr_implied_vol_hagan_A2(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2_fprime = _sabr_implied_vol_hagan_A2_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2_fhess = _sabr_implied_vol_hagan_A2_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)

    sqrt_inner = 1.0 - 2.0 * rho * A2 + A2 * A2
    assert(sqrt_inner > 0.0)

    factor1 = -rho * A2_fhess + A2_fprime ** 2 + A2 * A2_fhess
    term1 = factor1 * sqrt_inner
    term2 = (-rho * A2_fprime + A2 * A2_fprime) ** 2
    numerator = term1 - term2
    denominator = (sqrt_inner) ** 1.5
    return numerator / denominator + A2_fhess


def _sabr_implied_vol_hagan_A31_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A31_fprime_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A31`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A2 = _sabr_implied_vol_hagan_A2(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2_fprime = _sabr_implied_vol_hagan_A2_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)

    assert(1.0 - 2.0 * rho * A2 + A2 * A2 > 0.0)

    numerator = (-rho + A2) * A2_fprime
    denominator = math.sqrt(1.0 - 2.0 * rho * A2 + A2 * A2)
    return numerator / denominator + A2_fprime


def _sabr_implied_vol_hagan_A31_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A31_fhess_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A31`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A2 = _sabr_implied_vol_hagan_A2(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2_fprime = _sabr_implied_vol_hagan_A2_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2_fhess = _sabr_implied_vol_hagan_A2_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)

    sqrt_inner = 1.0 - 2.0 * rho * A2 + A2 * A2
    assert(sqrt_inner > 0.0)

    factor1 = -rho * A2_fhess + A2_fprime ** 2 + A2 * A2_fhess
    term1 = factor1 * sqrt_inner
    term2 = (-rho * A2_fprime + A2 * A2_fprime) ** 2
    numerator = term1 - term2
    denominator = (sqrt_inner) ** 1.5
    return numerator / denominator + A2_fhess


def _sabr_implied_vol_hagan_A3(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A3
    One of factor in hagan formua.
    See :py:func:`sabr_implied_vol_hagan`.

    .. math::
        A_{3}(K, S; T)
            & := &
                x(z)
        \\nonumber
        \\\\
            & = &
                \log
                \left(
                    \\frac{
                        \sqrt{1 - 2\\rho z + z^{2}} + z - \\rho
                    }{
                        1 - \\rho
                    }
                \\right)

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A31 = _sabr_implied_vol_hagan_A31(
        underlying, strike, maturity, alpha, beta, rho, nu)
    assert(A31 / (1.0 - rho) > 0.0)
    return math.log(A31 / (1.0 - rho))


def _sabr_implied_vol_hagan_A3_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A3_fprime_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A3`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A31 = _sabr_implied_vol_hagan_A31(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A31_fprime = _sabr_implied_vol_hagan_A31_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)

    assert(not np.isclose(A31, 0.0))
    return A31_fprime / A31


def _sabr_implied_vol_hagan_A3_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A3_fhess_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A3`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A31 = _sabr_implied_vol_hagan_A31(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A31_fprime = _sabr_implied_vol_hagan_A31_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A31_fhess = _sabr_implied_vol_hagan_A31_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    term1 = -((A31_fprime / A31) ** 2)
    term2 = A31_fhess / A31
    return term1 + term2


def _sabr_implied_vol_hagan_A3_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A3_fprime_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A3`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A31 = _sabr_implied_vol_hagan_A31(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A31_fprime = _sabr_implied_vol_hagan_A31_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)

    assert(not np.isclose(A31, 0.0))
    return A31_fprime / A31


def _sabr_implied_vol_hagan_A3_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A3_fhess_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A3`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    A31 = _sabr_implied_vol_hagan_A31(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A31_fprime = _sabr_implied_vol_hagan_A31_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A31_fhess = _sabr_implied_vol_hagan_A31_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    term1 = -((A31_fprime / A31) ** 2)
    term2 = A31_fhess / A31
    return term1 + term2


def _sabr_implied_vol_hagan_A4(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A4
    One of factor in hagan formua.
    See :py:func:`sabr_implied_vol_hagan`.

    A_{4}(K, S; T)
        & := &
            1
            +
            \left(
                \\frac{(1 - \\beta)^{2}}{24}
                    \\frac{\alpha^{2}}{(SK)^{1-\\beta}}
                + \\frac{1}{4}
                    \\frac{\\rho\\beta\nu\alpha}{(SK)^{(1-\\beta)/2}}
                + \\frac{2 - 3\\rho^{2}}{24}\nu^{2}
            \\right) T

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha:
    :param float beta:
    :param float rho:
    :param float nu:

    :return: value of factor.
    :rtype: float.
    """
    one_minus_beta = 1.0 - beta
    one_minus_beta_half = one_minus_beta / 2.0
    one_minus_beta2 = one_minus_beta ** 2

    numerator1 = one_minus_beta2 * alpha * alpha
    denominator1 = 24.0 * ((underlying * strike) ** one_minus_beta)
    term1 = numerator1 / denominator1

    numerator2 = rho * beta * nu * alpha
    denominator2 = 4.0 * ((underlying * strike) ** one_minus_beta_half)
    term2 = numerator2 / denominator2

    term3 = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0

    return 1.0 + (term1 + term2 + term3) * maturity


def _sabr_implied_vol_hagan_A4_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A4_fprime_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A4`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    one_minus_beta = 1.0 - beta
    one_minus_beta_half = one_minus_beta / 2.0
    one_minus_beta2 = one_minus_beta ** 2
    two_minus_beta = 2.0 - beta
    three_minus_beta_half = (3.0 - beta) / 2.0

    numerator1 = one_minus_beta2 * (alpha ** 2) * (strike ** (-two_minus_beta))
    denominator1 = 24.0 * (underlying ** one_minus_beta)
    term1 = numerator1 / denominator1

    numerator2 = rho * beta * nu * alpha * (strike ** (-three_minus_beta_half))
    denominator2 = 8.0 * (underlying ** one_minus_beta_half)
    term2 = numerator2 / denominator2

    return -(term1 + term2) * maturity * one_minus_beta


def _sabr_implied_vol_hagan_A4_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A4_fhess_by_strike
    See :py:func:`_sabr_implied_vol_hagan_A4`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    one_minus_beta = 1.0 - beta
    one_minus_beta_half = one_minus_beta / 2.0
    one_minus_beta2 = one_minus_beta ** 2
    two_minus_beta = 2.0 - beta
    three_minus_beta = 3.0 - beta
    five_minus_beta_half = (5.0 - beta) / 2.0

    factor1 = strike ** (-three_minus_beta)
    numerator1 = one_minus_beta2 * (alpha ** 2) * (-two_minus_beta)
    denominator1 = 24.0 * (underlying ** one_minus_beta)
    term1 = numerator1 * factor1 / denominator1

    factor2 = strike ** (-five_minus_beta_half)
    numerator2 = rho * beta * nu * alpha * (-three_minus_beta)
    denominator2 = 16.0 * (underlying ** one_minus_beta_half)
    term2 = numerator2 * factor2 / denominator2

    return -(term1 + term2) * maturity * one_minus_beta


def _sabr_implied_vol_hagan_A4_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A4_fprime_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A4`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    one_minus_beta = 1.0 - beta
    one_minus_beta_half = one_minus_beta / 2.0
    one_minus_beta2 = one_minus_beta ** 2
    two_minus_beta = 2.0 - beta
    three_minus_beta_half = (3.0 - beta) / 2.0
    alpha2 = alpha ** 2

    numerator1 = one_minus_beta2 * alpha2 * (underlying ** (-two_minus_beta))
    denominator1 = 24.0 * (strike ** one_minus_beta)
    term1 = numerator1 / denominator1

    numerator2 = (rho * beta * nu * alpha
                  * (underlying ** (-three_minus_beta_half)))
    denominator2 = 8.0 * (strike ** one_minus_beta_half)
    term2 = numerator2 / denominator2

    return -(term1 + term2) * maturity * one_minus_beta


def _sabr_implied_vol_hagan_A4_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """_sabr_implied_vol_hagan_A4_fhess_by_underlying
    See :py:func:`_sabr_implied_vol_hagan_A4`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be within [0, 1].
    :param float beta: must be greater than 0.
    :param float rho: must be within [-1, 1].
    :param float nu: volatility of volatility. This must be positive.

    :return: value of factor.
    :rtype: float.
    """
    one_minus_beta = 1.0 - beta
    one_minus_beta_half = one_minus_beta / 2.0
    one_minus_beta2 = one_minus_beta ** 2
    two_minus_beta = 2.0 - beta
    three_minus_beta = 3.0 - beta
    five_minus_beta_half = (5.0 - beta) / 2.0

    factor1 = underlying ** (-three_minus_beta)
    numerator1 = one_minus_beta2 * (alpha ** 2) * (-two_minus_beta)
    denominator1 = 24.0 * (strike ** one_minus_beta)
    term1 = numerator1 * factor1 / denominator1

    factor2 = underlying ** (-five_minus_beta_half)
    numerator2 = rho * beta * nu * alpha * (-three_minus_beta)
    denominator2 = 16.0 * (strike ** one_minus_beta_half)
    term2 = numerator2 * factor2 / denominator2

    return -(term1 + term2) * maturity * one_minus_beta


def sabr_implied_vol_hagan_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """sabr_implied_vol_hagan_fprime_by_strike
    first derivative of Hagan's SABR implied volatility formula
    with respect to strike.
    See :py:func:`sabr_implied_vol_hagan`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be positve.
    :param float beta: must be within [0, 1].
    :param float rho: must be within [-1, 1].
    :param float nu: must be positive.

    :return: first derivative of hagan implied volatility formula
        with respect to strike.
    :rtype: float
    """
    assert(alpha > 0)
    assert(0 <= beta <= 1.0)
    assert(-1.0 <= rho <= 1.0)
    assert(nu > 0.0)

    A1 = _sabr_implied_vol_hagan_A1(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A1_fprime = _sabr_implied_vol_hagan_A1_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2 = _sabr_implied_vol_hagan_A2(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2_fprime = _sabr_implied_vol_hagan_A2_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A3 = _sabr_implied_vol_hagan_A3(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A3_fprime = _sabr_implied_vol_hagan_A3_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A4 = _sabr_implied_vol_hagan_A4(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A4_fprime = _sabr_implied_vol_hagan_A4_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)

    factor1 = alpha / A1
    factor2 = A2 / A3
    factor3 = A4

    factor11 = -alpha * A1_fprime / (A1 ** 2)
    term1 = factor11 * factor2 * factor3

    factor22 = A2_fprime / A3 - A2 * A3_fprime / (A3 ** 2)
    term2 = factor1 * factor22 * factor3

    factor33 = A4_fprime
    term3 = factor1 * factor2 * factor33

    return term1 + term2 + term3


def sabr_implied_vol_hagan_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """sabr_implied_vol_hagan_fhess_by_strike
    second derivative of Hagan's SABR implied volatility formula
    with respect to strike.
    See :py:func:`sabr_implied_vol_hagan`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be positve.
    :param float beta: must be within [0, 1].
    :param float rho: must be within [-1, 1].
    :param float nu: must be positive.

    :return: second derivative of hagan implied volatility formula
        with respect to strike.
    :rtype: float
    """
    assert(alpha > 0)
    assert(0 <= beta <= 1.0)
    assert(-1.0 <= rho <= 1.0)
    assert(nu > 0.0)

    A1 = _sabr_implied_vol_hagan_A1(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A1_fprime = _sabr_implied_vol_hagan_A1_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A1_fhess = _sabr_implied_vol_hagan_A1_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2 = _sabr_implied_vol_hagan_A2(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2_fprime = _sabr_implied_vol_hagan_A2_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2_fhess = _sabr_implied_vol_hagan_A2_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A3 = _sabr_implied_vol_hagan_A3(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A3_fprime = _sabr_implied_vol_hagan_A3_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A3_fhess = _sabr_implied_vol_hagan_A3_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A4 = _sabr_implied_vol_hagan_A4(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A4_fprime = _sabr_implied_vol_hagan_A4_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A4_fhess = _sabr_implied_vol_hagan_A4_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)

    factor1 = alpha / A1
    A1_2 = A1 ** 2
    fprime1 = -alpha * A1_fprime / A1_2
    fhess1 = -alpha * (-2.0 * (A1_fprime ** 2) / (A1 ** 3) + A1_fhess / A1_2)

    factor2 = A2 / A3
    A3_2 = A3 ** 2
    fprime2 = A2_fprime / A3 - A2 * A3_fprime / A3_2
    fhess2 = (A2_fhess / A3
              - 2.0 * A2_fprime * A3_fprime / A3_2
              - A2 * A3_fhess / A3_2
              + 2.0 * A2 * (A3_fprime ** 2) / (A3 ** 3))

    factor3 = A4
    fprime3 = A4_fprime
    fhess3 = A4_fhess

    term1 = fhess1 * factor2 * factor3
    term2 = factor1 * fhess2 * factor3
    term3 = factor1 * factor2 * fhess3
    term4 = 2.0 * fprime1 * fprime2 * factor3
    term5 = 2.0 * fprime1 * factor2 * fprime3
    term6 = 2.0 * factor1 * fprime2 * fprime3
    return term1 + term2 + term3 + term4 + term5 + term6


def sabr_implied_vol_hagan_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """sabr_implied_vol_hagan_fprime_by_underlying
    first derivative of Hagan's SABR implied volatility formula
    with respect to underlying.
    See :py:func:`sabr_implied_vol_hagan`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be positve.
    :param float beta: must be within [0, 1].
    :param float rho: must be within [-1, 1].
    :param float nu: must be positive.

    :return: first derivative of hagan implied volatility formula
        with respect to underlying.
    :rtype: float
    """
    assert(alpha > 0)
    assert(0 <= beta <= 1.0)
    assert(-1.0 <= rho <= 1.0)
    assert(nu > 0.0)

    A1 = _sabr_implied_vol_hagan_A1(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A1_fprime = _sabr_implied_vol_hagan_A1_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2 = _sabr_implied_vol_hagan_A2(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2_fprime = _sabr_implied_vol_hagan_A2_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A3 = _sabr_implied_vol_hagan_A3(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A3_fprime = _sabr_implied_vol_hagan_A3_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A4 = _sabr_implied_vol_hagan_A4(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A4_fprime = _sabr_implied_vol_hagan_A4_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)

    factor1 = alpha / A1
    factor2 = A2 / A3
    factor3 = A4

    factor11 = -alpha * A1_fprime / (A1 ** 2)
    term1 = factor11 * factor2 * factor3

    factor22 = A2_fprime / A3 - A2 * A3_fprime / (A3 ** 2)
    term2 = factor1 * factor22 * factor3

    factor33 = A4_fprime
    term3 = factor1 * factor2 * factor33

    return term1 + term2 + term3


def sabr_implied_vol_hagan_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu):
    """sabr_implied_vol_hagan_fhess_by_underlying
    second derivative of Hagan's SABR implied volatility formula
    with respect to underlying.
    See :py:func:`sabr_implied_vol_hagan`.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha: must be positve.
    :param float beta: must be within [0, 1].
    :param float rho: must be within [-1, 1].
    :param float nu: must be positive.

    :return: second derivative of hagan implied volatility formula
        with respect to underlying.
    :rtype: float
    """
    assert(alpha > 0)
    assert(0 <= beta <= 1.0)
    assert(-1.0 <= rho <= 1.0)
    assert(nu > 0.0)

    A1 = _sabr_implied_vol_hagan_A1(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A1_fprime = _sabr_implied_vol_hagan_A1_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A1_fhess = _sabr_implied_vol_hagan_A1_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2 = _sabr_implied_vol_hagan_A2(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2_fprime = _sabr_implied_vol_hagan_A2_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A2_fhess = _sabr_implied_vol_hagan_A2_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A3 = _sabr_implied_vol_hagan_A3(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A3_fprime = _sabr_implied_vol_hagan_A3_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A3_fhess = _sabr_implied_vol_hagan_A3_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A4 = _sabr_implied_vol_hagan_A4(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A4_fprime = _sabr_implied_vol_hagan_A4_fprime_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)
    A4_fhess = _sabr_implied_vol_hagan_A4_fhess_by_underlying(
        underlying, strike, maturity, alpha, beta, rho, nu)

    factor1 = alpha / A1
    A1_2 = A1 ** 2
    fprime1 = -alpha * A1_fprime / A1_2
    fhess1 = -alpha * (-2.0 * (A1_fprime ** 2) / (A1 ** 3) + A1_fhess / A1_2)

    factor2 = A2 / A3
    A3_2 = A3 ** 2
    fprime2 = A2_fprime / A3 - A2 * A3_fprime / A3_2
    fhess2 = (A2_fhess / A3
              - 2.0 * A2_fprime * A3_fprime / A3_2
              - A2 * A3_fhess / A3_2
              + 2.0 * A2 * (A3_fprime ** 2) / (A3 ** 3))

    factor3 = A4
    fprime3 = A4_fprime
    fhess3 = A4_fhess

    term1 = fhess1 * factor2 * factor3
    term2 = factor1 * fhess2 * factor3
    term3 = factor1 * factor2 * fhess3
    term4 = 2.0 * fprime1 * fprime2 * factor3
    term5 = 2.0 * fprime1 * factor2 * fprime3
    term6 = 2.0 * factor1 * fprime2 * fprime3
    return term1 + term2 + term3 + term4 + term5 + term6


# ----------------------------------------------------------------------------
# SABR greeks
# ----------------------------------------------------------------------------
def sabr_payers_swaption_delta(
        init_swap_rate, option_strike, swap_annuity, option_maturity,
        alpha, beta, rho, nu):
    """sabr_payers_swaption_delta
    calculate payer's swaption delta under SABR model.
    See :py:func:`sabr_payers_swaption_value`.

    :param float init_swap_rate:
    :param float option_strike:
    :param float swap_annuity:
    :param float option_maturity:
    :param float alpha: must be greater than 0.
    :param float beta: must be within :math:`[0, 1]`.
    :param float rho: must be within :math:`[-1, 1]`.
    :param float nu: must be positive.

    :return: payer's swaption delta.
    :rtype: float.
    """
    assert(alpha > 0)
    assert(0 <= beta <= 1.0)
    assert(-1.0 <= rho <= 1.0)
    assert(nu > 0.0)

    if option_maturity <= 0.0:
        return 0.0

    vol = mafipy.function.sabr_implied_vol_hagan(
        init_swap_rate, option_strike, swap_annuity, alpha, beta, rho, nu)
    bs_delta = mafipy.function.black_payers_swaption_delta(
        init_swap_rate, option_strike, swap_annuity, option_maturity, vol)
    bs_vega = mafipy.function.black_payers_swaption_vega(
        init_swap_rate, option_strike, swap_annuity, option_maturity, vol)
    backbone = mafipy.function.sabr_implied_vol_hagan_fprime_by_underlying(
        init_swap_rate, option_strike, option_maturity,
        alpha, beta, rho, nu)

    return bs_delta + bs_vega * backbone


# ----------------------------------------------------------------------------
# SABR distribution
# ----------------------------------------------------------------------------
def sabr_cdf(underlying, strike, maturity, alpha, beta, rho, nu):
    """sabr_cdf
    calculates value of c.d.f. when underlying follows SABR model.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha:
    :param float beta:
    :param float rho:
    :param float nu:

    :return: value of c.d.f. under SABR model.
    :rtype: float.
    """
    vol = sabr_implied_vol_hagan(
        underlying, strike, maturity, alpha, beta, rho, nu)
    bs_cdf = mafipy.function.black_swaption_cdf(
        underlying, strike, 1.0, maturity, vol)
    bs_vega = mafipy.function.black_payers_swaption_vega(
        underlying, strike, 1.0, maturity, vol)
    vol_fprime = sabr_implied_vol_hagan_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)

    return bs_cdf + bs_vega * vol_fprime


def sabr_pdf(underlying, strike, maturity, alpha, beta, rho, nu):
    """sabr_pdf
    calculates value of p.d.f. when underlying follows SABR model.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha:
    :param float beta:
    :param float rho:
    :param float nu:

    :return: value of c.d.f. under SABR model.
    :rtype: float.
    """
    mf = mafipy.function
    vol = sabr_implied_vol_hagan(
        underlying, strike, maturity, alpha, beta, rho, nu)
    bs_pdf = mf.black_swaption_pdf(
        underlying, strike, 1.0, maturity, vol)
    bs_vega = mf.black_payers_swaption_vega(
        underlying, strike, 1.0, maturity, vol)
    bs_volga = mf.black_payers_swaption_volga(
        underlying, strike, 1.0, maturity, vol)
    bs_vega_fprime = mf.black_payers_swaption_vega_fprime_by_strike(
        underlying, strike, 1.0, maturity, vol)
    vol_fprime = sabr_implied_vol_hagan_fprime_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)
    vol_fhess = sabr_implied_vol_hagan_fhess_by_strike(
        underlying, strike, maturity, alpha, beta, rho, nu)

    return (bs_pdf
            + 2.0 * bs_vega_fprime * vol_fprime
            + bs_volga * vol_fprime * vol_fprime
            + bs_vega * vol_fhess)
