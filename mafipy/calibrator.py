#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division
from mafipy import analytic_formula
import scipy.optimize
import numpy as np
import math


def black_scholes_implied_vol(underlying,
                              strike,
                              rate,
                              maturity,
                              option_value,
                              vol_min=1e-4,
                              vol_max=6.0,
                              max_iter=2000):
    """black_scholes_implied_vol
    calculates implied volatility of black schoels model by brent method.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity:
    :param float option_value: value of option.
    :param float vol_min: lower bound of volatility. default value is 1e-4.
    :param float vol_max: upper bound of volatility. default value is 6.0.
    :param int max_iter: maximum number of iterations. default value is 2000.

    :return: implied volatility.
    :rtype: float.
    """

    def objective_func(vol):
        return (analytic_formula.black_scholes_call_formula(
            underlying, strike, rate, maturity, vol) - option_value)

    result = scipy.optimize.brentq(
        objective_func,
        vol_min,
        vol_max,
        xtol=4 * np.finfo(float).eps,
        rtol=4 * np.finfo(float).eps,
        maxiter=max_iter)

    return result


def black_swaption_implied_vol(init_swap_rate,
                               option_strike,
                               swap_annuity,
                               option_maturity,
                               option_value,
                               vol_min=1e-4,
                               vol_max=6.0,
                               max_iter=2000):
    """black_swaption_implied_vol
    calculates implied volatility of payer's swaption by brent method.

    :param float init_swap_rate:
    :param float option_strike:
    :param float swap_annuity:
    :param float option_maturity:
    :param float option_value: value of option.
    :param float vol_min: lower bound of volatility. default value is 1e-4.
    :param float vol_max: upper bound of volatility. default value is 6.0.
    :param int max_iter: maximum number of iterations. default value is 2000.

    :return: implied volatility.
    :rtype: float.
    """

    def objective_func(vol):
        return (analytic_formula.black_payers_swaption_value(
            init_swap_rate, option_strike, swap_annuity, option_maturity, vol)
            - option_value)

    result = scipy.optimize.brentq(
        objective_func,
        vol_min,
        vol_max,
        xtol=4 * np.finfo(float).eps,
        rtol=4 * np.finfo(float).eps,
        maxiter=max_iter)

    return result


# ----------------------------------------------------------------------------
# SABR calibration
# ----------------------------------------------------------------------------
def _guess_alpha_from_vol_atm(underlying, vol_atm, beta):
    """_guess_alpha_from_vol_atm_
    guess alpha from volatility at the money
    by using Hagan's approximated atm implied volatility.

    .. math::
        \ln\sigma(F, F)
            \\approx \ln \\alpha - (1 - \\beta) \ln F

    :param float underlying:
    :param float vol_atm: volatility at the money.
    :param float beta:

    :return: alpha.
    :rtype: float.
    """
    ln_underlying = math.log(underlying)
    ln_vol_atm = math.log(vol_atm)
    one_minus_beta = 1.0 - beta
    ln_alpha = ln_vol_atm + one_minus_beta * ln_underlying
    return math.exp(ln_alpha)


def sabr_caibration_simple(market_vols,
                           market_strikes,
                           option_maturity,
                           beta,
                           init_alpha=None,
                           init_rho=0.1,
                           init_nu=0.1,
                           nu_lower_bound=1e-8):
    """sabr_caibration_simple
    calibrates SABR parametes, alpha, rho and nu to market volatilities
    by simultaneously minimizing error of market volatilities.

    :param array market_vols: market volatilities.
        Middle of elements in the array must be atm volatility.
    :param array market_strikes: market strikes.
        Middle of elements in the array must be atm strike.
    :param float option_maturity:
    :param float beta: pre-determined beta.
    :param float init_alpha: initial guess of alpha.
        Default value is meaningless value, 0.1.
    :param float init_rho: initial guess of beta.
        Default value is meaningless value, 0.1.
    :param float init_nu: initial guess of nu.
        Default value is meaningless value, 0.1.
    :param float nu_lower_bound:

    :return: alpha, beta, rho, nu.
    :rtype: four float value.

    :raise AssertionError: if length of `market_vols` is not odd.
    :raise AssertionError:
        if length of `market_vols` and `market_strikes` are not same.
    """
    assert(len(market_vols) % 2 == 1,
           "lenght of makret_vols must be odd")
    assert(len(market_strikes) == len(market_vols),
           "market_vols and market_strikes must be same lenght")

    # atm strike is underlying
    underlying = market_strikes[int(len(market_strikes) / 2)]
    vol_atm = market_vols[int(len(market_vols) / 2)]

    if init_alpha is None:
        init_alpha = _guess_alpha_from_vol_atm(underlying, vol_atm, beta)

    # 3-dim func
    # (alpha, rho, nu)
    def objective_func(alpha_rho_nu):
        sabr_vols = [analytic_formula.sabr_implied_vol_hagan(
            underlying,
            strike,
            option_maturity,
            alpha_rho_nu[0],
            beta,
            alpha_rho_nu[1],
            alpha_rho_nu[2]) for strike in market_strikes]
        return np.linalg.norm(np.subtract(market_vols, sabr_vols)) ** 2

    result = scipy.optimize.minimize(
        objective_func,
        [init_alpha, init_rho, init_nu],
        method="L-BFGS-B",
        bounds=((0.0, None), (-1.0, 1.0), (nu_lower_bound, None)))
    alpha, rho, nu = result.x
    return alpha, beta, rho, nu


def _is_real_and_positive(val):
    """_is_real_and_positive

    :param complex val:

    :return: True if `val` is real and positive.
    :rtype: bool.
    """
    if np.isreal(val) and val > 0.0:
        return True
    return False


def _find_alpha(underlying, option_maturity, vol_atm, beta, rho, nu):
    """_find_alpha
    find value of alpha solving the following polynominal equation
    with respect to alpha.

    .. math::
        \\frac{
            (1 - \\beta)^{2} \\tau
        }{
            24F^{2-2\\beta}
        }
        \\alpha^{3}
        +
            \\frac{
                \\rho\\beta\\nu\\tau
            }{
                4F^{1-\\beta}
            }
            \\alpha^{2}
        +
            \left(
                1
                +
                \\frac{
                    2 - 3\\rho^{2}
                }{
                    24
                }
                \\nu^{2} \\tau
            \\right)
            \\alpha
        - \sigma_{\mathrm{atm}} F^{1-\\beta}
        = 0

    :param float underlying:
    :param float option_maturity:
    :param float vol_atm:
    :param float beta:
    :param float rho:
    :param float nu:

    :return: mininum real positive root of the equiation.
    :rtype: float.

    :raise ValueError:
        if there is no positive real roots.
    """
    one_minus_beta = 1.0 - beta
    one_minus_beta2 = one_minus_beta ** 2

    numerator1 = one_minus_beta2 * option_maturity
    denominator1 = 24.0 * (underlying ** (2.0 * one_minus_beta))
    coeff1 = numerator1 / denominator1

    numerator2 = rho * beta * nu * option_maturity
    denominator2 = 4.0 * (underlying ** one_minus_beta)
    coeff2 = numerator2 / denominator2

    numerator3 = (2.0 - 3.0 * rho * rho) * nu * nu * option_maturity
    coeff3 = (1.0 + numerator3 / 24.0)

    constant_term = -vol_atm * (underlying ** one_minus_beta)

    coeffs = [coeff1, coeff2, coeff3, constant_term]

    roots = np.roots(coeffs)
    # find smallest real roots
    positive_real_roots = [
        root.real for root in roots if _is_real_and_positive(root)]
    if len(positive_real_roots) == 0:
        raise ValueError("""
Find no real positive roots for alpha:
    underlying: {0},
    option_maturity: {1},
    vol_atm: {2},
    alpha: {3},
    beta: {4},
    rho: {5},
    nu: {6}""".format(underlying, option_maturity, vol_atm,
                      roots, beta, rho, nu))
    return min(positive_real_roots)


def sabr_caibration_west(market_vols,
                         market_strikes,
                         option_maturity,
                         beta,
                         init_rho=0.1,
                         init_nu=0.1):
    """sabr_caibration_west
    calibrates SABR parameters to market volatilities by algorithm
    in West, G. (2005).
    Calibration of the SABR Model in Illiquid Markets.
    Applied Mathematical Finance, 12(4), 371–385.
    https://doi.org/10.1080/13504860500148672

    :param array market_vols: market volatilities.
        Middle of elements in the array must be atm volatility.
    :param array market_strikes: market strikes corresponded to `market_vol`.
        Middle of elements in the array must be atm strike.
    :param float option_maturity:
    :param float beta: pre-determined beta.
    :param float init_rho: initial guess of rho.
        Default value is meaningless value, 0.1.
    :param float init_nu: initial guess of nu.
        Default value is meaningless value, 0.1.

    :return: alpha, beta, rho, nu.
    :rtype: four float value.

    :raise AssertionError: if length of `market_vols` is not odd.
    :raise AssertionError:
        if length of `market_vols` and `market_strikes` are not same.
    """
    assert(len(market_vols) % 2 == 1,
           "lenght of makret_vols must be odd")
    assert(len(market_strikes) == len(market_vols),
           "market_vols and market_strikes must be same lenght")

    # atm strike is underlying
    underlying = market_strikes[int(len(market_strikes) / 2)]
    vol_atm = market_vols[int(len(market_vols) / 2)]

    def find_alpha(rho, nu):
        return _find_alpha(underlying, option_maturity, vol_atm, beta, rho, nu)

    # 2-dim func
    def objective_func(rho_nu):
        alpha = find_alpha(rho_nu[0], rho_nu[1])
        sabr_vols = [analytic_formula.sabr_implied_vol_hagan(
            underlying,
            strike,
            option_maturity,
            alpha,
            beta,
            rho_nu[0],
            rho_nu[1]) for strike in market_strikes]
        return np.linalg.norm(np.subtract(market_vols, sabr_vols)) ** 2

    result = scipy.optimize.minimize(
        objective_func,
        [init_rho, init_nu],
        method="L-BFGS-B",
        bounds=((-1.0, 1.0), (0.0, None)))
    rho, nu = result.x
    alpha = find_alpha(rho, nu)
    return alpha, beta, rho, nu
