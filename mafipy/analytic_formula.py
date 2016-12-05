#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import math
import numpy as np
import scipy.special

import mafipy.math_formula


def _is_d1_or_d2_infinity(underlying, strike, vol):
    """is_d1_or_d2_infinity

    :param float underlying:
    :param float strike:
    :param float vol:
    :return: check whether :math:`d_{1}` and :math:`d_{2}` is infinity or not.
    :rtype: bool
    """
    return (np.isclose(underlying, 0.0)
            or strike < 0.0
            or vol < 0.0)


def func_d1(underlying, strike, rate, maturity, vol):
    """func_d1
    calculate :math:`d_{1}` in black scholes formula.
    See :py:func:`calc_black_scholes_call_formula`.

    :param float underlying: underlying/strike must be non-negative.
    :param float strike: underlying/strike must be non-negative.
    :param float rate:
    :param float maturity: must be non-negative.
    :param float vol: must be non-negative.
    :return: :math:`d_{1}`.
    :rtype: float
    """
    assert(underlying / strike >= 0.0)
    assert(maturity >= 0.0)
    assert(vol >= 0.0)
    numerator = (
        math.log(underlying / strike) + (rate + vol * vol * 0.5) * maturity)
    denominator = vol * math.sqrt(maturity)
    return numerator / denominator


def func_d2(underlying, strike, rate, maturity, vol):
    """func_d2
    calculate :math:`d_{2}` in black scholes formula.
    See :py:func:`calc_black_scholes_call_formula`.

    :param float underlying: underlying/strike must be non-negative.
    :param float strike: underlying/strike must be non-negative.
    :param float rate:
    :param float maturity: must be non-negative.
    :param float vol: must be non-negative.
    :return: :math:`d_{2}`.
    :rtype: float.
    """
    assert(underlying / strike >= 0.0)
    assert(maturity >= 0.0)
    assert(vol >= 0.0)
    numerator = (
        math.log(underlying / strike) + (rate - vol * vol * 0.5) * maturity)
    denominator = vol * math.sqrt(maturity)
    return numerator / denominator


def d_fprime_by_strike(underlying, strike, rate, maturity, vol):
    """d_fprime_by_strike
    derivative of :math:`d_{1}` with respect to :math:`K`
    where :math:`K` is strike.
    See :py:func:`func_d1`.

    .. math::
        \\frac{\partial }{\partial K} d_{1}(K)
        =  \\frac{K}{\sigma S \sqrt{T}}.

    Obviously, derivative of :math:`d_{1}` and :math:`d_{2}` is same.
    That is

    .. math::

        \\frac{\partial }{\partial K} d_{1}(K)
        = \\frac{\partial }{\partial K} d_{2}(K).

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity: must be non-negative.
    :param float vol:
    :return: value of derivative.
    :rtype: float
    """
    assert(maturity > 0.0)
    assert(vol > 0.0)
    return strike / (math.sqrt(maturity) * vol * underlying)


def d_fhess_by_strike(underlying, strike, rate, maturity, vol):
    """d_fhess_by_strike
    second derivative of :math:`d_{i}\ (i = 1, 2)` with respect to :math:`K`,
    where :math:`K` is strike.

    .. math::
        \\frac{\partial^{2}}{\partial K^{2}} d_{1}(K)
        = \\frac{1}{S \sigma \sqrt{T} },

    where
    :math:`S` is underlying,
    :math:`\sigma` is vol,
    :math:`T` is maturity.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity:
    :param float vol:
    :return: value of second derivative of :math:`d_{1}` or :math:`d_{2}`.
    :rtype: float
    """
    assert(maturity > 0.0)
    assert(vol > 0.0)
    return 1.0 / (math.sqrt(maturity) * vol * underlying)


def calc_black_scholes_call_formula(underlying, strike, rate, maturity, vol):
    """calc_black_scholes_call_formula
    calculate well known black scholes formula for call option.

    .. math::

        c(S, K, r, T, \sigma)
        := S N(d_{1}) - K e^{-rT} N(d_{2}),

    where
    :math:`S` is underlying,
    :math:`K` is strike,
    :math:`r` is rate,
    :math:`T` is maturity,
    :math:`\sigma` is vol,
    :math:`N(\cdot)` is standard normal distribution,
    and :math:`d_{1}` and :math:`d_{2}` are defined as follows:

    .. math::

        \\begin{eqnarray}
            d_{1}
            & = &
            \\frac{\ln(S/K) + (r + \sigma^{2}/2)T}{\sigma \sqrt{T}},
            \\
            d_{2}
            & = &
            \\frac{\ln(S/K) + (r - \sigma^{2}/2)T} {\sigma \sqrt{T}},
        \end{eqnarray}

    :param float underlying: value of underlying.
    :param float strike: strike of call option.
    :param float rate: risk free rate.
    :param float maturity: year fraction to maturity.
    :param float vol: volatility.
    :return: call value.
    :rtype: float
    """
    # option is expired
    if maturity < 0.0:
        return 0.0
    if underlying < 0.0:
        # max(S - K, 0) = (S - K) + max(-(S - K), 0)
        value = calc_black_scholes_call_formula(
            -underlying, -strike, rate, maturity, vol)
        return (underlying - strike) + value
    # underlying is positive
    if _is_d1_or_d2_infinity(underlying, strike, vol):
        return max(underlying - strike, 0)

    d1 = func_d1(underlying, strike, rate, maturity, vol)
    d2 = func_d2(underlying, strike, rate, maturity, vol)
    return (underlying * scipy.special.ndtr(d1)
            - strike * math.exp(-rate * maturity) * scipy.special.ndtr(d2))


def calc_black_scholes_put_formula(underlying, strike, rate, maturity, vol):
    """calc_black_scholes_put_formula
    calculate well known black scholes formula for put option.
    Here value of put option is calculated by put-call parity.

    .. math::

        \\begin{array}{cccl}
            & e^{-rT}(S - K)
                & = & c(S, K, r, T, \sigma) - p(S, K, r, T, \sigma)
                \\\\
            \iff & p(S, K, r, T, \sigma)
                & = & c(S, K, r, T, \sigma) - e^{-rT}(S - K)
        \end{array}

    where
    :math:`c(\cdot)` denotes value of call option,
    :math:`p(\cdot)` denotes value of put option,
    :math:`S` is value of underlying at today,
    :math:`K` is strike,
    :math:`r` is rate,
    :math:`T` is maturity,
    :math:`\sigma` is vol.

    :math:`c(\cdot)` is calculated
    by :py:func:`calc_black_scholes_call_formula`.

    :param float underlying: value of underlying.
    :param float strike: strike of put option.
    :param float rate: risk free rate.
    :param float maturity: year fraction to maturity.
    :param float vol: volatility.
    :return: put value.
    :rtype: float
    """
    call_value = calc_black_scholes_call_formula(
        underlying, strike, rate, maturity, vol)
    discount = math.exp(-rate * maturity)
    return call_value - (underlying - strike * discount)


def calc_black_scholes_call_value(
        underlying,
        strike,
        rate,
        maturity,
        vol,
        today=0.0):
    """calc_black_scholes_call_value
    calculate call value in the case of today is not zero.
    (`maturity` - `today`) is treated as time to expiry.
    See :py:func:`calc_black_scholes_call_formula`.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity:
    :param float vol:
    :param float today:
    :return: call value.
    :rtype: float
    """
    return calc_black_scholes_call_formula(
        underlying, strike, rate, maturity - today, vol)


def calc_black_scholes_put_value(
        underlying,
        strike,
        rate,
        maturity,
        vol,
        today=0.0):
    """calc_black_scholes_put_value

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity:
    :param float vol:
    :param float today:
    :return: put value.
    :rtype: float
    """
    return calc_black_scholes_put_formula(
        underlying, strike, rate, maturity - today, vol)


def black_scholes_call_value_fprime_by_strike(
        underlying,
        strike,
        rate,
        maturity,
        vol):
    """black_scholes_call_value_fprime_by_strike
    First derivative of value of call option with respect to strike
    under black scholes model.
    See :py:func:`calc_black_scholes_call_formula`.

    .. math::
        \\frac{\partial }{\partial K} c(K; S, r, T, \sigma)
        = S \phi(d_{1}(K)) d^{\prime}(K)
        - e^{-rT} \Phi(d_{1}(K))
        - K e^{-rT} \phi(d_{1}(K)) d^{\prime}(K)

    where
    :math:`S` is underlying,
    :math:`K` is strike,
    :math:`r` is rate,
    :math:`T` is maturity,
    :math:`\sigma` is vol,
    :math:`d_{1}, d_{2}` is defined
    in :py:func:`calc_black_scholes_call_formula`,
    :math:`\Phi(\cdot)` is c.d.f. of standard normal distribution,
    :math:`\phi(\cdot)` is p.d.f. of standard normal distribution.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity: must be non-negative.
    :param float vol: volatility. must be non-negative.
    :return: value of derivative.
    :rtype: float
    """
    norm = scipy.stats.norm
    assert(maturity > 0.0)
    assert(vol > 0.0)

    d1 = func_d1(underlying, strike, rate, maturity, vol)
    d2 = func_d2(underlying, strike, rate, maturity, vol)
    d_fprime = d_fprime_by_strike(
        underlying, strike, rate, maturity, vol)

    term1 = underlying * norm.pdf(d1) * d_fprime
    term2 = math.exp(-rate * maturity) * norm.cdf(d2)
    term3 = math.exp(-rate * maturity) * strike * norm.pdf(d2) * d_fprime
    return term1 - term2 - term3


def black_scholes_call_value_fhess_by_strike(
        underlying,
        strike,
        rate,
        maturity,
        vol):
    """black_scholes_call_value_fhess_by_strike
    Second derivative of value of call option with respect to strike
    under black scholes model.
    See :py:func:`calc_black_scholes_call_formula`
    and :py:func:`black_scholes_call_value_fprime_by_strike`.

    .. math::
        \\begin{array}{ccl}
            \\frac{\partial^{2}}{\partial K^{2}} c(0, S; T, K)
                & = &
                S\phi^{\prime}(d_{1}(K))(d_{1}^{\prime}(K))^{2}
                + S\phi(d_{1}(K))d_{1}^{\prime\prime}(K)
                - 2\phi(d_{2}(K))d_{2}^{\prime}(K)
                \\\\
                & &
                - K\phi^{\prime}(d_{2}(K))(d_{2}^{\prime}(K))^{2}
                - K\phi(d_{2}(K))d_{2}^{\prime\prime}(K))
        \end{array}

    where
    :math:`S` is underlying,
    :math:`K` is strike,
    :math:`r` is rate,
    :math:`T` is maturity,
    :math:`\sigma` is vol,
    :math:`d_{1}, d_{2}` is defined
    in :py:func:`calc_black_scholes_call_formula`,
    :math:`\Phi(\cdot)` is c.d.f. of standard normal distribution,
    :math:`\phi(\cdot)` is p.d.f. of standard normal distribution.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity: non-negative.
    :param float vol: volatility. non-negative.
    :return: value of second derivative.
    :rtype: float.
    """
    norm = scipy.stats.norm
    assert(maturity > 0.0)
    assert(vol > 0.0)

    d1 = func_d1(underlying, strike, rate, maturity, vol)
    d2 = func_d2(underlying, strike, rate, maturity, vol)
    d_fprime = d_fprime_by_strike(underlying, strike, rate, maturity, vol)
    d_fhess = d_fhess_by_strike(underlying, strike, rate, maturity, vol)
    d1_density = norm.pdf(d1)
    d1_density_fprime = mafipy.math_formula.norm_pdf_fprime(d1)
    d2_density = norm.pdf(d2)
    d2_density_fprime = mafipy.math_formula.norm_pdf_fprime(d2)

    term1 = underlying * d1_density_fprime * d_fprime * d_fprime
    term2 = underlying * d1_density * d_fhess
    term3 = 2.0 * d2_density * d_fprime
    term4 = strike * d2_density_fprime * d_fprime * d_fprime
    term5 = strike * d2_density * d_fhess
    return term1 + term2 - term3 - term4 - term5


def calc_local_vol_model_implied_vol(
        underlying,
        strike,
        maturity,
        local_vol_func,
        second_derivative_local_vol_func,
        today=0.0):
    """calc_local_vol_model_implied_vol

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param object local_vol_func:
    :param object second_derivative_local_vol_func:
    :param float today:
    """
    average_val = 0.5 * (underlying + strike)
    local_vol_val = local_vol_func(average_val)
    term = (1.0 + second_derivative_local_vol_func(average_val)
            * ((underlying - strike) ** 2)
            / (24.0 * local_vol_val))
    return local_vol_val * term


def calc_sabr_model_implied_vol(
        underlying,
        strike,
        maturity,
        alpha,
        beta,
        rho,
        nu):
    """calc_sabr_model_implied_vol
    calculate implied volatility under SABR model.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha:
    :param float beta:
    :param float rho:
    :param float nu:
    """

    if alpha <= 0:
        ValueError("alpha must be greater than 0.")
    if rho > 1.0 or rho < -1.0:
        ValueError("rho must be between -1 and 1.")
    if nu <= 0.0:
        ValueError("nu must be greater than 0.")
    if underlying <= 0.0:
        ValueError("Approximation not defined for non-positive underlying.")

    log_val = math.log(underlying / strike)
    log_val2 = math.log(underlying / strike) ** 2
    log_val4 = math.log(underlying / strike) ** 4
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
    x = math.log(numerator2 / (1.0 - rho)) if abs(rho - 1.0) > 1E-10 else 1.0
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


def calc_sabr_model_atm_implied_vol(
        underlying,
        maturity,
        alpha,
        beta,
        rho,
        nu):
    """calc_sabr_model_atm_implied_vol
    calculate implied volatility under SABR model.

    :param float underlying:
    :param float maturity:
    :param float alpha:
    :param float beta:
    :param float rho:
    :param float nu:
    """
    oneMinusBeta = 1.0 - beta
    A = underlying ** oneMinusBeta
    term1 = ((oneMinusBeta * alpha) ** 2.0) / (24.0 * (A ** 2))
    term2 = 0.25 * rho * beta * alpha * nu / A
    term3 = (2.0 - 3.0 * rho * rho) * nu * nu / 24.0
    vol = (alpha / A) * (1 + (term1 + term2 + term3) * maturity)
    return vol


def calc_sabr_model_implied_normal_vol(
        underlying,
        strike,
        maturity,
        alpha,
        beta,
        rho,
        nu):
    """calc_sabr_model_implied_vol
    calculate implied volatility under SABR model.

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param float alpha:
    :param float beta:
    :param float rho:
    :param float nu:
    """

    if alpha <= 0:
        ValueError("alpha must be greater than 0.")
    if rho > 1.0 or rho < -1.0:
        ValueError("rho must be between -1 and 1.")
    if nu <= 0.0:
        ValueError("nu must be greater than 0.")
    if underlying <= 0.0:
        ValueError("Approximation not defined for non-positive underlying.")

    log_val = math.log(underlying / strike)
    log_val2 = math.log(underlying / strike) ** 2
    log_val4 = math.log(underlying / strike) ** 4
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
    x = math.log(numerator2 / (1.0 - rho)) if abs(rho - 1.0) > 1E-10 else 1.0
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


class BlackScholesPricerHelper(object):
    """BlackScholesPricerHelper
    Helper functions to generate a function with respect to a sigle variable.
    For instance, black formula as a function of volatility is needed to
    evaluate market smile by implied volatility.
    """

    def make_call_wrt_strike(
            self,
            underlying,
            rate,
            maturity,
            vol,
            today=0.0):
        """make_call_wrt_strike
        make function of black shcoles call formula with respect to function.
        This function return :py:func:`calc_black_scholes_call_value`
        as funciton of a single variable.

        .. math::

            c(K; S, r, T, \sigma) := c(S, K, r, T, \sigma)

        :param float underlying:
        :param float rate:
        :param float maturity:
        :param float vol: volatility.
        :param float today: default value is 0.
        :return: call option pricer as a function of strike.
        :rtype: LambdaType
        """
        return lambda strike: calc_black_scholes_call_value(
            underlying=underlying,
            strike=strike,
            rate=rate,
            maturity=maturity,
            vol=vol,
            today=today)

    def make_put_wrt_strike(
            self,
            underlying,
            rate,
            maturity,
            vol,
            today=0.0):
        """make_put_wrt_strike
        make function of black shcoles put formula with respect to function.
        This function return :py:func:`calc_black_scholes_put_value`
        as funciton of a single variable.

        .. math::

            p(K; S, r, T, \sigma) := p(S, K, r, T, \sigma)

        :param float underlying:
        :param float rate:
        :param float maturity:
        :param float vol: volatility
        :param float today: default value is 0.
        :return: put option pricer as a function of strike.
        :rtype: LambdaType.
        """
        return lambda strike: calc_black_scholes_put_value(
            underlying=underlying,
            strike=strike,
            rate=rate,
            maturity=maturity,
            vol=vol,
            today=today)
