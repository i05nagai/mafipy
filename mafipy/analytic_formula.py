#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import math
import numpy as np
import scipy.special

import mafipy.math_formula


# ----------------------------------------------------------------------------
# Black scholes european call/put
# ----------------------------------------------------------------------------
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
    See :py:func:`black_scholes_call_formula`.

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
    See :py:func:`black_scholes_call_formula`.

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
    return - 1.0 / (math.sqrt(maturity) * vol * strike)


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
    return 1.0 / (math.sqrt(maturity) * vol * strike * strike)


def black_scholes_call_formula(underlying, strike, rate, maturity, vol):
    """black_scholes_call_formula
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
    d1 = func_d1(underlying, strike, rate, maturity, vol)
    d2 = func_d2(underlying, strike, rate, maturity, vol)
    return (underlying * scipy.special.ndtr(d1)
            - strike * math.exp(-rate * maturity) * scipy.special.ndtr(d2))


def black_scholes_put_formula(underlying, strike, rate, maturity, vol):
    """black_scholes_put_formula
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
    by :py:func:`black_scholes_call_formula`.

    :param float underlying: value of underlying.
    :param float strike: strike of put option.
    :param float rate: risk free rate.
    :param float maturity: year fraction to maturity.
    :param float vol: volatility.
    :return: put value.
    :rtype: float
    """
    call_value = black_scholes_call_formula(
        underlying, strike, rate, maturity, vol)
    discount = math.exp(-rate * maturity)
    return call_value - (underlying - strike * discount)


def black_scholes_call_value(
        underlying,
        strike,
        rate,
        maturity,
        vol,
        today=0.0):
    """black_scholes_call_value
    calculate call value in the case of today is not zero.
    (`maturity` - `today`) is treated as time to expiry.
    See :py:func:`black_scholes_call_formula`.

    * case :math:`S > 0, K < 0`

        * return :math:`S - e^{-rT} K`

    * case :math:`S < 0, K > 0`

        * return 0

    * case :math:`S < 0, K < 0`

        * return :math:`S - e^{-rT}K + E[(-(S - K))^{+}]`

    * case :math:`T \le 0`

        * return 0

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity:
    :param float vol: volatility. This must be positive.
    :param float today:
    :return: call value.
    :rtype: float
    """
    assert(vol >= 0.0)
    time = maturity - today
    # option is expired
    if time < 0.0 or np.isclose(time, 0.0):
        return 0.0
    elif np.isclose(underlying, 0.0):
        return math.exp(-rate * time) * max(-strike, 0.0)
    elif np.isclose(strike, 0.0) and underlying > 0.0:
        return math.exp(-rate * today) * underlying
    elif np.isclose(strike, 0.0) and underlying < 0.0:
        return 0.0
    # never below strike
    elif strike < 0.0 and underlying > 0.0:
        return underlying - math.exp(-rate * time) * strike
    # never beyond strike
    elif strike > 0.0 and underlying < 0.0:
        return 0.0
    elif underlying < 0.0:
        # max(S - K, 0) = (S - K) + max(-(S - K), 0)
        value = black_scholes_call_formula(
            -underlying, -strike, rate, time, vol)
        return (underlying - strike) + value

    return black_scholes_call_formula(
        underlying, strike, rate, time, vol)


def black_scholes_put_value(
        underlying,
        strike,
        rate,
        maturity,
        vol,
        today=0.0):
    """black_scholes_put_value
    evaluates value of put option using put-call parity so that
    this function calls :py:func:`black_scholes_call_value`.
    See :py:func:`black_scholes_put_formula`.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity:
    :param float vol:
    :param float today:
    :return: put value.
    :rtype: float
    """
    time = maturity - today
    # option is expired
    if time < 0.0 or np.isclose(time, 0.0):
        return 0.0
    elif np.isclose(strike, 0.0) and underlying > 0.0:
        return 0.0
    elif np.isclose(strike, 0.0) and underlying < 0.0:
        return underlying * math.exp(-rate * today)

    call_value = black_scholes_call_value(
        underlying, strike, rate, maturity, vol, today)
    discount = math.exp(-rate * time)
    return call_value - (underlying - strike * discount)


def black_scholes_call_value_fprime_by_strike(
        underlying,
        strike,
        rate,
        maturity,
        vol):
    """black_scholes_call_value_fprime_by_strike
    First derivative of value of call option with respect to strike
    under black scholes model.
    See :py:func:`black_scholes_call_formula`.

    .. math::
        \\frac{\partial }{\partial K} c(K; S, r, T, \sigma)
        = - e^{-rT} \Phi(d_{1}(K))

    where
    :math:`S` is underlying,
    :math:`K` is strike,
    :math:`r` is rate,
    :math:`T` is maturity,
    :math:`\sigma` is vol,
    :math:`d_{1}, d_{2}` is defined
    in :py:func:`black_scholes_call_formula`,
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

    d2 = func_d2(underlying, strike, rate, maturity, vol)
    discount = math.exp(-rate * maturity)

    return -discount * norm.cdf(d2)


def black_scholes_call_value_fhess_by_strike(
        underlying,
        strike,
        rate,
        maturity,
        vol):
    """black_scholes_call_value_fhess_by_strike
    Second derivative of value of call option with respect to strike
    under black scholes model.
    See :py:func:`black_scholes_call_formula`
    and :py:func:`black_scholes_call_value_fprime_by_strike`.

    .. math::
        \\begin{array}{ccl}
            \\frac{\partial^{2}}{\partial K^{2}} c(0, S; T, K)
                & = &
                    -e^{-rT}
                    \phi(d_{2}(K)) d^{\prime}(K)
        \end{array}

    where
    :math:`S` is underlying,
    :math:`K` is strike,
    :math:`r` is rate,
    :math:`T` is maturity,
    :math:`\sigma` is vol,
    :math:`d_{1}, d_{2}` is defined
    in :py:func:`black_scholes_call_formula`,
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

    # option is expired
    if maturity < 0.0 or np.isclose(maturity, 0.0):
        return 0.0
    # never below strike
    elif strike <= 0.0 and underlying > 0.0:
        return 0.0
    # never beyond strike
    elif strike > 0.0 and underlying < 0.0:
        return 0.0
    elif underlying < 0.0 and strike < 0.0:
        underlying = -underlying
        strike = -strike

    discount = math.exp(-rate * maturity)
    d2 = func_d2(underlying, strike, rate, maturity, vol)
    d_fprime = d_fprime_by_strike(underlying, strike, rate, maturity, vol)
    d2_density = norm.pdf(d2)

    return -discount * d2_density * d_fprime


def black_scholes_call_value_third_by_strike(
        underlying,
        strike,
        rate,
        maturity,
        vol):
    """black_scholes_call_value_third_by_strike
    Third derivative of value of call option with respect to strike
    under black scholes model.
    See :py:func:`black_scholes_call_formula`
    and :py:func:`black_scholes_call_value_fprime_by_strike`,
    and :py:func:`black_scholes_call_value_fhess_by_strike`.

    .. math::
        \\begin{array}{ccl}
            \\frac{\partial^{3}}{\partial K^{3}} c(0, S; T, K)
            & = &
                -e^{-rT}
                \left(
                    \phi^{\prime}(d_{2}(K))(d^{\prime}(K))^{2}
                        + \phi(d_{2}(K))d^{\prime\prime}(K)
                \\right)
        \end{array}

    where
    :math:`S` is underlying,
    :math:`K` is strike,
    :math:`r` is rate,
    :math:`T` is maturity,
    :math:`\sigma` is vol,
    :math:`d_{1}, d_{2}` is defined
    in :py:func:`black_scholes_call_formula`,
    :math:`\Phi(\cdot)` is c.d.f. of standard normal distribution,
    :math:`\phi(\cdot)` is p.d.f. of standard normal distribution.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity: non-negative.
    :param float vol: volatility. non-negative.
    :return: value of third derivative.
    :rtype: float.
    """
    norm = scipy.stats.norm
    assert(vol > 0.0)

    # option is expired
    if maturity < 0.0 or np.isclose(maturity, 0.0):
        return 0.0

    discount = math.exp(-rate * maturity)
    d2 = func_d2(underlying, strike, rate, maturity, vol)
    d_fprime = d_fprime_by_strike(underlying, strike, rate, maturity, vol)
    d_fhess = d_fhess_by_strike(underlying, strike, rate, maturity, vol)
    d2_density = norm.pdf(d2)
    d2_density_fprime = mafipy.math_formula.norm_pdf_fprime(d2)

    term1 = d2_density_fprime * d_fprime * d_fprime
    term2 = d2_density * d_fhess
    return -discount * (term1 + term2)


def implied_vol_brenner_subrahmanyam(
        underlying, strike, rate, maturity, option_value):
    """implied_vol_brenner_subrahmanyam
    calculates implied volatility by Brenner-Subrahmanym approximation formula.

    .. math::
        \sigma
            \\approx
                \sqrt{\\frac{2 \pi}{T}}
                    \\frac{C}{K}

    where
    :math:`K` is `strike`,
    :math:`T` is `maturity`,
    :math:`C` is `option_value`.
    If :math:`T<=0` or :math:`K \\neq 0` then, this function return 0.

    :param float underlying:
    :param float strike: this funciton returns 0 if strike is 0.
    :param float rate:
    :param float maturity:
    :param float option_value:

    :return: approximated implied vol at the money.
    :rtype: float.
    """
    if np.isclose(strike, 0.0):
        return 0.0
    if maturity <= 0.0:
        return 0.0
    return math.sqrt(2.0 * math.pi / maturity) * option_value / strike


def implied_vol_quadratic_approx(
        underlying, strike, rate, maturity, option_value):
    """implied_vol_quadratic_approx
    calculates implied volatility by Corrado-Miller approximation formula.
    See Corrado, C. J., & Miller, T. W. (1996).
    A note on a simple, accurate formula
    to compute implied standard deviations.
    Journal of Banking & Finance, 20(1996), 595–603.
    https://doi.org/10.1016/0378-4266(95)00014-3

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity:
    :param float option_value:

    :return: approximated implied volatility.
        If `maturity` is not positive, this function returns 0.
    :rtype: float.
    """
    if maturity <= 0.0:
        return 0.0
    discount_strike = math.exp(-rate * maturity) * strike
    moneyness_delta = underlying - discount_strike
    diff = option_value - moneyness_delta / 2.0
    moneyness_delta2 = moneyness_delta ** 2
    # take lower bound
    sqrt_inner = max(diff ** 2 - moneyness_delta2 / math.pi, 0.0)
    factor1 = diff + math.sqrt(sqrt_inner)
    factor2 = (math.sqrt(2.0 * math.pi / maturity) / (underlying + discount_strike))
    return factor1 * factor2


# ----------------------------------------------------------------------------
# Black payers/receivers swaption
# ----------------------------------------------------------------------------
def black_payers_swaption_value(
        init_swap_rate, option_strike, swap_annuity, option_maturity, vol):
    """black_payers_swaption_value
    calculates value of payer's swaptions under black model.

    .. math::
        \\begin{eqnarray}
            V_{\mathrm{payersswap}}(t)
            & = &
                A(t)
                \mathrm{E}_{t}^{A}
                \left[
                    (S(T) - K)^{+}
                \\right]
            \\\\
            & = &
                A(t)(S(t)N(d_{1}) - KN(d_{2})),
            \\\\
            d_{1}
                & = &
                    \\frac{
                        \ln\left(\\frac{S(t)}{K} \\right)
                            + \\frac{1}{2}\sigma^{2}(T - t)
                    }{
                        \sigma \sqrt{T - t}
                    },
            \\\\
            d_{2}
                & = &
                    \\frac{
                        \ln\left(\\frac{S(t)}{K} \\right)
                            - \\frac{1}{2}\sigma^{2}(T - t)
                    }{
                        \sigma \sqrt{T - t}
                    }
        \end{eqnarray}

    where
    :math:`A(t)` is `swap_annuity`,
    :math:`S(t)` is `init_swap_rate`,
    :math:`K` is `option_strike`,
    :math:`\sigma` is `vol`.

    :param float init_swap_rate: initial swap rate.
    :param float option_strike: swaption strike.
    :param float swap_annuity: annuity of referencing swap
    :param float option_maturity: swaption maturity.
    :param float vol: volatilty. this must be positive.

    :raises AssertionError: if volatility is not positive.
    """
    assert(vol > 0.0)
    option_value = black_scholes_call_value(
        init_swap_rate, option_strike, 0.0, option_maturity, vol)
    return swap_annuity * option_value


def black_receivers_swaption_value(
        init_swap_rate, option_strike, swap_annuity, option_maturity, vol):
    """black_receivers_swaption_value
    calculates value of receiver's swaptions under black model.

    .. math::
        \\begin{eqnarray}
            V_{\mathrm{receiver}}(t; S(t), K, A(t), T, \sigma)
            & = &
                A(t)
                \mathrm{E}_{t}^{A}
                \left[
                    (K - S(T))^{+}
                \\right]
        \end{eqnarray}

    where
    :math:`S(t)` is `init_swap_rate`,
    :math:`A(t)` is `swap_annuity`,
    :math:`K` is `option_strike`,
    :math:`T` is `option_maturity`,
    :math:`\sigma` is `vol`.
    :math:`d_{1}, d_{2}` are defined in :py:func:`black_payers_swaption_value`.

    :param init_swap_rate: initial swap rate.
    :param option_strike: strike of swaption.
    :param swap_annuity: annuity of referencing swap.
    :param option_maturity: maturity of swaption.
    :param vol: volatility. This must be non-negative.

    :raises AssertionError: if volatility is not positive.
    """
    assert(vol > 0.0)
    # option is expired
    if option_maturity < 0.0 or np.isclose(option_maturity, 0.0):
        return 0.0

    payer_value = black_payers_swaption_value(
        init_swap_rate, option_strike, swap_annuity, option_maturity, vol)
    forward_value = swap_annuity * (init_swap_rate - option_strike)
    return payer_value - forward_value


def black_payers_swaption_value_fprime_by_strike(
        init_swap_rate,
        option_strike,
        swap_annuity,
        option_maturity,
        vol):
    """black_payers_swaption_value_fprime_by_strike
    First derivative of value of payer's swaption with respect to strike
    under black model.
    See :py:func:`black_payers_swaption_value`.

    .. math::
        \\frac{\partial }{\partial K}
            V_{\mathrm{payer}}(K; S, A, T, \sigma)
            = - A\Phi(d_{2}(K))

    where
    :math:`S` is `init_swap_rate`,
    :math:`K` is `option_strike`,
    :math:`A` is `swap_annuity`,
    :math:`T` is `option_maturity`,
    :math:`\sigma` is `vol`,
    :math:`d_{1}, d_{2}` is defined
    in :py:func:`black_payers_swaption_value`,
    :math:`\Phi(\cdot)` is c.d.f. of standard normal distribution,
    :math:`\phi(\cdot)` is p.d.f. of standard normal distribution.

    :param float init_swap_rate:
    :param float option_strike:
    :param float swap_annuity:
    :param float option_maturity:
    :param float vol: volatility. must be non-negative.
    :return: value of derivative.
    :rtype: float

    :raises AssertionError: if volatility is not positive.

    .. note::
        Roughly speaking, this function calculates :math:`A(t) \Phi(d)`.
        Percentile of probability 1 is infinity so that
        assumption that annuity is equal to 1 is not good assumption
        beacuse of :math:`\Phi(d)` returns 1 in some cases.

    """
    assert(vol > 0.0)
    # option is expired
    if option_maturity < 0.0 or np.isclose(option_maturity, 0.0):
        return 0.0

    value = black_scholes_call_value_fprime_by_strike(
        init_swap_rate, option_strike, 0.0, option_maturity, vol)
    return swap_annuity * value


def black_payers_swaption_value_fhess_by_strike(
        init_swap_rate,
        option_strike,
        swap_annuity,
        option_maturity,
        vol):
    """black_payers_swaption_value_fhess_by_strike
    Second derivative of value of payer's swaption with respect to strike
    under black model.
    See :py:func:`black_payers_swaption_value`.

    .. math::
        \\frac{\partial^{2} }{\partial K^{2}}
            V_{\mathrm{payer}}(K; S, A, T, \sigma)
                = - A\phi(d_{2}(K)) d_{2}^{\prime}(K)

    where
    :math:`S` is `init_swap_rate`,
    :math:`K` is `option_strike`,
    :math:`A` is `swap_annuity`,
    :math:`T` is `option_maturity`,
    :math:`\sigma` is `vol`,
    :math:`d_{1}, d_{2}` is defined
    in :py:func:`black_payers_swaption_value`,
    :math:`\Phi(\cdot)` is c.d.f. of standard normal distribution,
    :math:`\phi(\cdot)` is p.d.f. of standard normal distribution.

    :param float init_swap_rate: initial swap rate.
    :param float option_strike:
    :param float swap_annuity:
    :param float option_maturity:
    :param float vol: volatility. must be non-negative.
    :return: value of derivative.
    :rtype: float

    :raises AssertionError: if volatility is not positive.
    """
    assert(vol > 0.0)

    value = black_scholes_call_value_fhess_by_strike(
        init_swap_rate, option_strike, 0.0, option_maturity, vol)

    return swap_annuity * value


def black_payers_swaption_value_third_by_strike(
        init_swap_rate,
        option_strike,
        swap_annuity,
        option_maturity,
        vol):
    """black_payers_swaption_value_third_by_strike
    Third derivative of value of payer's swaption with respect to strike
    under black model.
    See :py:func:`black_payers_swaption_value`.

    .. math::
        \\frac{\partial^{3} }{\partial K^{3}}
        V_{\mathrm{payer}}(K; S, A, T, \sigma)
            = - A
            \left(
                \phi^{\prime}(d_{2}(K)) (d_{2}^{\prime}(K))^{2}
                    + \phi(d_{2}(K)) d_{2}^{\prime\prime}(K)
            \\right)

    where
    :math:`S` is `init_swap_rate`,
    :math:`K` is `option_strike`,
    :math:`A` is `swap_annuity`,
    :math:`T` is `option_maturity`,
    :math:`\sigma` is `vol`,
    :math:`d_{1}, d_{2}` is defined
    in :py:func:`black_payers_swaption_value`,
    :math:`\Phi(\cdot)` is c.d.f. of standard normal distribution,
    :math:`\phi(\cdot)` is p.d.f. of standard normal distribution.

    :param float init_swap_rate: initial swap rate.
    :param float option_strike: strike of swaption.
    :param float swap_annuity: annuity of referencing swap.
    :param float option_maturity: maturity of swaption.
    :param float vol: volatility. must be non-negative.
    :return: value of derivative.
    :rtype: float

    :raises AssertionError: if volatility is not positive.
    """
    assert(vol > 0.0)
    # option is expired
    if option_maturity < 0.0 or np.isclose(option_maturity, 0.0):
        return 0.0

    value = black_scholes_call_value_third_by_strike(
        init_swap_rate, option_strike, 0.0, option_maturity, vol)

    return swap_annuity * value


# ----------------------------------------------------------------------------
# Black scholes greeks
# ----------------------------------------------------------------------------
def black_scholes_call_delta(
        underlying,
        strike,
        rate,
        maturity,
        vol):
    """black_scholes_call_delta
    calculates black scholes delta.

    .. math::
        \\frac{\partial}{\partial S} c(S, K, r, T, \sigma)
            = \Phi(d_{1}(S))

    where
    :math:`S` is underlying,
    :math:`K` is strike,
    :math:`r` is rate,
    :math:`T` is maturity,
    :math:`\sigma` is volatility,
    :math:`\Phi` is standard normal c.d.f,
    :math:`d_{1}` is defined in
    :py:func:`func_d1`.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity: must be non-negative.
    :param float vol: volatility. This must be positive.
    :return: value of delta.
    :rtype: float.
    """
    assert(maturity >= 0.0)
    assert(vol >= 0.0)
    d1 = func_d1(underlying, strike, rate, maturity, vol)
    return scipy.stats.norm.cdf(d1)


def black_scholes_call_gamma(
        underlying,
        strike,
        rate,
        maturity,
        vol):
    """black_scholes_call_gamma
    calculates black scholes gamma.

    .. math::
        \\frac{\partial^{2}}{\partial S^{2}} c(S, K, r, T, \sigma)
            = -\phi(d_{1}(S, K, r, T, \sigma))
                \\frac{1}{S^{2}\sigma\sqrt{T}}

    where
    :math:`S` is underlying,
    :math:`K` is strike,
    :math:`r` is rate,
    :math:`T` is maturity,
    :math:`\sigma` is volatility,
    :math:`\Phi` is standard normal c.d.f,
    :math:`d_{1}` is defined in
    :py:func:`func_d1`.

    See :py:func:`black_scholes_call_value`.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity: must be non-negative.
    :param float vol: volatility. This must be positive.
    :return: value of gamma.
    :rtype: float.
    """
    assert(maturity >= 0.0)
    assert(vol >= 0.0)
    d1 = func_d1(underlying, strike, rate, maturity, vol)
    denominator = (underlying ** 2) * vol * math.sqrt(maturity)
    return -scipy.stats.norm.pdf(d1) / denominator


def black_scholes_call_vega(
        underlying,
        strike,
        rate,
        maturity,
        vol):
    """black_scholes_call_vega
    calculates black scholes vega.

    .. math::
        \\frac{\partial}{\partial \sigma} c(S, K, r, T, \sigma)
            = \sqrt{T}S\phi(d_{1}(S, K, r, T, \sigma))

    where
    :math:`S` is underlying,
    :math:`K` is strike,
    :math:`r` is rate,
    :math:`T` is maturity,
    :math:`\sigma` is volatility,
    :math:`\phi` is standard normal p.d.f,
    :math:`d_{1}` is defined in
    :py:func:`func_d1`.

    See :py:func:`black_scholes_call_value`.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity: must be non-negative.
    :param float vol: volatility. This must be positive.
    :return: value of vega.
    :rtype: float.
    """
    assert(maturity >= 0.0)
    assert(vol >= 0.0)
    d1 = func_d1(underlying, strike, rate, maturity, vol)
    return math.sqrt(maturity) * underlying * scipy.stats.norm.pdf(d1)


def black_scholes_call_volga(
        underlying, strike, rate, maturity, vol):
    """black_scholes_call_volg
    calculates black scholes volga.

    .. math::
        \\frac{\partial^{2}}{\partial \sigma^{2}} c(S, K, r, T, \sigma)
            S \phi^{\prime}(d_{1}(\sigma))
            \\frac{
                (\\frac{1}{2} \sigma^{2} - r)T
            }{
                \sigma^{2}
            }

    where
    :math:`S` is underlying,
    :math:`K` is strike,
    :math:`r` is rate,
    :math:`T` is maturity,
    :math:`\sigma` is volatility,
    :math:`\phi` is standard normal p.d.f,
    :math:`d_{1}` is defined in
    :py:func:`func_d1`.

    See :py:func:`black_scholes_call_value`.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity: must be non-negative.
    :param float vol: volatility. This must be positive.

    :return: value of volga.
    :rtype: float.
    """
    assert(vol >= 0.0)

    if maturity < 0.0:
        return 0.0
    d1 = func_d1(underlying, strike, rate, maturity, vol)
    pdf_fprime = mafipy.math_formula.norm_pdf_fprime(d1)
    factor = (0.5 * vol * vol - rate) * maturity / (vol * vol)

    return underlying * pdf_fprime * factor


def black_scholes_call_theta(
        underlying,
        strike,
        rate,
        maturity,
        vol,
        today):
    """black_scholes_call_theta
    calculates black scholes theta.

    .. math::
        \\frac{\partial}{\partial t} c(t, S, K, r, T, \sigma)
            = - S * \phi(d_{1})
                \left(
                    \\frac{\sigma}{2\sqrt{T - t}}
                \\right)
                - r e^{-r(T - t)} K \Phi(d_{2})

    where
    :math:`S` is underlying,
    :math:`K` is strike,
    :math:`r` is rate,
    :math:`T` is maturity,
    :math:`\sigma` is volatility,
    :math:`\phi` is standard normal p.d.f,
    :math:`d_{1}` is defined in
    :py:func:`func_d1`.

    See :py:func:`black_scholes_call_value`.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity: must be non-negative.
    :param float vol: volatility. This must be positive.
    :return: value of theta.
    :rtype: float.
    """
    assert(maturity >= 0.0)
    assert(vol >= 0.0)
    norm = scipy.stats.norm
    time = maturity - today
    d1 = func_d1(underlying, strike, rate, time, vol)
    d2 = func_d2(underlying, strike, rate, time, vol)
    term1 = underlying * norm.pdf(d1) * (vol / (2.0 * math.sqrt(time)))
    term2 = rate * math.exp(-rate * time) * strike * norm.cdf(d2)
    return - term1 - term2


def black_scholes_call_rho(
        underlying,
        strike,
        rate,
        maturity,
        vol,
        today):
    """black_scholes_call_rho
    calculates black scholes rho.

    .. math::
        \\frac{\partial}{\partial t} c(t, S, K, r, T, \sigma)
            = (T - t)
                e^{-r (T - t)}
                K \Phi(d_{2})

    where
    :math:`S` is underlying,
    :math:`K` is strike,
    :math:`r` is rate,
    :math:`T` is maturity,
    :math:`\sigma` is volatility,
    :math:`\phi` is standard normal p.d.f,
    :math:`d_{2}` is defined in
    :py:func:`func_d2`.

    See :py:func:`black_scholes_call_value`.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity: must be non-negative.
    :param float vol: volatility. This must be positive.
    :return: value of rho.
    :rtype: float.
    """
    assert(maturity >= 0.0)
    assert(vol >= 0.0)
    norm = scipy.stats.norm
    time = maturity - today
    d2 = func_d2(underlying, strike, rate, time, vol)
    return time * math.exp(-rate * time) * strike * norm.cdf(d2)


# ----------------------------------------------------------------------------
# Local volatility model
# ----------------------------------------------------------------------------
def calc_local_vol_model_implied_vol(
        underlying,
        strike,
        maturity,
        local_vol_func,
        local_vol_fhess,
        today=0.0):
    """calc_local_vol_model_implied_vol
    implied volatility function under following local volatility model:

    .. math::
        dF_{t} = \sigma_{loc}(F_{t}) F dW_{t},
        \quad
        F_{0} = f.

    where
    :math:`F` is underlying,
    :math:`\sigma_{loc}(\cdot)` is local volatility funciton,
    :math:`W` is brownian motion,
    :math:`f` is initial value of underlying.

    Implied volatility function of the model is defined as follows:

    .. math::
        \sigma_{B}(K, f)
            = \sigma_{loc}(\\frac{1}{2}(f + K))
            \left(
                1
                + \\frac{1}{24} \\frac{
                    \sigma_{loc}^{\prime\prime}(\\frac{1}{2}(f + K))
                }{
                    \sigma_{loc}(\\frac{1}{2}(f + K))
                }(f - K)^{2} + \cdots
            \\right).

    See
    Hagan, P. S., Kumar, D., Lesniewski, A. S., & Woodward, D. E. (2002).
    Managing smile risk. Wilmott Magazine, m, 84–108.
    Retrieved from http://www.math.columbia.edu/~lrb/sabrAll.pdf

    :param float underlying:
    :param float strike:
    :param float maturity:
    :param callable local_vol_func:
        local volatility function :math:`\sigma_{loc}` .
    :param callable local_vol_fhess: second derivative of
        local vol function :math:`\sigma_{loc}^{\prime\prime}`
    :param float today:
    :return: implied volatility.
    :rtype: float.
    """
    average_val = 0.5 * (underlying + strike)
    local_vol_val = local_vol_func(average_val)
    # escape zero division
    if np.isclose(local_vol_val, 0.0):
        return 0.0
    term = (1.0 + local_vol_fhess(average_val) * ((underlying - strike) ** 2)
            / (24.0 * local_vol_val))
    return local_vol_val * term


# ----------------------------------------------------------------------------
# SABR model
# ----------------------------------------------------------------------------
def sabr_payers_swaption_value(
        init_swap_rate,
        option_strike,
        swap_annuity,
        option_maturity,
        implied_vol_func):
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
    :param callable implied_vol_func: arguments are
        underlying, strike, swap_annuity.

    :return: value.
    :rtype: float.
    """
    vol = implied_vol_func(
        init_swap_rate, option_strike, option_maturity)
    return black_payers_swaption_value(
        init_swap_rate, option_strike, swap_annuity, option_maturity, vol)


def sabr_receivers_swaption_value(
        init_swap_rate,
        option_strike,
        swap_annuity,
        option_maturity,
        implied_vol_func):
    """sabr_receivers_swaption_value
    calculate european reciever's swaption value.
    This value is calculated by put-call parity.
    See :py:func:`sabr_payers_swaption_value`.

    :param float init_swap_rate:
    :param float option_strike:
    :param float swap_annuity:
    :param float option_maturity:
    :param callable implied_vol_func: arguments are
        underlying, strike, swap_annuity.

    :return: value.
    :rtype: float.
    """
    payers_value = sabr_payers_swaption_value(
        init_swap_rate,
        option_strike,
        swap_annuity,
        option_maturity,
        implied_vol_func)
    forward_value = swap_annuity * (init_swap_rate - option_strike)
    return payers_value - forward_value


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
        This function return :py:func:`black_scholes_call_value`
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
        return lambda strike: black_scholes_call_value(
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
        This function return :py:func:`black_scholes_put_value`
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
        return lambda strike: black_scholes_put_value(
            underlying=underlying,
            strike=strike,
            rate=rate,
            maturity=maturity,
            vol=vol,
            today=today)


class BlackSwaptionPricerHelper(object):
    """BlackSwaptionPricerHelper
    Helper functions to generate a function with respect to a sigle variable.
    For instance, black formula as a function of volatility is needed to
    evaluate market smile by implied volatility.
    """

    def make_payers_swaption_wrt_strike(
            self,
            init_swap_rate,
            swap_annuity,
            option_maturity,
            vol):
        """make_payers_swaption_wrt_strike
        make function of black payer's swaption formula
        with respect to function.
        This function return :py:func:`black_receivers_swaption_value`
        as funciton of a single variable.

        .. math::

            V_{\mathrm{payers}}(K; S, A, T, \sigma)
                := V_{\mathrm{payers}}(S, K, A, T, \sigma)

        :param float init_swap_rate:
        :param float swap_annuity:
        :param float option_maturity:
        :param float vol: volatility.
        :return: payer's swaption pricer as a function of strike.
        :rtype: LambdaType
        """
        return lambda option_strike: black_payers_swaption_value(
            init_swap_rate=init_swap_rate,
            option_strike=option_strike,
            swap_annuity=swap_annuity,
            option_maturity=option_maturity,
            vol=vol)

    def make_receivers_swaption_wrt_strike(
            self,
            init_swap_rate,
            swap_annuity,
            option_maturity,
            vol):
        """make_put_wrt_strike
        make function of black receiver's swaption formula
        with respect to strike.
        This function return :py:func:`black_receivers_swaption_value`
        as funciton of a single variable.

        .. math::

            V_{\mathrm{receivers}}(K; S, A, T, \sigma)
                := V_{\mathrm{receiver}}(S, K, A, T, \sigma)

        :param float init_swap_rate:
        :param float swap_annuity:
        :param float option_maturity:
        :param float vol: volatility
        :return: receiver's swaption pricer as a function of strike.
        :rtype: LambdaType.
        """
        return lambda option_strike: black_receivers_swaption_value(
            init_swap_rate=init_swap_rate,
            option_strike=option_strike,
            swap_annuity=swap_annuity,
            option_maturity=option_maturity,
            vol=vol)
