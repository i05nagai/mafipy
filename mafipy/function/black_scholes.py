from __future__ import division, print_function, absolute_import
import math
import numpy as np
import scipy.special

import mafipy.function


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
        underlying, strike, rate, maturity, vol):
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
        underlying, strike, rate, maturity, vol):
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
        underlying, strike, rate, maturity, vol):
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
    d2_density_fprime = mafipy.function.norm_pdf_fprime(d2)

    term1 = d2_density_fprime * d_fprime * d_fprime
    term2 = d2_density * d_fhess
    return -discount * (term1 + term2)


# ----------------------------------------------------------------------------
# Black scholes greeks
# ----------------------------------------------------------------------------
def black_scholes_call_delta(underlying, strike, rate, maturity, vol):
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
    :param float maturity: if maturity <= 0, this function returns 0.
    :param float vol: volatility. This must be positive.

    :return: value of delta.
    :rtype: float.
    """
    assert(vol >= 0.0)
    if maturity <= 0.0:
        return 0.0
    d1 = func_d1(underlying, strike, rate, maturity, vol)
    return scipy.stats.norm.cdf(d1)


def black_scholes_call_gamma(underlying, strike, rate, maturity, vol):
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
    :param float maturity:
        if maturity is not positive, this function returns 0.0.
    :param float vol: volatility. This must be positive.

    :return: value of gamma.
    :rtype: float.
    """
    assert(vol >= 0.0)
    if maturity <= 0.0:
        return 0.0
    d1 = func_d1(underlying, strike, rate, maturity, vol)
    denominator = underlying * vol * math.sqrt(maturity)
    return scipy.stats.norm.pdf(d1) / denominator


def black_scholes_call_vega(underlying, strike, rate, maturity, vol):
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
    :param float maturity: if maturity <= 0.0, this function returns 0.
    :param float vol: volatility. This must be positive.

    :return: value of vega.
    :rtype: float.
    """
    assert(vol >= 0.0)
    if maturity <= 0.0:
        return 0.0
    d1 = func_d1(underlying, strike, rate, maturity, vol)
    return math.sqrt(maturity) * underlying * scipy.stats.norm.pdf(d1)


def black_scholes_call_volga(underlying, strike, rate, maturity, vol):
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
    pdf_fprime = mafipy.function.norm_pdf_fprime(d1)
    ln_moneyness = math.log(underlying / strike)
    numerator = -ln_moneyness + (0.5 * vol * vol - rate) * maturity
    factor = numerator / (vol * vol)

    return underlying * pdf_fprime * factor


def black_scholes_call_theta(underlying, strike, rate, maturity, vol, today):
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


def black_scholes_call_rho(underlying, strike, rate, maturity, vol, today):
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


def black_scholes_call_vega_fprime_by_strike(
        underlying, strike, rate, maturity, vol):
    """black_scholes_call_vega_fprime_by_strike
    calculates derivative of black scholes vega with respect to strike.
    This is required for :py:func:`sabr_pdf`.

    .. math::
        \\frac{\partial}{\partial K}
        \mathrm{Vega}{\mathrm{BSCall}}(S, K, r, T, \sigma)
        =
        S\phi^{\prime}(d_{1}(S, K, r, T, \sigma))
        \\frac{1}{\sigma K}

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
    :param float maturity: if maturity <= 0.0, this function returns 0.
    :param float vol: volatility. This must be positive.

    :return: derivative of vega with respect to strike.
    :rtype: float.
    """
    assert(vol >= 0.0)
    if maturity <= 0.0:
        return 0.0
    d1 = func_d1(underlying, strike, rate, maturity, vol)
    density_fprime = mafipy.function.norm_pdf_fprime(d1)
    return -underlying * density_fprime / (vol * strike)


# ----------------------------------------------------------------------------
# Black scholes distributions
# ----------------------------------------------------------------------------
def black_scholes_cdf(underlying, strike, rate, maturity, vol):
    """black_scholes_cdf
    calculates value of c.d.f. of black scholes model.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity:
    :param float vol: must be positive.

    :return: value of p.d.f. of black scholes model.
    :rtype: float.
    """
    assert(vol > 0.0)
    return (1.0
            + black_scholes_call_value_fprime_by_strike(
                underlying,
                strike,
                rate,
                maturity,
                vol) * math.exp(rate * maturity))


def black_scholes_pdf(underlying, strike, rate, maturity, vol):
    """black_scholes_pdf
    calculates value of p.d.f. of black scholes model.

    :param float underlying:
    :param float strike:
    :param float rate:
    :param float maturity:
    :param float vol: must be positive.

    :return: value of p.d.f. of black scholes model.
    :rtype: float.
    """
    assert(vol > 0.0)
    return (black_scholes_call_value_fhess_by_strike(
        underlying,
        strike,
        rate,
        maturity,
        vol) * math.exp(rate * maturity))
