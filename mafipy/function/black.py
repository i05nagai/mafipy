#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import numpy as np

import mafipy.function


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
    option_value = mafipy.function.black_scholes_call_value(
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

    value = mafipy.function.black_scholes_call_value_fprime_by_strike(
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

    value = mafipy.function.black_scholes_call_value_fhess_by_strike(
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

    value = mafipy.function.black_scholes_call_value_third_by_strike(
        init_swap_rate, option_strike, 0.0, option_maturity, vol)

    return swap_annuity * value


# ----------------------------------------------------------------------------
# black payer's/reciever's swaption greeks
# ----------------------------------------------------------------------------
def black_payers_swaption_delta(
        init_swap_rate, option_strike, swap_annuity, option_maturity, vol):
    """black_payers_swaption_delta
    calculates delta of payer's swaptions under black model.

    :param float init_swap_rate: initial swap rate.
    :param float option_strike: swaption strike.
    :param float swap_annuity: annuity of referencing swap
    :param float option_maturity: swaption maturity.
    :param float vol: volatilty. this must be positive.

    :return: delta
    :rtype: float.

    :raises AssertionError: if volatility is not positive.
    """
    assert(vol > 0.0)
    bs_delta = mafipy.function.black_scholes_call_delta(
        init_swap_rate, option_strike, 0.0, option_maturity, vol)
    return swap_annuity * bs_delta


def black_payers_swaption_vega(
        init_swap_rate, option_strike, swap_annuity, option_maturity, vol):
    """black_payers_swaption_vega
    calculates vega of payer's swaption under black model.

    :param float init_swap_rate: initial swap rate.
    :param float option_strike: swaption strike.
    :param float swap_annuity: annuity of referencing swap
    :param float option_maturity: swaption maturity.
    :param float vol: volatilty. this must be positive.

    :return: vega.
    :rtype: float.

    :raises AssertionError: if volatility is not positive.
    """
    assert(vol > 0.0)
    bs_vega = mafipy.function.black_scholes_call_vega(
        init_swap_rate, option_strike, 0.0, option_maturity, vol)
    return swap_annuity * bs_vega


def black_payers_swaption_volga(
        init_swap_rate, option_strike, swap_annuity, option_maturity, vol):
    """black_payers_swaption_volga
    calculates volga of payer's swaption under black model.

    :param float init_swap_rate: initial swap rate.
    :param float option_strike: swaption strike.
    :param float swap_annuity: annuity of referencing swap
    :param float option_maturity: swaption maturity.
    :param float vol: volatilty. this must be positive.

    :return: volga.
    :rtype: float.

    :raises AssertionError: if volatility is not positive.
    """
    assert(vol > 0.0)
    bs_volga = mafipy.function.black_scholes_call_volga(
        init_swap_rate, option_strike, 0.0, option_maturity, vol)
    return swap_annuity * bs_volga


def black_payers_swaption_vega_fprime_by_strike(
        init_swap_rate, option_strike, swap_annuity, option_maturity, vol):
    """black_payers_swaption_vega_fprime_by_strike
    calculates derivative of vega with respect to strike under black model.
    This is required for :py:func:`sabr_pdf`.

    :param float init_swap_rate: initial swap rate.
    :param float option_strike: swaption strike.
    :param float swap_annuity: annuity of referencing swap
    :param float option_maturity: swaption maturity.
    :param float vol: volatilty. this must be positive.

    :return: derivative of vega w.r.t. strike.
    :rtype: float.

    :raises AssertionError: if volatility is not positive.
    """
    assert(vol > 0.0)
    bs_vega_fprime = mafipy.function.black_scholes_call_vega_fprime_by_strike(
        init_swap_rate, option_strike, 0.0, option_maturity, vol)
    return swap_annuity * bs_vega_fprime


# ----------------------------------------------------------------------------
# black payer's/reciever's swaption distribution
# ----------------------------------------------------------------------------
def black_swaption_cdf(
        init_swap_rate, option_strike, swap_annuity, option_maturity, vol):
    """black_swaption_cdf
    calculates value of c.d.f. of black swaption.
    :py:func:`black_payers_swaption_value_fprime_by_strike`.

    :param float init_swap_rate: initial swap rate.
    :param float option_strike: option strike.
    :param float swap_annuity: swap annuity.
    :param float option_maturity: maturity of swaption.
    :param float vol: volatility. non-negative.

    :return: value of c.d.f. of black swaption model.
    :rtype: float.
    """
    assert(vol > 0.0)
    return (1.0
            + black_payers_swaption_value_fprime_by_strike(
                init_swap_rate,
                option_strike,
                swap_annuity,
                option_maturity,
                vol) / swap_annuity)


def black_swaption_pdf(
        init_swap_rate, option_strike, swap_annuity, option_maturity, vol):
    """black_swaption_pdf
    calculates value of p.d.f. of black swaption.
    :py:func:`black_payers_swaption_value_fhess_by_strike`.

    :param float init_swap_rate: initial swap rate.
    :param float option_strike: option strike.
    :param float swap_annuity: swap annuity.
    :param float option_maturity: maturity of swaption.
    :param float vol: volatility. non-negative.

    :return: value of p.d.f. of black swaption model.
    :rtype: float.
    """
    assert(vol > 0.0)
    return (black_payers_swaption_value_fhess_by_strike(
        init_swap_rate,
        option_strike,
        swap_annuity,
        option_maturity,
        vol) / swap_annuity)
