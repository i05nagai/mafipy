#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import math
import numpy as np

import mafipy.function


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
    :param float strike: this function returns 0 if strike is 0.
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
    factor2 = (math.sqrt(2.0 * math.pi / maturity)
               / (underlying + discount_strike))
    return factor1 * factor2


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
        as function of a single variable.

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
        return lambda strike: mafipy.function.black_scholes_call_value(
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
        as function of a single variable.

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
        return lambda strike: mafipy.function.black_scholes_put_value(
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
        as function of a single variable.

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
        function = mafipy.function
        return lambda option_strike: function.black_payers_swaption_value(
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
        as function of a single variable.

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
        function = mafipy.function
        return lambda option_strike: function.black_receivers_swaption_value(
            init_swap_rate=init_swap_rate,
            option_strike=option_strike,
            swap_annuity=swap_annuity,
            option_maturity=option_maturity,
            vol=vol)
