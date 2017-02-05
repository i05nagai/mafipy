#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import functools

from . import util


def payoff_call(underlying, strike, gearing=1.0):
    """payoff_call
    Payoff of call option.

    :param float underlying:
    :param float strike:
    :param float gearing: Coefficient of this option. Default value is 1.
    :return: payoff call option.
    :rtype: float
    """
    return gearing * max(underlying - strike, 0)


def payoff_call_fprime(underlying, strike, gearing=1.0):
    """payoff_call_fprime
    derivative of payoff of call option with respect to underlying.

    :param underlying:
    :param strike:
    :param gearing:
    :return: value of derivative with respect to underlying.
    :rtype: float
    """
    if underlying > strike:
        return gearing
    else:
        return 0.0


def payoff_put(underlying, strike, gearing=1.0):
    """payoff_put
    Payoff of put option.

    :param float underlying:
    :param float strike:
    :param float gearing: Coefficient of this option. Default value is 1.
    """
    return gearing * max(strike - underlying, 0)


def payoff_put_fprime(underlying, strike, gearing=1.0):
    """payoff_put_fprime
    derivative of payoff of call option with respect to underlying.

    :param underlying:
    :param strike:
    :param gearing:
    :return: value of derivative with respect to underlying.
    :rtype: float
    """
    if underlying < strike:
        return -gearing
    else:
        return 0.0


def payoff_bull_spread(underlying, lower_strike, upper_strike, gearing=1.0):
    """payoff_bull_spread
    Buy call option with lower_strike :math:`K_{\mathrm{lower}}`
    and sell put option with upper_strike :math:`K_{\mathrm{upper}}`.
    As the name denotes, lower_strike is lower than upper_strike.
    Payoff formula is as follows:

    .. math::

        g(\max(S - K_{\mathrm{lower}}, 0) - \min(S - K_{\mathrm{upper}}, 0))
        = g\min(K_{\mathrm{upper}}, \max(S - K_{\mathrm{lower}}))

    where :math:`S` is underlying, :math:`g` is gearing.

    :param float underlying:
    :param float lower_strike:
    :param float upper_strike:
    :param float gearing: coefficient of this option. Default value is 1.
    :return: payoff of bull spread option.
        If lower_strike >= upper_strike, then return 0.
    :rtype: float
    """
    if lower_strike >= upper_strike:
        return 0.0
    return gearing * min(upper_strike, max(underlying - lower_strike, 0))


def payoff_bull_spread_fprime(
        underlying,
        lower_strike,
        upper_strike,
        gearing=1.0):
    """payoff_bull_spread_fprime
    calculate value of derivative of payoff of bull spread option
    with respect to underlying.

    :param underlying:
    :param lower_strike:
    :param upper_strike:
    :param float gearing: coefficient of this option. Default value is 1.
    :return: derivative of bull spread option.
        If lower_strike >= upper_strike, then return 0.
    :rtype: float
    """
    if lower_strike >= upper_strike:
        return 0.0
    if lower_strike < underlying < upper_strike:
        return gearing
    else:
        return 0.0


def payoff_straddle(underlying, strike, gearing=1.0):
    """payoff_straddle
    Buy call option and put option at same time with same strike.

    :param float underlying:
    :param float strike:
    :param float gearing: Coefficient of this option. Default value is 1.
    """
    return (payoff_call(underlying, strike, gearing)
            + payoff_put(underlying, strike, gearing))


def payoff_strangle(underlying, lower_strike, upper_strike, gearing=1.0):
    """payoff_strangle
    Buy call option and put option at same time with different strike.

    :param float underlying:
    :param float lower_strike:
    :param float upper_strike:
    :param float gearing: Coefficient of this option. Default value is 1.
    :return: payoff of strangle option.
        If lower_strike >= upper_strike, then return 0.
    :rtype: float
    """
    if lower_strike >= upper_strike:
        return 0.0
    return (payoff_put(underlying, lower_strike, gearing)
            + payoff_call(underlying, upper_strike, gearing))


def payoff_butterfly_spread(underlying, spot_price, spread, gearing=1.0):
    """payoff_butterfly_spread
    Butterfly option consists of following options:

    * Buy 1 call with a (spot_price - spread)
    * Sell 2 calls with a spread
    * Buy 1 call with (spot_price + spread)

    :param float underlying:
    :param float spot_price: spot price of underlying.
    :param float spread: non-negative value is required.
        If value is 0, this is same as straddle.
    :param float gearing: coefficient of this option. Default value is 1.
    :return: payoff of butterfly spread option.
        If spot_price is negative, return 0.
    :rtype: float
    """
    if spread < 0.0:
        return 0.0
    return (payoff_call(underlying, spot_price - spread, gearing)
            - 2.0 * payoff_call(underlying, spot_price, gearing)
            + payoff_call(underlying, spot_price + spread, gearing))


def payoff_risk_reversal(underlying, lower_strike, upper_strike, gearing=1.0):
    """payoff_risk_reversal

    * Sell 1 out of the money put option.
    * Buy 1 out of the money call option.

    :param float underlying:
    :param float lower_strike:
    :param float upper_strike:
    :param float gearing: Coefficient of this option. Default value is 1.
    :return: payoff of risk reversal option.
        If lower_strike > upper_strike, then return 0.
        If lower_strike = upper_strike, this is called synthetic forward.
    :rtype: float
    """
    return (-payoff_put(underlying, lower_strike, gearing)
            + payoff_call(underlying, upper_strike, gearing))


class PayoffHelper(object):
    """PayoffFactory
    generate payoff function as a single variable.
    derivative is with respect to the variable.
    """

    def make_func(self):
        raise NotImplementedError

    def make_fprime(self):
        raise NotImplementedError

    def make_fhess(self):
        raise NotImplementedError


class CallStrikePayoffHelper(PayoffHelper):
    """CallStrikePayoffHelper
    generate call option payoff as a signle variable funciton of strike.
    """

    def __init__(self, **params):
        """__init__

        :param **params: underlying and gearing is required as key.
        """
        keys = ["underlying", "gearing"]
        self.params = util.check_keys(keys, params, locals())

    def make_func(self):
        return lambda strike: payoff_call(
            self.params["underlying"], strike, self.params["gearing"])

    def make_fprime(self):
        return lambda strike: payoff_call_fprime(
            self.params["underlying"], strike, self.params["gearing"])

    def make_fhess(self):
        return ValueError("dirac delta function")


class CallUnderlyingPayoffHelper(PayoffHelper):
    """CallUnderlyingPayoffHelper
    generate call option payoff as a single variable funciton of underlying.
    """

    def __init__(self, **params):
        """__init__

        :param **params: strike and gearing is required as key.
        """
        keys = ["strike", "gearing"]
        self.params = util.check_keys(keys, params, locals())

    def make_func(self):
        """make_func
        """
        return functools.partial(payoff_call, **self.params)

    def make_fprime(self):
        """make_fprime
        """
        return functools.partial(payoff_call_fprime, **self.params)

    def make_fhess(self):
        """make_fhess
        """
        raise ValueError("dirac delta function")


class BullSpreadUnderlyingPayoffHelper(PayoffHelper):
    """BullSpreadUnderlyingPayoffHelper
    generate bull-spread option payoff as a funciton of underlying.

    :param float lower_strike: required.
    :param float upper_strike: required.
    :param float gearing: required.
    """

    def __init__(self, **params):
        """__init__
        """
        keys = ["lower_strike", "upper_strike", "gearing"]
        self.params = util.check_keys(keys, params, locals())
        if self.params["lower_strike"] > self.params["upper_strike"]:
            raise ValueError("lower strike must be greater than upper strike")

    def make_func(self):
        """make_func
        """
        return functools.partial(payoff_bull_spread, **self.params)

    def make_fprime(self):
        """make_fprime
        """
        return functools.partial(payoff_bull_spread_fprime, **self.params)

    def make_fhess(self):
        """make_fhess
        """
        raise ValueError("dirac delta function")
