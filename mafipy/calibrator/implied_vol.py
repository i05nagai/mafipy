#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import numpy as np
import scipy.optimize

import mafipy.function


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
        return (mafipy.function.black_scholes_call_formula(
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
        return (mafipy.function.black_payers_swaption_value(
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


