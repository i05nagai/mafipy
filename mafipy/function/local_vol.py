#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import numpy as np


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
