"""
====================================
function (:mod:`mafipy.function`)
====================================

.. currentmodule:: mafipy.function

Black Model
===========

Basics
------

.. autosummary::
    :toctree: generated/

    black_payers_swaption_value
    black_receivers_swaption_value
    black_payers_swaption_value_fprime_by_strike
    black_payers_swaption_value_fhess_by_strike
    black_payers_swaption_value_third_by_strike

Greeks
------

.. autosummary::
    :toctree: generated/

    black_payers_swaption_delta
    black_payers_swaption_vega
    black_payers_swaption_volga
    black_payers_swaption_vega_fprime_by_strike

Distribution
------------

.. autosummary::
    :toctree: generated/

    black_swaption_cdf
    black_swaption_pdf

Black Scholes Model
===================

Miscs
-----

.. autosummary::
    :toctree: generated/

    func_d1
    func_d2
    d_fprime_by_strike
    d_fhess_by_strike

Basics
------

.. autosummary::
    :toctree: generated/

    black_scholes_call_value
    black_scholes_put_value
    black_scholes_call_value_fprime_by_strike
    black_scholes_call_value_fhess_by_strike
    black_scholes_call_value_third_by_strike

Greeks
------

.. autosummary::
    :toctree: generated/

    black_scholes_call_delta
    black_scholes_call_gamma
    black_scholes_call_vega
    black_scholes_call_volga
    black_scholes_call_theta
    black_scholes_call_rho
    black_scholes_call_vega_fprime_by_strike

Distribution
------------

.. autosummary::
    :toctree: generated/

    black_scholes_cdf
    black_scholes_pdf

Local Volatility model
======================
To be displayed.


Payoff functions
================

.. autosummary::
    :toctree: generated/

    payoff_call
    payoff_call_fprime
    payoff_put
    payoff_put_fprime
    payoff_bull_spread
    payoff_bull_spread_fprime
    payoff_straddle
    payoff_strangle
    payoff_butterfly_spread
    payoff_risk_reversal

SABR model
==========

Basics
------

.. autosummary::
    :toctree: generated/

    sabr_payers_swaption_value
    sabr_receivers_swaption_value
    sabr_implied_vol_hagan
    sabr_atm_implied_vol_hagan

Greeks
------

.. autosummary::
    :toctree: generated/

    sabr_implied_vol_hagan_fprime_by_strike
    sabr_implied_vol_hagan_fhess_by_strike
    sabr_implied_vol_hagan_fprime_by_underlying
    sabr_implied_vol_hagan_fhess_by_underlying
    sabr_payers_swaption_delta

Distribution
------------

.. autosummary::
    :toctree: generated/

    sabr_cdf
    sabr_pdf

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .analytic_formula import *
from .black import *
from .black_scholes import *
from .local_vol import *
from .math_formula import *
from .payoff import *
from .sabr import *

__all__ = [s for s in dir() if not s.startswith('_')]
