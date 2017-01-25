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

    black_payers_swaption_value - black payer's swaption formula.
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

Local Volatility model
======================
To be displayed.

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
"""

from __future__ import division, print_function, absolute_import

from .analytic_formula import *
from .black import *
from .black_scholes import *
from .local_vol import *
from .math_formula import *
from .payoff import *
from .sabr import *

__all__ = [s for s in dir() if not s.startswith('_')]
