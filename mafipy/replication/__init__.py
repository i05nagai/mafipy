"""
=======================================
replication (:mod:`mafipy.replication`)
=======================================

.. currentmodule:: mafipy.replication

Quanto CMS
==================

Black swaption model
--------------------

.. autosummary::
    :toctree: generated/

    make_pdf_black_swaption
    make_pdf_fprime_black_swaption
    make_cdf_black_swaption

Black Scholes model
-------------------

.. autosummary::
    :toctree: generated/

    make_pdf_black_swaption
    make_pdf_fprime_black_scholes
    make_cdf_black_scholes

Pricer
---------

.. autosummary::
    :toctree: generated/

    replicate
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .pricer_quanto_cms import *
from .replication_method import *

__all__ = [s for s in dir() if not s.startswith('_')]
