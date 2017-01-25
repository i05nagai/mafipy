"""
=======================================
replication (:mod:`mafipy.replication`)
=======================================

.. currentmodule:: mafipy.replication

"""

from __future__ import division, print_function, absolute_import

from .pricer_quanto_cms import *
from .replication_method import *

__all__ = [s for s in dir() if not s.startswith('_')]
