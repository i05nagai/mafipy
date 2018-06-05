"""
mafipy: A mathematical finance in python
=========================================

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from . import calibrator
from . import function
from . import math
from . import replication

__all__ = [
    "calibrator",
    "function",
    "replication",
    'math',
]
