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

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = '0.1.dev12'

__all__ = [
    'calibrator',
    'function',
    'math',
    'replication',
]
