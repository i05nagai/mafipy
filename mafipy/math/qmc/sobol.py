from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mafipy.math.qmc._sobol as _sobol


def make_sobol(dimension, seq='sobol'):
    return _sobol.make_sobol(dimension)
