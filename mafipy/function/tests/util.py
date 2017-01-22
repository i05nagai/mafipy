from __future__ import division, print_function, absolute_import
import random


def get_real(size=1, min_range=0.0, max_range=1.0):
    return [random.uniform(min_range, max_range) for i in range(size)]


def get_real_t(size=1, min_range=0.0, max_range=1.0):
    return tuple(get_real(size, min_range, max_range))


def get(min_range=0.0, max_range=1.0):
    return random.uniform(min_range, max_range)


def _to_bool(value):
    if value > 0.5:
        return True
    else:
        return False


def get_bool(size=1):
    return [_to_bool(v) for v in get_real(size)]
