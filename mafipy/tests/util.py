import random


def get_real(size=1, min_range=0.0, max_range=1.0):
    return [random.uniform(min_range, max_range) for i in range(size)]


def get_real_t(size=1, min_range=0.0, max_range=1.0):
    return tuple(get_real(size, min_range, max_range))
