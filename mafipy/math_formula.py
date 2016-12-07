#!/bin/python
# -*- coding: utf-8 -*-

import scipy.stats


def norm_pdf_fprime(x):
    """norm_pdf_fprime

    :param x:
    :return: value of derivative of p.d.f.
    :rtype: float
    """
    return -x * scipy.stats.norm.pdf(x)

