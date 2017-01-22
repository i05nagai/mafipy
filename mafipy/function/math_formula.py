#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import scipy.stats


def norm_pdf_fprime(x):
    """norm_pdf_fprime

    :param x:
    :return: value of derivative of p.d.f.
    :rtype: float
    """
    return -x * scipy.stats.norm.pdf(x)


def norm_pdf_fhess(x):
    """norm_pdf_fprime
    Second derivative of p.d.f. of standard normal distribution.
    See :py:func:`norm_pdf_fprime`.

    .. math::
        \phi^{\prime}(x) = (x^{2} - 1)\phi(x)

    where
    :math:`\phi(\cdot)` is p.d.f. of standard normal distribution.

    :param x:
    :return: value of derivative of p.d.f.
    :rtype: float
    """
    density = scipy.stats.norm.pdf(x)
    return (x * x - 1.0) * density
