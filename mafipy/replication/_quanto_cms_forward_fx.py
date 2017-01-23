#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import math
import scipy

from mafipy import function


def _calc_h(swap_rate_cdf, swap_rate, min_prob=1e-16, max_prob=0.99999):
    """_calc_h
    calculates following value:

    .. math::
        h(s) := \Phi^{-1}(\Phi^{A}(s))

    where
    :math:`\Phi^{-1}(\cdot)` is inverse function of standard gaussian c.d.f,
    :math:`\Phi^{A}(s)` is c.d.f. of swap rate under annuity measure.

    There is no meaning of this value
    because of simplification to calculate forward fx diffusion
    See :py:func:`_forward_fx_diffusion`.

    :param float swap_rate_cdf:
    :param float swap_rate:
    :param float min_prob: lower bound of swap rate probability.
        Default value is 1e-16.
    :param float max_prob: upper bound of swap rate probability.
        Default value is 0.99999.

    :return: :math:`\Phi^{-1}(\Phi^{A}(s))`.
    :rtype: float
    """
    prob = swap_rate_cdf(swap_rate)
    prob = max(min_prob, prob)
    prob = min(max_prob, prob)

    return scipy.stats.norm.ppf(prob)


def _calc_h_fprime(swap_rate_pdf, swap_rate, h):
    """_calc_h_fprime
    calculates derivative of :py:func:`_calc_h`.

    .. math::
        h^{\prime}(s)
        = \\frac{1}{\phi(h(s))} \psi^{A}(s),

    where
    :math:`\phi^{-1}(\cdot)` is inverse function of standard gaussian p.d.f.,
    :math:`\psi^{A}(s)` is p.d.f. of swap rate under annuity measure,
    :math:`s` is swap_rate.

    There is no meaning of this value
    because of simplification to calculate forward fx diffusion
    See :py:func:`_forward_fx_diffusion_fprime`.

    :param float swap_rate_pdf:
    :param float swap_rate:
    :param float h: value of :math:`h(s)`.
    :return: :math:`h^{\prime}(s)`.
    :rtype: float
    """
    norm = scipy.stats.norm

    return swap_rate_pdf(swap_rate) / norm.pdf(h)


def _calc_h_fhess(swap_rate_pdf_fprime, swap_rate_pdf, swap_rate, h, h_fprime):
    """_calc_h_fhess
    calculates second derivative of :py:func:`_calc_h`.

    .. math::
        h^{\prime\prime}(s)
        & = & \\frac{
            (\psi^{A})^{\prime}(s) \phi(h(s))
            - \psi^{A}(s) \phi^{\prime}(h(s)) h^{\prime}(s)
        }{
            \phi(h(s))^{2}
        }

    where
    :math:`\phi^{-1}(\cdot)` is inverse function of standard gaussian p.d.f.,
    :math:`\psi^{A}(s)` is p.d.f. of swap rate under annuity measure,
    :math:`s` is swap_rate.

    There is no meaning of this value
    because of simplification to calculate forward fx diffusion
    See :py:func:`_forward_fx_diffusion_fhess`.

    :param float swap_rate_pdf:
    :param float swap_rate:
    :param float h: value of :math:`h(s)`.
    :param float h_prime: value of :math:`h^{\prime}(s)`.
    :return: :math:`h^{\prime\prime}(s)`.
    :rtype: float
    """
    norm = scipy.stats.norm
    h_term1 = swap_rate_pdf_fprime(swap_rate) * norm.pdf(h)
    h_term2 = (swap_rate_pdf(swap_rate)
               * function.norm_pdf_fprime(h) * h_fprime)
    denominator = norm.pdf(h) ** 2
    return (h_term1 - h_term2) / denominator


def _forward_fx_diffusion(
        swap_rate,
        time,
        vol,
        corr,
        swap_rate_cdf,
        swap_rate_pdf,
        swap_rate_pdf_fprime,
        is_inverse):
    """_forward_fx_diffusion
    calculate following value:

    .. math::

        \\tilde{\chi}(s)
        := \exp
            \left(
                \\rho_{XS}\sigma_{X}\sqrt{T}\Phi^{-1}(\Psi^{A}(s))
                    + \\frac{\sigma_{X}^{2}T}{2}(1 - \\rho_{XS}^{2})
            \\right),

    where
    :math:`s` is swap_rate,
    :math:`\\rho_{XS}` is corr,
    :math:`\sigma_{X}` is vol,
    :math:`\Psi^{A}(\cdot)` is c.d.f. of swap_rate under annuity measure.

    :param float swap_rate:
    :param float time:
    :param float vol: must be positive. volatility.
    :param float corr: must be within [-1, 1]. correlation.
    :param callable swap_rate_cdf: c.d.f. of foward swap rate
        under annuity measure.
    :param callable swap_rate_pdf: not used. p.d.f. of foward swap rate
        under annuity measure.
    :param callable swap_rate_pdf_fprime: not used. derivative of p.d.f.
        of foward swap rate under annuity measure.
    :param bool is_inverse:

    :return: value of forward fx diffusion.
    :rtype: float
    """
    assert(vol > 0.0)
    assert(-1.0 <= corr <= 1.0)
    h = _calc_h(swap_rate_cdf, swap_rate)
    term1 = corr * vol * math.sqrt(time) * h
    term2 = vol * vol * time * (1.0 - corr * corr) * 0.5
    if is_inverse:
        return math.exp(-term1 + term2)
    else:
        return math.exp(term1 + term2)


def _forward_fx_diffusion_fprime(
        swap_rate,
        time,
        vol,
        corr,
        swap_rate_cdf,
        swap_rate_pdf,
        swap_rate_pdf_fprime,
        is_inverse):
    """_forward_fx_diffusion_fprime
    derivative of forward fx diffusion.
    See :py:func:`_forward_fx_diffusion`.

    .. math::

        \\tilde{\chi}^{\prime}(s)
            & = & \\rho_{XS}\sigma_{X}\sqrt{T}h'(s) \\tilde{\chi}(s)

    where
    :math:`h(s) := \Phi^{-1}(\Psi^{A}(s))`,
    :math:`\\rho_{XS}` is corr,
    :math:`\sigma_{X}` is vol.

    See :py:func:`_forward_fx_diffusion`.

    :param float swap_rate:
    :param float time:
    :param float vol: must be positive. volatility
    :param float corr: must be within [-1, 1]. correlation.
    :param callable swap_rate_cdf: c.d.f. of foward swap rate
        under annuity measure.
    :param callable swap_rate_pdf: not used. p.d.f. of foward swap rate
        under annuity measure.
    :param callable swap_rate_pdf_fprime: not used. derivative of p.d.f.
        of foward swap rate under annuity measure.
    :param bool is_inverse:

    :return: value of derivative of forward fx diffusion.
    :rtype: float
    """
    assert(vol > 0.0)
    assert(-1.0 <= corr <= 1.0)
    forward_fx_diffusion = _forward_fx_diffusion(
        swap_rate,
        time,
        vol,
        corr,
        swap_rate_cdf,
        swap_rate_pdf,
        swap_rate_pdf_fprime,
        is_inverse)
    h = _calc_h(swap_rate_cdf, swap_rate)
    h_fprime = _calc_h_fprime(swap_rate_pdf, swap_rate, h)
    value = corr * vol * math.sqrt(time) * h_fprime * forward_fx_diffusion
    if is_inverse:
        return -value
    else:
        return value


def _forward_fx_diffusion_fhess(
        swap_rate,
        time,
        vol,
        corr,
        swap_rate_cdf,
        swap_rate_pdf,
        swap_rate_pdf_fprime,
        is_inverse):
    """_forward_fx_diffusion_fhess
    Calculate second derivative of diffusion part of forward FX.
    See :py:func:`_forward_fx_diffusion`
    and :py:func:`_forward_fx_diffusion_fprime`.

    .. math::
        \\tilde{\chi}^{\prime\prime}(s)
            & = & \\rho_{XS}\sigma_{X}\sqrt{T}
            \left(
               h^{\prime\prime}(s)\\tilde{\chi}(s)
                + \\rho_{XS}\sigma_{X}\sqrt{T}
                    h^{\prime}(s)^{2} \\tilde{\chi}(s)
            \\right)

    :param float swap_rate:
    :param float time: non-negative.
    :param float vol: non-negative.
    :param float corr: correlatiion. value must be from -1 to 1.
    :param callable swap_rate_cdf: distribution of forward swap
        rate under annuity measure
    :param cllable swap_rate_pdf: probability density of forwrad swap rate
        under annuity measure
    :param callable swap_rate_pdf_fprime: derivative of probability density of
        forward swap rate under annuity measure.
    :param bool is_inverse:

    :return: second derivative of diffusion part of forward FX.
    :rtype: float
    """
    assert(time >= 0.0)
    assert(vol >= 0.0)
    assert(-1.0 <= corr <= 1.0)

    h = _calc_h(swap_rate_cdf, swap_rate)
    h_fprime = _calc_h_fprime(swap_rate_pdf, swap_rate, h)
    h_fhess = _calc_h_fhess(
        swap_rate_pdf_fprime, swap_rate_pdf, swap_rate, h, h_fprime)
    # fhess
    forward_fx_diffusion = _forward_fx_diffusion(
        swap_rate,
        time,
        vol,
        corr,
        swap_rate_cdf,
        swap_rate_pdf,
        swap_rate_pdf_fprime,
        is_inverse)
    factor1 = corr * vol * math.sqrt(time) * forward_fx_diffusion
    if is_inverse:
        factor2 = -h_fhess + corr * vol * math.sqrt(time) * (h_fprime ** 2)
    else:
        factor2 = h_fhess + corr * vol * math.sqrt(time) * (h_fprime ** 2)
    return factor1 * factor2


class _ForwardFxDiffusionHelper(object):
    """_ForwardFxDiffusionHelper
    Generate forward FX diffusion under simple qunato cms model.
    This helper class makes forward FX diffusion,
    differential of the diffusion
    and second derivative of the diffusion as a function of swap rate.

    :param float time:
    :param float vol: must be positive. volatility.
    :param float corr: must be within [-1, 1]. correlation.
    :param callable swap_rate_cdf:
    :param callable swap_rate_pdf:
    :param callable swap_rate_pdf_fprime:
    :param bool is_inverse:
    """

    def __init__(self,
                 time,
                 vol,
                 corr,
                 swap_rate_cdf,
                 swap_rate_pdf,
                 swap_rate_pdf_fprime,
                 is_inverse):
        """__init__
        """
        assert(time >= 0.0)
        assert(vol >= 0.0)
        assert(-1.0 <= corr <= 1.0)
        self.time = time
        self.vol = vol
        self.corr = corr
        self.swap_rate_cdf = swap_rate_cdf
        self.swap_rate_pdf = swap_rate_pdf
        self.swap_rate_pdf_fprime = swap_rate_pdf_fprime
        self.is_inverse = is_inverse

    def make_func(self):
        """make_func
        makes forward FX diffusion as a function of swap rate.
        See :py:func:`_forward_fx_diffusion`.
        """
        return lambda swap_rate: _forward_fx_diffusion(
            swap_rate=swap_rate,
            time=self.time,
            vol=self.vol,
            corr=self.corr,
            swap_rate_cdf=self.swap_rate_cdf,
            swap_rate_pdf=self.swap_rate_pdf,
            swap_rate_pdf_fprime=self.swap_rate_pdf_fprime,
            is_inverse=self.is_inverse)

    def make_fprime(self):
        """make_fprime
        makes derivative of forward FX diffusion as a function of swap rate.
        See :py:func:`_forward_fx_diffusion_fprime`.
        """
        return lambda swap_rate: _forward_fx_diffusion_fprime(
            swap_rate=swap_rate,
            time=self.time,
            vol=self.vol,
            corr=self.corr,
            swap_rate_cdf=self.swap_rate_cdf,
            swap_rate_pdf=self.swap_rate_pdf,
            swap_rate_pdf_fprime=self.swap_rate_pdf_fprime,
            is_inverse=self.is_inverse)

    def make_fhess(self):
        """make_fhess
        makes second derivative of forward FX diffusion
        as a function of swap rate.
        See :py:func:`_forward_fx_diffusion_fhess`.
        """
        return lambda swap_rate: _forward_fx_diffusion_fhess(
            swap_rate=swap_rate,
            time=self.time,
            vol=self.vol,
            corr=self.corr,
            swap_rate_cdf=self.swap_rate_cdf,
            swap_rate_pdf=self.swap_rate_pdf,
            swap_rate_pdf_fprime=self.swap_rate_pdf_fprime,
            is_inverse=self.is_inverse)
