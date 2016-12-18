#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import math
import numpy as np
import scipy
from . import analytic_formula
from . import math_formula
from . import payoff
from . import replication


def make_pdf_black_scholes(
        underlying,
        rate,
        maturity,
        vol):
    """make_pdf_black_scholes
    return p.d.f. of black scholes model

    .. math::
        \Phi_{S}(k) := \\frac{\partial}{\partial K^{2}} c(0, S; T, K)

    where
    :math:`S` is underlying,
    :math:`T` is maturity,
    :math:`K` is strike,
    :math:`\Phi_{S}(k)` is p.d.f. of :math:`S`,
    :math:`c(0, S; T, K)` is value of call option
    with strike :math:`K` and :math:`T` maturity at time 0.

    :param float underlying:
    :param float rate:
    :param float maturity: non-negative.
    :param float vol: volatility. non-negative.
    :return: p.d.f. of call price under black scholes model
        as a function of strike.
    :rtype: function with strike.
    """
    assert(maturity >= 0.0)
    assert(vol >= 0.0)

    def pdf(strike):
        return analytic_formula.black_scholes_call_value_fhess_by_strike(
            underlying=underlying,
            strike=strike,
            rate=rate,
            maturity=maturity,
            vol=vol)

    return pdf


def make_pdf_fprime_black_scholes(
        underlying,
        rate,
        maturity,
        vol):
    """make_pdf_fprime_black_scholes_model
    return first derivative of p.d.f. of black scholes.
    See :py:func:`make_pdf_black_scholes`.

    .. math::
        \Phi_{S}^{\prime}(k)
        := \\frac{\partial}{\partial K^{2}} c(0, S; T, K)

    where
    :math:`S` is underlying,
    :math:`T` is maturity,
    :math:`K` is strike,
    :math:`\Phi_{S}(k)` is p.d.f. of :math:`S`,
    :math:`c(0, S; T, K)` is value of call option
    with strike :math:`K` and :math:`T` maturity at time 0.

    :param float underlying:
    :param float rate:
    :param float maturity: non-negative.
    :param float vol: volatility. non-negative.
    :return: first derivative of p.d.f. of call price under black scholes model
        as a function of strike.
    :rtype: function
    """
    assert(maturity >= 0.0)
    assert(vol >= 0.0)

    def pdf_fprime(strike):
        return analytic_formula.black_scholes_call_value_third_by_strike(
            underlying=underlying,
            strike=strike,
            rate=rate,
            maturity=maturity,
            vol=vol)

    return pdf_fprime


def make_cdf_black_scholes(
        underlying,
        rate,
        maturity,
        vol):
    """make_cdf_black_scholes
    returns c.d.f. of black scholes model.

    :param underlying:
    :param rate:
    :param maturity:
    :param vol:
    :return: distribution of call price under black scholes model
        as a funciton of strike.
    :rtype: function
    """
    assert(maturity >= 0.0)
    assert(vol >= 0.0)
    return lambda strike: (
        1.0 + analytic_formula.black_scholes_call_value_fprime_by_strike(
            underlying, strike, rate, maturity, vol))


def _calc_h(swap_rate_cdf, swap_rate):
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
    :return: :math:`\Phi^{-1}(\Phi^{A}(s))`.
    :rtype: float
    """
    norm = scipy.stats.norm

    return norm.ppf(swap_rate_cdf(swap_rate))


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

    try:
        h_fprime = swap_rate_pdf(swap_rate) / norm.pdf(h)
    except RuntimeWarning:
        print("swap_rate:", swap_rate,
              "swap_rate_pdf(swap_rate):", swap_rate_pdf(swap_rate),
              "h:", h,
              "norm.pdf(h):", norm.pdf(h))
    finally:
        return h_fprime


def _calc_h_fhess(swap_rate_pdf_fprime, swap_rate_pdf, swap_rate, h, h_fprime):
    """_calc_h_fhess
    calculates second derivative of :py:func:`_calc_h`.

    .. math::
        h^{\prime\prime}(s)
        & = & \\frac{
            (\psi^{A})^{\prime}(s) \phi(h(s)) - \psi^{A}(s) \phi^{\prime}(h(s)) h^{\prime}(s)
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
               * math_formula.norm_pdf_fprime(h) * h_fprime)
    denominator = norm.pdf(h) ** 2
    return (h_term1 - h_term2) / denominator


def _forward_fx_diffusion(
        swap_rate,
        time,
        vol,
        corr,
        swap_rate_cdf,
        swap_rate_pdf,
        swap_rate_pdf_fprime):
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
    :return: value of forward fx diffusion.
    :rtype: float
    """
    assert(vol > 0.0)
    assert(-1.0 <= corr <= 1.0)
    h = _calc_h(swap_rate_cdf, swap_rate)
    term1 = corr * vol * math.sqrt(time) * h
    term2 = vol * vol * time * (1.0 - corr * corr) * 0.5
    return math.exp(term1 + term2)


def _forward_fx_diffusion_fprime(
        swap_rate,
        time,
        vol,
        corr,
        swap_rate_cdf,
        swap_rate_pdf,
        swap_rate_pdf_fprime):
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
        swap_rate_pdf_fprime)
    h = _calc_h(swap_rate_cdf, swap_rate)
    h_fprime = _calc_h_fprime(swap_rate_pdf, swap_rate, h)
    return corr * vol * math.sqrt(time) * h_fprime * forward_fx_diffusion


def _forward_fx_diffusion_fhess(
        swap_rate,
        time,
        vol,
        corr,
        swap_rate_cdf,
        swap_rate_pdf,
        swap_rate_pdf_fprime):
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
        swap_rate_pdf_fprime)
    factor1 = corr * vol * math.sqrt(time) * forward_fx_diffusion
    factor2 = h_fhess + corr * vol * math.sqrt(time) * (h_fprime ** 2)
    return factor1 * factor2


class _ForwardFxDiffusionHelper(object):
    """_ForwardFxDiffusionHelper
    Generate forward FX diffusion under simple qunato cms model.
    This helper class makes forward FX diffusion,
    differential of the diffusion
    and second derivative of the diffusion as a function of swap rate.
    """

    def __init__(self,
                 time,
                 vol,
                 corr,
                 swap_rate_cdf,
                 swap_rate_pdf,
                 swap_rate_pdf_fprime):
        """__init__

        :param float time:
        :param float vol: must be positive. volatility.
        :param float corr: must be within [-1, 1]. correlation.
        :param callable swap_rate_cdf:
        :param callable swap_rate_pdf:
        :param callable swap_rate_pdf_fprime:
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
            swap_rate_pdf_fprime=self.swap_rate_pdf_fprime)

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
            swap_rate_pdf_fprime=self.swap_rate_pdf_fprime)

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
            swap_rate_pdf_fprime=self.swap_rate_pdf_fprime)


class SimpleQuantoCmsHelper(object):
    """SimpleQuantoCmsHelper
    Interface of helper class to generate integrands of replication method
    to expectation of numerator and denominator.
    See :py:func:`SimpleQuantoCmsPricer`.
    The integrands are part of terms of stochastic weight.
    Stochastic weight of numerator are

    .. math::
        \\begin{eqnarray*}
            w(s)
                & := &
                    \\frac{d^{2} }{d s^{2}} (g(s) \\alpha(s) \\tilde{\chi}(s))
                \\\\
                & = &
                g^{\prime\prime}(s)\\alpha(s)\\tilde{\chi}(s)
                    + g(s)\\alpha^{\prime\prime}(s)\\tilde{\chi}(s)
                    + g(s)\\alpha(s)\\tilde{\chi}^{\prime\prime}(s)
                \\\\
                & &
                    + 2g^{\prime}(s)\\alpha^{\prime}(s)\\tilde{\chi}(s)
                    + 2g^{\prime}(s)\\alpha(s)\\tilde{\chi}^{\prime}(s)
                    + 2g(s)\\alpha^{\prime}(s)\\tilde{\chi}^{\prime}(s).
        \end{eqnarray*}

    where
    :math:`g` is payoff function,
    :math:`\\alpha` is annuity mapping function,
    :math:`\\tilde{\chi}(s)` is forward fx diffusion,
    see :py:func:`_forward_fx_diffusion`.

    The stochastic weight of denominator is

    .. math::
        \\begin{eqnarray*}
            w(s)
                & := &
                    \\frac{d^{2} }{d s^{2}} (\\alpha(s) \\tilde{\chi}(s))
                \\\\
                & = &
                    \\alpha^{\prime\prime}(s)\\tilde{\chi}(s)
                    + \\alpha(s)\\tilde{\chi}^{\prime\prime}(s)
                    + 2\\alpha^{\prime}(s)\\tilde{\chi}^{\prime}(s).
        \end{eqnarray*}

    """

    def make_numerator_call_integrands(self, **kwargs):
        """make_numerator_call_integrands
        Interface of integrands which are integrated with call pricer
        in replication method in the numerator.
        """
        raise NotImplementedError

    def make_numerator_put_integrands(self, **kwargs):
        """make_numerator_put_integrands
        Interface of integrands which are integrated with put pricer
        in replication method in the numerator.
        """
        raise NotImplementedError

    def make_numerator_analytic_funcs(self, **kwargs):
        """make_numerator_analytic_funcs
        Interface of analytically computable terms in replication method.
        The terms depedend on the model of forward FX rate
        and forward swap rate.
        """
        raise NotImplementedError

    def make_denominator_call_integrands(self, **kwargs):
        """make_denominator_call_integrands
        Interface of integrands which are integrated with call pricer
        in replication method in the denominator.
        """
        raise NotImplementedError

    def make_denominator_put_integrands(self, **kwargs):
        """make_denominator_put_integrands
        Interface of integrands which are integrated with put pricer
        in replication method in the denominator.
        """
        raise NotImplementedError

    def make_denominator_analytic_funcs(self, **kwargs):
        """make_denominator_analytic_funcs
        Interface of analytically computable terms in replication method.
        The terms depedend on the model of forward FX rate
        and forward swap rate.
        """
        raise NotImplementedError


class _SimpleQuantoCmsLinearCallHelper(SimpleQuantoCmsHelper):
    """_SimpleQuantoCmsLinearCallHelper
    In linear TSR model with call payoff,
    stochastic weight in the numerator is

        .. math::
            \\begin{eqnarray}
                g_{\mathrm{call}}^{\prime\prime}(s; K)\\alpha(s)\\tilde{\chi}(s)
                    & = &
                    \delta(s - K)\\alpha(s)\\tilde{\chi}(s),
                \\\\
                g_{\mathrm{call}}(s; K)\\alpha(s)\\tilde{\chi}^{\prime\prime}(s)
                    & = &
                    (s - K)^{+}
                    (\\alpha_{1}s + \\alpha_{2})
                    \\rho_{XS}\sigma_{X}\sqrt{T}
                    \left(
                       h^{\prime\prime}(s)\\tilde{\chi}(s)
                        + \\rho_{XS}\sigma_{X}\sqrt{T} h^{\prime}(s)^{2} \\tilde{\chi}(s)
                    \\right),
                \\\\
                2g_{\mathrm{call}}^{\prime}(s; K)\\alpha^{\prime}(s)\\tilde{\chi}(s)
                    & = & 2 1_{[K, \infty)}(s) \\alpha_{1}
                        \exp
                        \left(
                            \\rho_{XS}\sigma_{X}\sqrt{T}\Phi^{-1}(\Psi^{A}(s))
                                + \\frac{\sigma_{X}^{2}T}{2}(1 - \\rho_{XS}^{2})
                        \\right),
                \\\\
                2g_{\mathrm{call}}^{\prime}(s; K)\\alpha(s)\\tilde{\chi}^{\prime}(s)
                    & = & 2 1_{[K, \infty)}(s)
                        (\\alpha_{1}s + \\alpha_{2})
                        \\rho_{XS}\sigma_{X}\sqrt{T}h^{\prime}(s)\\tilde{\chi}(s),
                \\\\
                2g_{\mathrm{call}}(s; K)\\alpha^{\prime}(s)\\tilde{\chi}^{\prime}(s)
                    & = &
                        2(s - K)^{+} \\alpha_{1}
                            \\rho_{XS}\sigma_{X}\sqrt{T}h^{\prime}(s)\\tilde{\chi}(s),
            \end{eqnarray}

    where
    :math:`g_{\mathrm{callplet}}` is payoff function,
    :math:`\\alpha` is annuity mapping function,
    :math:`\\tilde{\chi}(s)` is forward fx diffusion,
    see :py:func:`_forward_fx_diffusion`.

    Stochastic weight in the denominator is

    .. math::
        \\begin{eqnarray}
            \\alpha^{\prime\prime}(s)\\tilde{\chi}(s)
                & = & 0
                \\\\
            \\alpha(s)\\tilde{\chi}^{\prime\prime}(s)
                & = &
                (\\alpha_{1}s + \\alpha_{2})
                \\rho_{XS}\sigma_{X}\sqrt{T}
                \left(
                       h^{\prime\prime}(s)\\tilde{\chi}(s)
                    + \\rho_{XS}\sigma_{X}\sqrt{T} h^{\prime}(s)^{2} \\tilde{\chi}(s)
                \\right),
                \\\\
            2\\alpha^{\prime}(s)\\tilde{\chi}^{\prime}(s)
                & = &
                    2\\alpha_{1}
                    \\rho_{XS}\sigma_{X}\sqrt{T}h^{\prime}(s)\\tilde{\chi}(s),
        \end{eqnarray}

    :param AnnuityMappingFuncHelper annuity_mapping_helper:
    :param PayoffHelper call_payoff_helper:
    :param _ForwardFxDiffusionHelper forward_fx_diffusion_helper:
    :param call_pricer: call option pricer.
    :param put_pricer: put option pricer.
    :param float payoff_strike: strike of call option payoff function.
    """

    def __init__(self,
                 annuity_mapping_helper,
                 call_payoff_helper,
                 forward_fx_diffusion_helper,
                 call_pricer,
                 put_pricer,
                 payoff_strike,
                 min_put_range=-np.inf,
                 max_put_range=np.inf,
                 min_call_range=-np.inf,
                 max_call_range=np.inf):
        """__init__
        """
        # annutiy mapping funciton
        self.annuity_mapping_func = annuity_mapping_helper.make_func()
        self.annuity_mapping_fprime = annuity_mapping_helper.make_fprime()
        self.annuity_mapping_fhess = annuity_mapping_helper.make_fhess()
        # payoff
        self.payoff_func = call_payoff_helper.make_func()
        self.payoff_fprime = call_payoff_helper.make_fprime()
        self.payoff_strike = payoff_strike
        # forawad fx diffusion
        fwd_fx_helper = forward_fx_diffusion_helper
        self.forward_fx_diffusion = fwd_fx_helper.make_func()
        self.forward_fx_diffusion_fprime = fwd_fx_helper.make_fprime()
        self.forward_fx_diffusion_fhess = fwd_fx_helper.make_fhess()
        # pricer
        self.call_pricer = call_pricer
        self.put_pricer = put_pricer
        # range
        self.min_put_range = min_put_range
        self.max_put_range = max_put_range
        self.min_call_range = min_call_range
        self.max_call_range = max_call_range

    def _make_numerator_integrands(self):
        """_make_numerator_integrands
        Returns following functions:

        .. math::
            g_{\mathrm{call}}(s; K)\\alpha(s)\\tilde{\chi}^{\prime\prime}(s)
            \\\\
            2g_{\mathrm{call}}^{\prime}(s; K)\\alpha^{\prime}(s)\\tilde{\chi}(s)
            \\\\
            2g_{\mathrm{call}}^{\prime}(s; K)\\alpha(s)\\tilde{\chi}^{\prime}(s)
            \\\\
            2g_{\mathrm{call}}(s; K)\\alpha^{\prime}(s)\\tilde{\chi}^{\prime}(s)

        :return: array of function.
        :rtype: array.
        """
        def func1(swap_rate):
            """func1
            """
            return (self.payoff_func(swap_rate)
                    * self.annuity_mapping_func(swap_rate)
                    * self.forward_fx_diffusion_fhess(swap_rate))

        def func2(swap_rate):
            return (2.0
                    * self.payoff_fprime(swap_rate)
                    * self.annuity_mapping_fprime(swap_rate)
                    * self.forward_fx_diffusion(swap_rate))

        def func3(swap_rate):
            return (2.0
                    * self.payoff_fprime(swap_rate)
                    * self.annuity_mapping_func(swap_rate)
                    * self.forward_fx_diffusion_fprime(swap_rate))

        def func4(swap_rate):
            return (2.0
                    * self.payoff_func(swap_rate)
                    * self.annuity_mapping_fprime(swap_rate)
                    * self.forward_fx_diffusion_fprime(swap_rate))

        return [func1, func2, func3, func4]

    def make_numerator_call_integrands(self, **kwargs):
        """make_numerator_call_integrands
        See :py:func:`_make_numerator_integrands`.

        :return: array of function.
        :rtype: array.
        """
        return self._make_numerator_integrands()

    def make_numerator_put_integrands(self, **kwargs):
        """make_numerator_put_integrands
        See :py:func:`_make_numerator_integrands`.

        :return: array of function.
        :rtype: array.
        """
        return self._make_numerator_integrands()

    def make_numerator_analytic_funcs(self, **kwargs):
        """make_numerator_analytic_funcs
        There are 3 functions which is calculated analytically.
        First is constant term in replication method.
        Second and third is the term
        which contains dirac delta function as integrands.
        Specifically, the second and third term are

        .. math::
            p(K) \\alpha(K)\\tilde{\chi}(K)
            1_{(L_{\min}^{\mathrm{put}},L_{\max}^{\mathrm{put}})}(K)
            \\\\
            c(K) \\alpha(K)\\tilde{\chi}(K)1_{(L_{\min},L_{\max})}(K)
            1_{(L_{\min}^{\mathrm{call}},L_{\max}^{\mathrm{call}})}(K)

        where
        :math:`p(K) := p(0, S(0); K, T)`,
        :math:`c(K) := c(0, S(0); K, T)`,
        :math:`L_{\min}^{\mathrm{put}}`
        is lower bound of integral range for put,
        :math:`L_{\max}^{\mathrm{put}}`
        is upper bound of integral range for put.
        :math:`L_{\min}^{\mathrm{call}}`
        is lower bound of integral range for call,
        :math:`L_{\max}^{\mathrm{call}}`
        is upper bound of integral range for call.

        This term contains dirac delta function
        (second derivative of payoff function)
        so that can be calculated analytically.

        :return: array of function.
        :rtype: array.
        """
        def func1(init_swap_rate):
            return (self.payoff_func(init_swap_rate)
                    * self.annuity_mapping_func(init_swap_rate)
                    * self.forward_fx_diffusion(init_swap_rate))

        def func2(init_swap_rate):
            return (self.put_pricer(self.payoff_strike)
                    * self.annuity_mapping_func(self.payoff_strike)
                    * self.forward_fx_diffusion(self.payoff_strike))

        def func3(init_swap_rate):
            return (self.call_pricer(self.payoff_strike)
                    * self.annuity_mapping_func(self.payoff_strike)
                    * self.forward_fx_diffusion(self.payoff_strike))

        terms = [func1]
        if self.min_put_range < self.payoff_strike < self.max_put_range:
            terms.append(func2)
        if self.min_call_range < self.payoff_strike < self.max_call_range:
            terms.append(func3)
        return terms

    def _make_denominator_integrands(self):
        """`_make_denominator_integrands`
        Returns following functions:

        .. math::
            \\alpha^{\prime\prime}(s)\\tilde{\chi}(s)
            \\\\
            \\alpha(s)\\tilde{\chi}^{\prime\prime}(s)
            \\\\
            2\\alpha^{\prime}(s)\\tilde{\chi}^{\prime}(s)

        :return: array of function.
        :rtype: array.
        """
        def func1(swap_rate):
            return (self.annuity_mapping_fhess(swap_rate)
                    * self.forward_fx_diffusion(swap_rate))

        def func2(swap_rate):
            return (self.annuity_mapping_func(swap_rate)
                    * self.forward_fx_diffusion_fhess(swap_rate))

        def func3(swap_rate):
            return (2.0
                    * self.annuity_mapping_fprime(swap_rate)
                    * self.forward_fx_diffusion_fprime(swap_rate))
        return [func1, func2, func3]

    def make_denominator_call_integrands(self, **kwargs):
        """make_denominator_call_integrands
        See :py:func:`_make_denominator_integrands`.

        :return: array of function.
        :rtype: array.
        """
        return self._make_denominator_integrands()

    def make_denominator_put_integrands(self, **kwargs):
        """make_denominator_put_integrands
        See :py:func:`_make_denominator_integrands`.

        :return: array of function.
        :rtype: array.
        """
        return self._make_denominator_integrands()

    def make_denominator_analytic_funcs(self, **kwargs):
        """make_denominator_analytic_funcs
        Return constant term in replication method.

        :return: array of function.
        :rtype: array.
        """
        def func1(init_swap_rate):
            return (self.annuity_mapping_func(init_swap_rate)
                    * self.forward_fx_diffusion(init_swap_rate))
        return [func1]


class _SimpleQuantoCmsLinearBullSpreadHelper(SimpleQuantoCmsHelper):
    """_SimpleQuantoCmsLinearBullSpreadHelper
    In linear TSR model with bull-spread payoff,
    stochastic weight in the numerator is

        .. math::
            \\begin{eqnarray}
                g_{\mathrm{bullspread}}^{\prime\prime}(s; K)\\alpha(s)\\tilde{\chi}(s)
                    & = &
                    \delta(s - K)\\alpha(s)\\tilde{\chi}(s),
                \\\\
                g_{\mathrm{bullspread}}(s; K)\\alpha(s)\\tilde{\chi}^{\prime\prime}(s)
                    & = &
                    (s - K)^{+}
                    (\\alpha_{1}s + \\alpha_{2})
                    \\rho_{XS}\sigma_{X}\sqrt{T}
                    \left(
                       h^{\prime\prime}(s)\\tilde{\chi}(s)
                        + \\rho_{XS}\sigma_{X}\sqrt{T} h^{\prime}(s)^{2} \\tilde{\chi}(s)
                    \\right),
                \\\\
                2g_{\mathrm{bullspread}}^{\prime}(s; K)\\alpha^{\prime}(s)\\tilde{\chi}(s)
                    & = & 2 1_{[K, \infty)}(s) \\alpha_{1}
                        \exp
                        \left(
                            \\rho_{XS}\sigma_{X}\sqrt{T}\Phi^{-1}(\Psi^{A}(s))
                                + \\frac{\sigma_{X}^{2}T}{2}(1 - \\rho_{XS}^{2})
                        \\right),
                \\\\
                2g_{\mathrm{bullspread}}^{\prime}(s; K)\\alpha(s)\\tilde{\chi}^{\prime}(s)
                    & = & 2 1_{[K, \infty)}(s)
                        (\\alpha_{1}s + \\alpha_{2})
                        \\rho_{XS}\sigma_{X}\sqrt{T}h^{\prime}(s)\\tilde{\chi}(s),
                \\\\
                2g_{\mathrm{bullspread}}(s; K)\\alpha^{\prime}(s)\\tilde{\chi}^{\prime}(s)
                    & = &
                        2(s - K)^{+} \\alpha_{1}
                            \\rho_{XS}\sigma_{X}\sqrt{T}h^{\prime}(s)\\tilde{\chi}(s),
            \end{eqnarray}

    where
    :math:`g_{\mathrm{bullspread}}` is payoff function,
    :math:`\\alpha` is annuity mapping function,
    :math:`\\tilde{\chi}(s)` is forward fx diffusion,
    see :py:func:`_forward_fx_diffusion`.

    Stochastic weight in the denominator is

    .. math::
        \\begin{eqnarray}
            \\alpha^{\prime\prime}(s)\\tilde{\chi}(s)
                & = & 0
                \\\\
            \\alpha(s)\\tilde{\chi}^{\prime\prime}(s)
                & = &
                (\\alpha_{1}s + \\alpha_{2})
                \\rho_{XS}\sigma_{X}\sqrt{T}
                \left(
                       h^{\prime\prime}(s)\\tilde{\chi}(s)
                    + \\rho_{XS}\sigma_{X}\sqrt{T} h^{\prime}(s)^{2} \\tilde{\chi}(s)
                \\right),
                \\\\
            2\\alpha^{\prime}(s)\\tilde{\chi}^{\prime}(s)
                & = &
                    2\\alpha_{1}
                    \\rho_{XS}\sigma_{X}\sqrt{T}h^{\prime}(s)\\tilde{\chi}(s),
        \end{eqnarray}

    :param AnnuityMappingFuncHelper annuity_mapping_helper:
    :param PayoffHelper bull_spread_payoff_helper:
    :param _ForwardFxDiffusionHelper forward_fx_diffusion_helper:
    :param call_pricer: call option pricer.
    :param put_pricer: put option pricer.
    :param float payoff_lower_strike: lower strike of
        bull spread option payoff function.
    :param float payoff_upper_strike: upper strike of
        bull spread option payoff function.
    """

    def __init__(self,
                 annuity_mapping_helper,
                 bull_spread_payoff_helper,
                 forward_fx_diffusion_helper,
                 call_pricer,
                 put_pricer,
                 payoff_lower_strike,
                 payoff_upper_strike,
                 min_put_range=-np.inf,
                 max_put_range=np.inf,
                 min_call_range=-np.inf,
                 max_call_range=np.inf):
        """__init__
        """
        # annutiy mapping funciton
        self.annuity_mapping_func = annuity_mapping_helper.make_func()
        self.annuity_mapping_fprime = annuity_mapping_helper.make_fprime()
        self.annuity_mapping_fhess = annuity_mapping_helper.make_fhess()
        # payoff
        self.payoff_func = bull_spread_payoff_helper.make_func()
        self.payoff_fprime = bull_spread_payoff_helper.make_fprime()
        self.payoff_lower_strike = payoff_lower_strike
        self.payoff_upper_strike = payoff_upper_strike
        # forawad fx diffusion
        fwd_fx_helper = forward_fx_diffusion_helper
        self.forward_fx_diffusion = fwd_fx_helper.make_func()
        self.forward_fx_diffusion_fprime = fwd_fx_helper.make_fprime()
        self.forward_fx_diffusion_fhess = fwd_fx_helper.make_fhess()
        # pricer
        self.call_pricer = call_pricer
        self.put_pricer = put_pricer
        # range
        self.min_put_range = min_put_range
        self.max_put_range = max_put_range
        self.min_call_range = min_call_range
        self.max_call_range = max_call_range

    def _make_numerator_integrands(self):
        """_make_numerator_integrands
        Returns following functions:

        .. math::
            g_{\mathrm{bullspread}}(s; K_{l}, K_{u})
                \\alpha(s)\\tilde{\chi}^{\prime\prime}(s)
            \\\\
            2g_{\mathrm{bullspread}}^{\prime}(s; K_{l}, K_{u})
                \\alpha^{\prime}(s)\\tilde{\chi}(s)
            \\\\
            2g_{\mathrm{bullspread}}^{\prime}(s; K_{l}, K_{u})
                \\alpha(s)\\tilde{\chi}^{\prime}(s)
            \\\\
            2g_{\mathrm{bullspread}}(s; K_{l}, K_{u})
                \\alpha^{\prime}(s)\\tilde{\chi}^{\prime}(s)

        :return: array of function.
        :rtype: array.
        """
        def func1(swap_rate):
            """func1
            """
            return (self.payoff_func(swap_rate)
                    * self.annuity_mapping_func(swap_rate)
                    * self.forward_fx_diffusion_fhess(swap_rate))

        def func2(swap_rate):
            return (2.0
                    * self.payoff_fprime(swap_rate)
                    * self.annuity_mapping_fprime(swap_rate)
                    * self.forward_fx_diffusion(swap_rate))

        def func3(swap_rate):
            return (2.0
                    * self.payoff_fprime(swap_rate)
                    * self.annuity_mapping_func(swap_rate)
                    * self.forward_fx_diffusion_fprime(swap_rate))

        def func4(swap_rate):
            return (2.0
                    * self.payoff_func(swap_rate)
                    * self.annuity_mapping_fprime(swap_rate)
                    * self.forward_fx_diffusion_fprime(swap_rate))

        return [func1, func2, func3, func4]

    def make_numerator_call_integrands(self, **kwargs):
        """make_numerator_call_integrands
        See :py:func:`_make_numerator_integrands`.

        :return: array of function.
        :rtype: array.
        """
        return self._make_numerator_integrands()

    def make_numerator_put_integrands(self, **kwargs):
        """make_numerator_put_integrands
        See :py:func:`_make_numerator_integrands`.

        :return: array of function.
        :rtype: array.
        """
        return self._make_numerator_integrands()

    def make_numerator_analytic_funcs(self, **kwargs):
        """make_numerator_analytic_funcs
        There are 3 functions which is calculated analytically.
        First is constant term in replication method.
        Second and third is the term
        which contains dirac delta function as integrands.
        Specifically, the second and third term are

        .. math::
            p(K_{f}) \\alpha(K_{f})\\tilde{\chi}(K_{f})
            1_{(L_{\min}^{\mathrm{put}},L_{\max}^{\mathrm{put}})}(K_{f})
            \\\\
            p(K_{c}) \\alpha(K_{c})\\tilde{\chi}(K_{c})
            1_{(L_{\min}^{\mathrm{put}},L_{\max}^{\mathrm{put}})}(K_{c})
            \\\\
            c(K_{f}) \\alpha(K_{f})\\tilde{\chi}(K_{f})
            1_{(L_{\min}^{\mathrm{call}},L_{\max}^{\mathrm{call}})}(K_{f})
            \\\\
            c(K_{c}) \\alpha(K_{c})\\tilde{\chi}(K_{c})
            1_{(L_{\min}^{\mathrm{call}},L_{\max}^{\mathrm{call}})}(K_{c})

        where
        :math:`p(K) := p(0, S(0); K, T)`,
        :math:`c(K) := c(0, S(0); K, T)`,
        :math:`L_{\min}^{\mathrm{put}}`
        is lower bound of integral range for put,
        :math:`L_{\max}^{\mathrm{put}}`
        is upper bound of integral range for put.
        :math:`L_{\min}^{\mathrm{call}}`
        is lower bound of integral range for call,
        :math:`L_{\max}^{\mathrm{call}}`
        is upper bound of integral range for call.

        This term contains dirac delta function
        (second derivative of payoff function)
        so that can be calculated analytically.

        :return: array of function.
        :rtype: array.
        """
        def func1(init_swap_rate):
            return (self.payoff_func(init_swap_rate)
                    * self.annuity_mapping_func(init_swap_rate)
                    * self.forward_fx_diffusion(init_swap_rate))

        # put term1
        def func21(init_swap_rate):
            return (self.put_pricer(self.payoff_lower_strike)
                    * self.annuity_mapping_func(self.payoff_lower_strike)
                    * self.forward_fx_diffusion(self.payoff_lower_strike))

        # put term2
        def func22(init_swap_rate):
            return (self.put_pricer(self.payoff_upper_strike)
                    * self.annuity_mapping_func(self.payoff_upper_strike)
                    * self.forward_fx_diffusion(self.payoff_upper_strike))

        # call term1
        def func31(init_swap_rate):
            return (self.call_pricer(self.payoff_lower_strike)
                    * self.annuity_mapping_func(self.payoff_lower_strike)
                    * self.forward_fx_diffusion(self.payoff_lower_strike))

        # call term2
        def func32(init_swap_rate):
            return (self.call_pricer(self.payoff_upper_strike)
                    * self.annuity_mapping_func(self.payoff_upper_strike)
                    * self.forward_fx_diffusion(self.payoff_upper_strike))

        terms = [func1]
        if self.min_put_range < self.payoff_lower_strike < self.max_put_range:
            terms.append(func21)
        if self.min_put_range < self.payoff_upper_strike < self.max_put_range:
            terms.append(func22)
        if self.min_call_range < self.payoff_lower_strike < self.max_call_range:
            terms.append(func31)
        if self.min_call_range < self.payoff_upper_strike < self.max_call_range:
            terms.append(func32)
        return terms

    def _make_denominator_integrands(self):
        """`_make_denominator_integrands`
        Returns following functions:

        .. math::
            \\alpha^{\prime\prime}(s)\\tilde{\chi}(s)
            \\\\
            \\alpha(s)\\tilde{\chi}^{\prime\prime}(s)
            \\\\
            2\\alpha^{\prime}(s)\\tilde{\chi}^{\prime}(s)

        :return: array of function.
        :rtype: array.
        """
        def func1(swap_rate):
            return (self.annuity_mapping_fhess(swap_rate)
                    * self.forward_fx_diffusion(swap_rate))

        def func2(swap_rate):
            return (self.annuity_mapping_func(swap_rate)
                    * self.forward_fx_diffusion_fhess(swap_rate))

        def func3(swap_rate):
            return (2.0
                    * self.annuity_mapping_fprime(swap_rate)
                    * self.forward_fx_diffusion_fprime(swap_rate))
        return [func1, func2, func3]

    def make_denominator_call_integrands(self, **kwargs):
        """make_denominator_call_integrands
        See :py:func:`_make_denominator_integrands`.

        :return: array of function.
        :rtype: array.
        """
        return self._make_denominator_integrands()

    def make_denominator_put_integrands(self, **kwargs):
        """make_denominator_put_integrands
        See :py:func:`_make_denominator_integrands`.

        :return: array of function.
        :rtype: array.
        """
        return self._make_denominator_integrands()

    def make_denominator_analytic_funcs(self, **kwargs):
        """make_denominator_analytic_funcs
        Return constant term in replication method.

        :return: array of function.
        :rtype: array.
        """
        def func1(init_swap_rate):
            return (self.annuity_mapping_func(init_swap_rate)
                    * self.forward_fx_diffusion(init_swap_rate))
        return [func1]


def _make_numerator_replication(quanto_cms_helper, call_pricer, put_pricer):
    """_make_numerator_replication

    :param SimpleQuantoCmsHelper quanto_cms_helper:
    :param call_pricer:
    :param put_pricer:
    :return: generated replication object
    :rtype: :py:class:`AnalyticReplication`
    """
    put_integrands = quanto_cms_helper.make_numerator_put_integrands()
    call_integrands = quanto_cms_helper.make_numerator_call_integrands()
    analytic_funcs = quanto_cms_helper.make_numerator_analytic_funcs()
    return replication.AnalyticReplication(
        call_pricer,
        put_pricer,
        analytic_funcs,
        put_integrands,
        call_integrands)


def _make_denominator_replication(quanto_cms_helper, call_pricer, put_pricer):
    """_make_denominator_replication

    :param qunato_cms_helper:
    :param call_pricer:
    :param put_pricer:
    :return: generated replication object
    :rtype: :py:class:`AnalyticReplication`
    """
    put_integrands = quanto_cms_helper.make_denominator_put_integrands()
    call_integrands = quanto_cms_helper.make_denominator_call_integrands()
    analytic_funcs = quanto_cms_helper.make_denominator_analytic_funcs()
    return replication.AnalyticReplication(
        call_pricer,
        put_pricer,
        analytic_funcs,
        put_integrands,
        call_integrands)


def replicate(init_swap_rate,
              discount_factor,
              call_pricer,
              put_pricer,
              payoff_type,
              payoff_params,
              forward_fx_diffusion_params,
              annuity_mapping_type,
              annuity_mapping_params,
              min_put_range=-np.inf,
              max_call_range=np.inf):
    """replicate
    Simple qunato cms pricer.

    .. math::
        V_{\mathrm{QuantoCMS}}(0)
        \\approx
            P_{f}(0, T_{p})
            \\frac{\mathrm{E}^{A,d}
            \left[
                g(S(T)) \\alpha(S(T)) \\tilde{\chi}(S(T))
            \\right]
            }{\mathrm{E}^{A,d}
            \left[
                \\alpha(S(T)) \\tilde{\chi}(S(T))
            \\right]
            }

    where
    :math:`T_{p}` is payment date of Quanto CMS,
    :math:`P_{f}(0, T)` is value of foreign currency zero coupon bond
    with maturity :math:`T` at time 0.
    :math:`\mathrm{E}^{A,d}` is expectation under domestic annuity measure.
    :math:`g` is payoff function,
    :math:`\\alpha(\cdot)` is annuity mapping function,
    :math:`\\tilde{\chi}(\cdot)` is forward fx diffusion,
    :math:`S(T)` is time :math:`T` forward swap rate.

    See
    `Interest Rate Modeling. Volume 3: Products and Risk Management, Chap16`.

    :param float init_swap_rate:
    :param float discount_factor:
    :param callable call_pricer:
    :param callable put_pricer:
    :param str payoff_type:
        Specify type of payoff function as a string
        ("call", "put", "bull_spread").
    :param dict payoff_params:
    :param dict forward_fx_diffusion_params:
    :param str annuity_mapping_type:
        Specify type of annuity mapping function as a string ("linear").
    :param dict annuity_mapping_params:
    :param float min_put_range:
    :param float max_call_range:
    """
    payoff_dict = {
        "call": 1,
        "put": 2,
        "bull_spread": 3,
    }
    annuity_mapping_dict = {
        "linear": 1,
    }

    annuity_mapping_type = annuity_mapping_dict[annuity_mapping_type]
    payoff_type = payoff_dict[payoff_type]
    # forward fx diffusion
    forward_fx_diffusion_helper = _ForwardFxDiffusionHelper(
        **forward_fx_diffusion_params)
    # annuity mapping func
    if annuity_mapping_type == 1:
        annuity_mapping_helper = replication.LinearAnnuityMappingFuncHelper(
            **annuity_mapping_params)
    # payoff
    if payoff_type == 1:
        payoff_helper = payoff.CallUnderlyingPayoffHelper(**payoff_params)
    payoff_strike = payoff_params["strike"]
    # pricer
    # quanto cms helper
    if payoff_type == 1 and annuity_mapping_type == 1:
        quanto_cms_helper = _SimpleQuantoCmsLinearCallHelper(
            annuity_mapping_helper,
            payoff_helper,
            forward_fx_diffusion_helper,
            call_pricer,
            put_pricer,
            payoff_strike,
            min_put_range,
            max_call_range)
    # replication
    numerator_replication = _make_numerator_replication(
        quanto_cms_helper, call_pricer, put_pricer)
    denominator_replication = _make_denominator_replication(
        quanto_cms_helper, call_pricer, put_pricer)
    # calculate
    numerator = numerator_replication.eval(
        init_swap_rate, min_put_range, max_call_range)
    print("numerator", numerator)
    denominator = denominator_replication.eval(
        init_swap_rate, min_put_range, max_call_range)
    print("denominator", denominator)
    return discount_factor * numerator / denominator
