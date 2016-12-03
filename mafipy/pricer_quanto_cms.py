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


def make_pdf_black_scholes_model(
        underlying,
        rate,
        maturity,
        vol):
    """make_pdf_black_scholes_model

    :param underlying:
    :param rate:
    :param maturity: non-negative.
    :param vol: volatility. non-negative.
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


def make_pdf_fprime_black_scholes_model(
        underlying,
        rate,
        maturity,
        vol):
    """make_pdf_fprime_black_scholes_model

    :param float underlying:
    :param float rate:
    :param float maturity: non-negative.
    :param float vol: volatility. non-negative.
    :return: p.d.f. of call price under black scholes model
        as a function of strike.
    :rtype: lambda
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


def make_cdf_black_scholes_model(
        underlying,
        rate,
        maturity,
        vol):
    """make_cdf_black_scholes_model

    :param underlying:
    :param rate:
    :param maturity:
    :param vol:
    :return: distribution of call price under black scholes model
        as a funciton of strike.
    :rtype: lambda

    """
    return lambda strike: (
        1.0 + analytic_formula.black_scholes_call_value_fprime_by_strike(
            underlying, strike, rate, maturity, vol))


def _calc_h(swap_rate_cdf, swap_rate):
    norm = scipy.stats.norm

    return norm.ppf(swap_rate_cdf(swap_rate))


def _calc_h_fprime(swap_rate_pdf, swap_rate, h):
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

    .. math::

        \\tilde{\chi}(s)
        := \exp
            \left(
                \\rho_{XS}\sigma_{X}\sqrt{T}\Phi^{-1}(\Psi^{A}(s))
                    + \\frac{\sigma_{X}^{2}T}{2}(1 - \\rho_{XS}^{2})
            \\right).

    where
    :math:`s` is :param swap_rate:,
    :math:`\\rho_{XS}` is :param corr:,
    :math:`\sigma_{X}` is :param vol:,

    :param float swap_rate:
    :param float time:
    :param float vol:
    :param float corr:
    :param callable dist_func: distribution of foward swap rate
        under annuity measure
    :return: value of forward fx diffusion.
    """
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
    :math:`h(s) := \Phi^{-1}(\Psi^{A}(s))`

    :param float swap_rate:
    :param float time:
    :param float vol:
    :param float corr:
    :param function swap_rate_cdf: distribution of forward swap rate
        under annuity measure
    """
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

    def __init__(self,
                 time,
                 vol,
                 corr,
                 swap_rate_cdf,
                 swap_rate_pdf,
                 swap_rate_pdf_fprime):
        self.time = time
        self.vol = vol
        self.corr = corr
        self.swap_rate_cdf = swap_rate_cdf
        self.swap_rate_pdf = swap_rate_pdf
        self.swap_rate_pdf_fprime = swap_rate_pdf_fprime

    def make_func(self):
        return lambda swap_rate: _forward_fx_diffusion(
            swap_rate=swap_rate,
            time=self.time,
            vol=self.vol,
            corr=self.corr,
            swap_rate_cdf=self.swap_rate_cdf,
            swap_rate_pdf=self.swap_rate_pdf,
            swap_rate_pdf_fprime=self.swap_rate_pdf_fprime)

    def make_fprime(self):
        return lambda swap_rate: _forward_fx_diffusion_fprime(
            swap_rate=swap_rate,
            time=self.time,
            vol=self.vol,
            corr=self.corr,
            swap_rate_cdf=self.swap_rate_cdf,
            swap_rate_pdf=self.swap_rate_pdf,
            swap_rate_pdf_fprime=self.swap_rate_pdf_fprime)

    def make_fhess(self):
        return lambda swap_rate: _forward_fx_diffusion_fhess(
            swap_rate=swap_rate,
            time=self.time,
            vol=self.vol,
            corr=self.corr,
            swap_rate_cdf=self.swap_rate_cdf,
            swap_rate_pdf=self.swap_rate_pdf,
            swap_rate_pdf_fprime=self.swap_rate_pdf_fprime)


class SimpleQuantoCmsHelper(object):

    def make_numerator_call_integrands(self, **kwargs):
        raise NotImplementedError

    def make_numerator_put_integrands(self, **kwargs):
        raise NotImplementedError

    def make_numerator_analytic_funcs(self, **kwargs):
        raise NotImplementedError

    def make_denominator_call_integrands(self, **kwargs):
        raise NotImplementedError

    def make_denominator_put_integrands(self, **kwargs):
        raise NotImplementedError

    def make_denominator_analytic_funcs(self, **kwargs):
        raise NotImplementedError


class SimpleQuantoCmsLinearCallHelper(SimpleQuantoCmsHelper):
    """SimpleQuantoCmsLinearCallHelper"""

    def __init__(self,
                 annuity_mapping_helper,
                 call_payoff_helper,
                 forward_fx_diffusion_helper,
                 call_pricer,
                 put_pricer,
                 payoff_strike):
        # annutiy mapping funciton
        self.annuity_mapping_func = annuity_mapping_helper.make_func()
        self.annuity_mapping_fprime = annuity_mapping_helper.make_fprime()
        self.annuity_mapping_fhess = annuity_mapping_helper.make_fhess()
        # payoff
        self.payoff_func = call_payoff_helper.make_func()
        self.payoff_fprime = call_payoff_helper.make_fprime()
        self.payoff_fhess = call_payoff_helper.make_fhess()
        self.payoff_strike = payoff_strike
        # forawad fx diffusion
        self.forward_fx_diffusion = forward_fx_diffusion_helper.make_func()
        self.forward_fx_diffusion_fprime = forward_fx_diffusion_helper.make_fprime()
        self.forward_fx_diffusion_fhess = forward_fx_diffusion_helper.make_fhess()
        # pricer
        self.call_pricer = call_pricer
        self.put_pricer = put_pricer

    def _make_numerator_integrands(self):
        def func1(swap_rate):
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
        return self._make_numerator_integrands()

    def make_numerator_put_integrands(self, **kwargs):
        return self._make_numerator_integrands()

    def make_numerator_analytic_funcs(self, **kwargs):
        def func1(init_swap_rate):
            return (self.payoff_func(init_swap_rate)
                    * self.annuity_mapping_func(init_swap_rate)
                    * self.forward_fx_diffusion(init_swap_rate))

        def func2(init_swap_rate):
            return (self.put_pricer(self.payoff_strike)
                    * self.annuity_mapping_func(self.payoff_strike)
                    * self.annuity_mapping_func(self.payoff_strike))

        def func3(init_swap_rate):
            return (self.call_pricer(self.payoff_strike)
                    * self.annuity_mapping_func(self.payoff_strike)
                    * self.annuity_mapping_func(self.payoff_strike))
        return [func1, func2, func3]

    def _make_denominator_integrands(self):
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
        return self._make_denominator_integrands()

    def make_denominator_put_integrands(self, **kwargs):
        return self._make_denominator_integrands()

    def make_denominator_analytic_funcs(self, **kwargs):
        def func1(init_swap_rate):
            return (self.annuity_mapping_func(init_swap_rate)
                    * self.forward_fx_diffusion(init_swap_rate))
        return [func1]


class SimpleQuantoCmsPricer(object):
    """SimpleQuantoCmsPricer"""

    payoff_dict = {
        "call": 1,
        "put": 2,
        "bull_spread": 3,
    }

    annuity_mapping_dict = {
        "linear": 1,
    }

    def __init__(self,
                 annuity_mapping_type,
                 annuity_mapping_params,
                 payoff_type,
                 payoff_params,
                 forward_fx_diffusion_params,
                 call_pricer,
                 put_pricer):
        """__init__

        :param annuity_mapping_type:
        :param annuity_mapping_params:
        :param payoff_type:
        :param payoff_params:
        :param forward_fx_diffusion_params:
        :param call_pricer:
        :param put_pricer:
        """
        self.annuity_mapping_type = \
            self.annuity_mapping_dict[annuity_mapping_type]
        self.payoff_type = self.payoff_dict[payoff_type]
        # forward fx diffusion
        self.forward_fx_diffusion_helper = _ForwardFxDiffusionHelper(
            **forward_fx_diffusion_params)
        # annuity mapping func
        if self.annuity_mapping_type == 1:
            self.annuity_mapping_helper = (
                replication.LinearAnnuityMappingFuncHelper(
                    **annuity_mapping_params))
        # payoff
        if self.payoff_type == 1:
            self.payoff_helper = payoff.CallUnderlyingPayoffHelper(
                **payoff_params)
        payoff_strike = payoff_params["strike"]
        # pricer
        self.call_pricer = call_pricer
        self.put_pricer = put_pricer
        # quanto cms
        if self.payoff_type == 1 and self.annuity_mapping_type == 1:
            quanto_cms_helper = SimpleQuantoCmsLinearCallHelper(
                self.annuity_mapping_helper,
                self.payoff_helper,
                self.forward_fx_diffusion_helper,
                call_pricer,
                put_pricer,
                payoff_strike)
        # replication
        self.numerator_replication = self._make_numerator_replication(
            quanto_cms_helper)
        self.denominator_replication = self._make_denominator_replication(
            quanto_cms_helper)

    def _make_numerator_replication(self, quanto_cms_helper):
        put_integrands = quanto_cms_helper.make_numerator_put_integrands()
        call_integrands = quanto_cms_helper.make_numerator_call_integrands()
        analytic_funcs = quanto_cms_helper.make_numerator_analytic_funcs()
        return replication.AnalyticReplication(
            self.call_pricer,
            self.put_pricer,
            analytic_funcs,
            put_integrands,
            call_integrands)

    def _make_denominator_replication(self, quanto_cms_helper):
        """_make_denominator_replication

        :param qunato_cms_helper:
        :return: generated replication object
        :rtype: :py:class:`AnalyticReplication`
        """
        put_integrands = quanto_cms_helper.make_denominator_put_integrands()
        call_integrands = quanto_cms_helper.make_denominator_call_integrands()
        analytic_funcs = quanto_cms_helper.make_denominator_analytic_funcs()
        return replication.AnalyticReplication(
            self.call_pricer,
            self.put_pricer,
            analytic_funcs,
            put_integrands,
            call_integrands)

    def eval(self,
             discount_factor,
             init_swap_rate,
             min_put_range=-np.inf,
             max_call_range=np.inf):
        """eval

        :param float discount_factor: discount factor at payment date.
        :param float init_swap_rate: initial swap rate.
        :param float min_put_range:
        :param float max_call_range:
        :return: value of quanto cms.
        :rtype: float
        """
        numerator = self.numerator_replication.eval(
            init_swap_rate,
            min_put_range=min_put_range,
            max_call_range=max_call_range)
        print("numerator", numerator)
        denominator = self.denominator_replication.eval(
            init_swap_rate,
            min_put_range=min_put_range,
            max_call_range=max_call_range)
        print("denominator", denominator)
        return discount_factor * numerator / denominator
