#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import numpy as np

from mafipy import function
from mafipy.replication import _quanto_cms_forward_fx as _fx
from mafipy.replication import replication_method as replication_method


# -----------------------------------------------------------------------------
# Black swaption model
# -----------------------------------------------------------------------------
def make_pdf_black_swaption(
        init_swap_rate,
        swap_annuity,
        option_maturity,
        vol):
    """make_pdf_black_swaption
    return p.d.f. of black swaption model.

    .. math::
        \Phi_{S}(k) := \\frac{\partial^{2}}{\partial K^{2}}
            V_{\mathrm{payer}}(0, S; T, K, A, \sigma)

    where
    :math:`S` is `init_swap_rate`,
    :math:`A` is `swap_annuity`,
    :math:`T` is `option_maturity`,
    :math:`K` is `option_strike`,
    :math:`\Phi_{S}(k)` is p.d.f. of :math:`S`,
    :math:`V_{\mathrm{payers}}(0, S; T, K)` is value of payer`s swaption
    with strike :math:`K` and :math:`T` maturity at time 0.

    :param float init_swap_rate: initial swap rate.
    :param float swap_annuity: annuity of referencing swap.
    :param float option_maturity: maturity of swaption.
    :param float vol: volatility. non-negative.
    :return: p.d.f. of swaption price under black swaption model
        as a function of strike.
    :rtype: function with strike.

    :raises AssertionError: if volatility is not positive.
    """
    assert(vol > 0.0)

    def pdf(option_strike):
        return function.black_payers_swaption_value_fhess_by_strike(
            init_swap_rate=init_swap_rate,
            option_strike=option_strike,
            swap_annuity=1.0,
            option_maturity=option_maturity,
            vol=vol)

    return pdf


def make_pdf_fprime_black_swaption(
        init_swap_rate,
        swap_annuity,
        option_maturity,
        vol):
    """make_pdf_fprime_black_swaption
    return first derivative of p.d.f. of black swaption.
    See :py:func:`make_pdf_black_swaption`.

    .. math::
        \Phi_{S}^{\prime}(k)
        := \\frac{\partial^{3}}{\partial K^{3}}
            V_{\mathrm{payers}}(0, S; K, A, T)

    where
    :math:`S` is `init_swap_rate`,
    :math:`S` is `swap_annuity`,
    :math:`T` is `option_maturity`,
    :math:`K` is `option_strike`,
    :math:`\Phi_{S}(k)` is p.d.f. of :math:`S`,
    :math:`V_{\mathrm{payers}}(0, S; K, A, T)` is value of call option
    with strike :math:`K` and :math:`T` maturity at time 0.

    :param float init_swap_rate: initial swap rate.
    :param float swap_annuity: annuity of referencing swap.
    :param float option_maturity: maturity of swaption.
    :param float vol: volatility. non-negative.
    :return: first derivative of p.d.f. of black swaption model
        as a function of strike.
    :rtype: function

    :raises AssertionError: if volatility is not positive.
    """
    assert(vol > 0.0)

    def pdf_fprime(option_strike):
        return function.black_payers_swaption_value_third_by_strike(
            init_swap_rate=init_swap_rate,
            option_strike=option_strike,
            swap_annuity=1.0,
            option_maturity=option_maturity,
            vol=vol)

    return pdf_fprime


def make_cdf_black_swaption(
        init_swap_rate,
        swap_annuity,
        option_maturity,
        vol):
    """make_cdf_black_swaption
    returns c.d.f. of black swaption.

    :param init_swap_rate: initial swap rate.
    :param swap_annuity: swap annuity.
    :param option_maturity: maturity of swaption.
    :param vol: volatility. non-negative.
    :return: distribution of black swaption model
        as a funciton of strike.
    :rtype: function

    :raises AssertionError: if volatility is not positive.
    """
    assert(vol > 0.0)

    return lambda option_strike: (
        1.0 + function.black_payers_swaption_value_fprime_by_strike(
            init_swap_rate, option_strike, 1.0, option_maturity, vol))


# -----------------------------------------------------------------------------
# Black scholes model
# -----------------------------------------------------------------------------
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
        return function.black_scholes_call_value_fhess_by_strike(
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
        return function.black_scholes_call_value_third_by_strike(
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
        1.0 + function.black_scholes_call_value_fprime_by_strike(
            underlying, strike, rate, maturity, vol))


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


class _SimpleQuantoCmsLinearCallHelper(SimpleQuantoCmsHelper):
    """_SimpleQuantoCmsLinearCallHelper
    In linear TSR model with call payoff,
    stochastic weight in the numerator is

        .. math::
            \\begin{eqnarray}
                g_{\mathrm{call}}^{\prime\prime}(s; K)
                    \\alpha(s)\\tilde{\chi}(s)
                    & = &
                    \delta(s - K)\\alpha(s)\\tilde{\chi}(s),
                \\\\
                g_{\mathrm{call}}(s; K)\\alpha(s)
                    \\tilde{\chi}^{\prime\prime}(s)
                    & = &
                    (s - K)^{+}
                    (\\alpha_{1}s + \\alpha_{2})
                    \\rho_{XS}\sigma_{X}\sqrt{T}
                    \left(
                       h^{\prime\prime}(s)\\tilde{\chi}(s)
                        + \\rho_{XS}\sigma_{X}\sqrt{T}
                            h^{\prime}(s)^{2} \\tilde{\chi}(s)
                    \\right),
                \\\\
                2g_{\mathrm{call}}^{\prime}(s; K)
                    \\alpha^{\prime}(s)\\tilde{\chi}(s)
                    & = & 2 1_{[K, \infty)}(s) \\alpha_{1}
                        \exp
                        \left(
                            \\rho_{XS}\sigma_{X}\sqrt{T}\Phi^{-1}(\Psi^{A}(s))
                                + \\frac{\sigma_{X}^{2}T}{2}
                                    (1 - \\rho_{XS}^{2})
                        \\right),
                \\\\
                2g_{\mathrm{call}}^{\prime}(s; K)
                    \\alpha(s)\\tilde{\chi}^{\prime}(s)
                    & = & 2 1_{[K, \infty)}(s)
                        (\\alpha_{1}s + \\alpha_{2})
                        \\rho_{XS}\sigma_{X}\sqrt{T}
                            h^{\prime}(s)\\tilde{\chi}(s),
                \\\\
                2g_{\mathrm{call}}(s; K)
                    \\alpha^{\prime}(s)\\tilde{\chi}^{\prime}(s)
                    & = &
                        2(s - K)^{+} \\alpha_{1}
                            \\rho_{XS}\sigma_{X}\sqrt{T}
                                h^{\prime}(s)\\tilde{\chi}(s),
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
                    + \\rho_{XS}\sigma_{X}\sqrt{T}
                        h^{\prime}(s)^{2} \\tilde{\chi}(s)
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
                 payoff_gearing,
                 min_put_range,
                 max_put_range,
                 min_call_range,
                 max_call_range):
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
        self.payoff_gearing = payoff_gearing
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
        # check range condition
        # min_put_range <= max_put_range = min_call_range <= max_call_range
        assert(min_put_range <= max_put_range)
        assert(np.isclose(max_put_range, min_call_range))
        assert(min_call_range <= max_call_range)

    def _make_numerator_integrands(self):
        """_make_numerator_integrands
        Returns following functions:

        .. math::
            g_{\mathrm{call}}(s; K)\\alpha(s)\\tilde{\chi}^{\prime\prime}(s)
            \\\\
            2g_{\mathrm{call}}^{\prime}(s; K)
                \\alpha^{\prime}(s)\\tilde{\chi}(s)
            \\\\
            2g_{\mathrm{call}}^{\prime}(s; K)
                \\alpha(s)\\tilde{\chi}^{\prime}(s)
            \\\\
            2g_{\mathrm{call}}(s; K)
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
            # g * a * chi
            return (self.payoff_func(init_swap_rate)
                    * self.annuity_mapping_func(init_swap_rate)
                    * self.forward_fx_diffusion(init_swap_rate))

        def func2(init_swap_rate):
            # p * g'' * a * chi
            return (self.put_pricer(self.payoff_strike)
                    * self.payoff_gearing
                    * self.annuity_mapping_func(self.payoff_strike)
                    * self.forward_fx_diffusion(self.payoff_strike))

        def func3(init_swap_rate):
            # c * g'' * a * chi
            return (self.call_pricer(self.payoff_strike)
                    * self.payoff_gearing
                    * self.annuity_mapping_func(self.payoff_strike)
                    * self.forward_fx_diffusion(self.payoff_strike))

        terms = [func1]
        if self.min_put_range <= self.payoff_strike <= self.max_put_range:
            terms.append(func2)
        if self.min_call_range <= self.payoff_strike <= self.max_call_range:
            terms.append(func3)
        return terms


class _SimpleQuantoCmsLinearBullSpreadHelper(SimpleQuantoCmsHelper):
    """_SimpleQuantoCmsLinearBullSpreadHelper
    In linear TSR model with bull-spread payoff,
    stochastic weight in the numerator is

        .. math::
            \\begin{eqnarray}
                g_{\mathrm{bullspread}}^{\prime\prime}(s; K)
                \\alpha(s)\\tilde{\chi}(s)
                    & = &
                    \delta(s - K)\\alpha(s)\\tilde{\chi}(s),
                \\\\
                g_{\mathrm{bullspread}}(s; K)
                \\alpha(s)\\tilde{\chi}^{\prime\prime}(s)
                    & = &
                    (s - K)^{+}
                    (\\alpha_{1}s + \\alpha_{2})
                    \\rho_{XS}\sigma_{X}\sqrt{T}
                    \left(
                       h^{\prime\prime}(s)\\tilde{\chi}(s)
                        + \\rho_{XS}\sigma_{X}\sqrt{T} h^{\prime}(s)^{2}
                            \\tilde{\chi}(s)
                    \\right),
                \\\\
                2g_{\mathrm{bullspread}}^{\prime}(s; K)
                \\alpha^{\prime}(s)\\tilde{\chi}(s)
                    & = & 2 1_{[K, \infty)}(s) \\alpha_{1}
                        \exp
                        \left(
                            \\rho_{XS}\sigma_{X}\sqrt{T}\Phi^{-1}(\Psi^{A}(s))
                                + \\frac{\sigma_{X}^{2}T}{2}
                                    (1 - \\rho_{XS}^{2})
                        \\right),
                \\\\
                2g_{\mathrm{bullspread}}^{\prime}(s; K)
                \\alpha(s)\\tilde{\chi}^{\prime}(s)
                    & = & 2 1_{[K, \infty)}(s)
                        (\\alpha_{1}s + \\alpha_{2})
                        \\rho_{XS}\sigma_{X}\sqrt{T}h^{\prime}(s)
                        \\tilde{\chi}(s),
                \\\\
                2g_{\mathrm{bullspread}}(s; K)
                \\alpha^{\prime}(s)\\tilde{\chi}^{\prime}(s)
                    & = &
                        2(s - K)^{+} \\alpha_{1}
                            \\rho_{XS}\sigma_{X}\sqrt{T}h^{\prime}(s)
                            \\tilde{\chi}(s),
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
                    + \\rho_{XS} \sigma_{X}
                        \sqrt{T} h^{\prime}(s)^{2} \\tilde{\chi}(s)
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
                 payoff_gearing,
                 min_put_range,
                 max_put_range,
                 min_call_range,
                 max_call_range):
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
        self.payoff_gearing = payoff_gearing
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
        # check range condition
        # min_put_range <= max_put_range = min_call_range <= max_call_range
        assert(min_put_range <= max_put_range)
        assert(np.isclose(max_put_range, min_call_range))
        assert(min_call_range <= max_call_range)

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
            # g * a * chi
            return (self.payoff_func(init_swap_rate)
                    * self.annuity_mapping_func(init_swap_rate)
                    * self.forward_fx_diffusion(init_swap_rate))

        # put term1
        def func21(init_swap_rate):
            # p * g'' * a * chi at lower_strike
            return (self.put_pricer(self.payoff_lower_strike)
                    * self.payoff_gearing
                    * self.annuity_mapping_func(self.payoff_lower_strike)
                    * self.forward_fx_diffusion(self.payoff_lower_strike))

        # put term2
        def func22(init_swap_rate):
            # p * g'' * a * chi at upper_strike
            return -(self.put_pricer(self.payoff_upper_strike)
                     * self.payoff_gearing
                     * self.annuity_mapping_func(self.payoff_upper_strike)
                     * self.forward_fx_diffusion(self.payoff_upper_strike))

        # call term1
        def func31(init_swap_rate):
            # c * g'' * a * chi at lower_strike
            return (self.call_pricer(self.payoff_lower_strike)
                    * self.payoff_gearing
                    * self.annuity_mapping_func(self.payoff_lower_strike)
                    * self.forward_fx_diffusion(self.payoff_lower_strike))

        # call term2
        def func32(init_swap_rate):
            # c * g'' * a * chi at upper_strike
            return -(self.call_pricer(self.payoff_upper_strike)
                     * self.payoff_gearing
                     * self.annuity_mapping_func(self.payoff_upper_strike)
                     * self.forward_fx_diffusion(self.payoff_upper_strike))

        terms = [func1]
        # check whether delta function concentrate within integral range
        # second derivative of payoff function heuristically is
        # dirac delta function which cocentrates on strike so that
        # the terms need to be calculated as analytic term
        # if integral range includes strike

        # integral with put pricer
        if (self.min_put_range
                <= self.payoff_lower_strike
                <= self.max_put_range):
            terms.append(func21)
        elif (self.min_call_range
              <= self.payoff_lower_strike
              <= self.max_call_range):
            terms.append(func31)
        if (self.min_put_range
                <= self.payoff_upper_strike
                <= self.max_put_range):
            terms.append(func22)
        elif (self.min_call_range
              <= self.payoff_upper_strike
              <= self.max_call_range):
            terms.append(func32)
        return terms


def _replicate_numerator(init_swap_rate,
                         quanto_cms_helper,
                         call_pricer,
                         put_pricer,
                         min_put_range=-np.inf,
                         max_call_range=np.inf):
    """_replicate_numerator

    :param float init_swap_rate:
    :param SimpleQuantoCmsHelper quanto_cms_helper:
    :param callable call_pricer:
    :param callable put_pricer:
    :param float min_put_range:
    :param float max_call_range:

    :return: expectation in numerator.
    :rtype: float.
    """
    put_integrands = quanto_cms_helper.make_numerator_put_integrands()
    call_integrands = quanto_cms_helper.make_numerator_call_integrands()
    analytic_funcs = quanto_cms_helper.make_numerator_analytic_funcs()
    replicator = replication_method.AnalyticReplication(
        call_pricer,
        put_pricer,
        analytic_funcs,
        call_integrands,
        put_integrands)
    return replicator.eval(
        init_swap_rate, min_put_range, max_call_range)


def _replicate_denominator(init_swap_rate,
                           call_pricer,
                           put_pricer,
                           annuity_mapping_helper,
                           forward_fx_diffusion_helper,
                           min_put_range=-np.inf,
                           max_call_range=np.inf):
    """_replicate_denominator


    .. math::
        \\alpha^{\prime\prime}(s)\\tilde{\chi}(s)
        \\\\
        \\alpha(s)\\tilde{\chi}^{\prime\prime}(s)
        \\\\
        2\\alpha^{\prime}(s)\\tilde{\chi}^{\prime}(s)

    :param float init_swap_rate:
    :param float call_pricer:
    :param float put_pricer:
    :param AnnuityMappingFuncHelper annuity_mapping_helper:
    :param _ForwardFxDiffusionHelper forward_fx_diffusion_helper:
    :param float min_put_range:
    :param float max_call_range:
    """
    annuity_mapping_func = annuity_mapping_helper.make_func()
    annuity_mapping_fprime = annuity_mapping_helper.make_fprime()
    annuity_mapping_fhess = annuity_mapping_helper.make_fhess()
    forward_fx_diffusion = forward_fx_diffusion_helper.make_func()
    forward_fx_diffusion_fprime = forward_fx_diffusion_helper.make_fprime()
    forward_fx_diffusion_fhess = forward_fx_diffusion_helper.make_fhess()

    # analytic funcs
    def alpha_chi(init_swap_rate):
        return (annuity_mapping_func(init_swap_rate)
                * forward_fx_diffusion(init_swap_rate))

    analytic_funcs = [alpha_chi]

    # integrands with pricer
    def alpha_fhess_chi(swap_rate):
        return (annuity_mapping_fhess(swap_rate)
                * forward_fx_diffusion(swap_rate))

    def alpha_fprime_chi_fprime(swap_rate):
        return (2.0
                * annuity_mapping_fprime(swap_rate)
                * forward_fx_diffusion_fprime(swap_rate))

    def alpha_chi_fhess(swap_rate):
        return (annuity_mapping_func(swap_rate)
                * forward_fx_diffusion_fhess(swap_rate))

    # put integrands
    put_integrands = [
        alpha_fhess_chi, alpha_fprime_chi_fprime, alpha_chi_fhess]
    # call integrands
    call_integrands = put_integrands

    # replication
    replicator = replication_method.AnalyticReplication(
        call_pricer,
        put_pricer,
        analytic_funcs,
        call_integrands,
        put_integrands)
    return replicator.eval(
        init_swap_rate, min_put_range, max_call_range)


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
    forward_fx_diffusion_helper = _fx._ForwardFxDiffusionHelper(
        **forward_fx_diffusion_params)
    # annuity mapping func
    if annuity_mapping_type == 1:
        annuity_mapping_helper = (
            replication_method.LinearAnnuityMappingFuncHelper(
                **annuity_mapping_params))
    # payoff
    if payoff_type == 1:
        payoff_helper = function.CallUnderlyingPayoffHelper(**payoff_params)
    elif payoff_type == 3:
        payoff_helper = function.BullSpreadUnderlyingPayoffHelper(
            **payoff_params)
    # pricer
    # quanto cms helper
    if payoff_type == 1 and annuity_mapping_type == 1:
        quanto_cms_helper = _SimpleQuantoCmsLinearCallHelper(
            annuity_mapping_helper,
            payoff_helper,
            forward_fx_diffusion_helper,
            call_pricer,
            put_pricer,
            payoff_params["strike"],
            payoff_params["gearing"],
            min_put_range,
            init_swap_rate,
            init_swap_rate,
            max_call_range)
    elif payoff_type == 3 and annuity_mapping_type == 1:
        quanto_cms_helper = _SimpleQuantoCmsLinearBullSpreadHelper(
            annuity_mapping_helper,
            payoff_helper,
            forward_fx_diffusion_helper,
            call_pricer,
            put_pricer,
            payoff_params["lower_strike"],
            payoff_params["upper_strike"],
            payoff_params["gearing"],
            min_put_range,
            init_swap_rate,
            init_swap_rate,
            max_call_range)

    # replication
    numerator = _replicate_numerator(init_swap_rate,
                                     quanto_cms_helper,
                                     call_pricer,
                                     put_pricer,
                                     min_put_range,
                                     max_call_range)
    denominator = _replicate_denominator(init_swap_rate,
                                         call_pricer,
                                         put_pricer,
                                         annuity_mapping_helper,
                                         forward_fx_diffusion_helper,
                                         min_put_range,
                                         max_call_range)
    return discount_factor * numerator / denominator
