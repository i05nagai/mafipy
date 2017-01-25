#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import functools
import scipy.integrate

from . import util


# ----------------------------------------------------------------------------
# Annuity mapping function
# ----------------------------------------------------------------------------
def linear_annuity_mapping_func(underlying, alpha0, alpha1):
    """linear_annuity_mapping_func
    calculate linear annuity mapping function.
    Annuity mapping function is model of $P(t, T) / A(t)$
    so that it's value is positive.
    linear annuity mapping function calculates following formula:

    .. math::
        \\alpha(S) := S \\alpha_{0} + \\alpha_{1}.

    where
    :math:`S` is underlying,
    :math:`\\alpha_{0}` is alpha0,
    :math:`\\alpha_{1}` is alpha1.

    :param float underlying:
    :param float alpha0:
    :param float alpha1:
    :return: value of linear annuity mapping function.
    :rtype: float.
    """
    assert(underlying * alpha0 + alpha1 > 0)
    return underlying * alpha0 + alpha1


def linear_annuity_mapping_fprime(underlying, alpha0, alpha1):
    """linear_annuity_mapping_fprime
    first derivative of linear annuity mapping function.
    See :py:func:`linear_annuity_mapping_func`.
    The function calculates following formula:

    .. math::
        \\alpha^{\prime}(S) := \\alpha_{0.}

    where
    :math:`S` is underlying,
    :math:`\\alpha_{0}` is alpha0.

    :param float underlying:
    :param float alpha0:
    :param float alpha1: not used.
    :return: value of first derivative of linear annuity mapping function.
    :rtype: float.
    """
    return alpha0


def linear_annuity_mapping_fhess(underlying, alpha0, alpha1):
    """linear_annuity_mapping_fhess
    second derivative of linear annuity mapping function.
    See :py:func:`linear_annuity_mapping_func`
    and :py:func:`linear_annuity_mapping_fprime`.
    The value is 0.

    :param float underlying
    :param float alpha0: not used.
    :param float alpha1: not used.
    :return: return 0. value of second derivative.
    :rtype: float.
    """
    return 0.0


class AnnuityMappingFuncHelper(object):
    """AnnuityMappingFuncHelper
    Interface to generate annuity mapping function
    and it's first and second derivative.
    """

    def make_func(self):
        raise NotImplementedError

    def make_fprime(self):
        raise NotImplementedError

    def make_fhess(self):
        raise NotImplementedError


class LinearAnnuityMappingFuncHelper(AnnuityMappingFuncHelper):
    """LinearAnnuityMappingFuncHelper
    Helper class to generate linear annuity mappiing function
    and it's first and second derivative as a function of swap rate.

    :param float alpha0: required.
    :param float alpha1: required.
    """

    def __init__(self, **params):
        keys = ["alpha0", "alpha1"]
        self.params = util.check_keys(keys, params, locals())

    def make_func(self):
        """make_func
        returns linear annuity mapping function.
        See :py:func:`linear_annuity_mapping_func`.

        :return: linear annuity mapping function as a function of swap rate.
        :rtype: FuncionType
        """
        return functools.partial(linear_annuity_mapping_func, **self.params)

    def make_fprime(self):
        """make_fpirme
        returns first derivative of linear annuity mapping function.
        See :py:func:`linear_annuity_mapping_fprime`.

        :return: first derivative of linear annuity mapping function
            as a function of swap rate.
        :rtype: FuncionType
        """
        return functools.partial(linear_annuity_mapping_fprime, **self.params)

    def make_fhess(self):
        """make_fhess
        returns first derivative of linear annuity mapping function.
        See :py:func:`linear_annuity_mapping_fprime`.

        :return: first derivative of linear annuity mapping function
            as a function of swap rate.
        :rtype: FuncionType
        """
        return functools.partial(linear_annuity_mapping_fhess, **self.params)


# ----------------------------------------------------------------------------
# Replication
# ----------------------------------------------------------------------------
class Replication(object):
    """Replication
    Interface class to the class which uses replication method.
    """

    def eval(self):
        """eval
        evaluates value of derivative by replication methods.

        .. math::
            \mathrm{E}
            \left[
                f(S(T))
            \\right]
            = f(S(0))
                + \int_{-\infty}^{S(0)} f^{\prime\prime}(K)p(K)\ d K
                + \int_{S(0)}^{\infty} f^{\prime\prime}(K)c(K)\ d K

        where
        :math:`S(t)` is value of underlying at time :math:`t`,
        :math:`f(\cdot)` is payoff function as a function of underlying,
        :math:`p(K; S(0), T, \sigma)` is value of put pricer
        with strike :math:`K`.
        :math:`c(K; S(0), T, \sigma)` is value of call pricer
        with strike :math:`K`.
        """
        raise NotImplementedError


class AnalyticReplication(Replication):
    """AnalyticReplication
    This class calculates replication methods using knowledge of derivative.
    In mathmetical finance, payoff function is discontinuous function
    or the function which is diffentialble but cannot differentiate twice.
    Replication method calculates integration with stochastic weight
    which may contains derivative of payoff funciton.
    If integrands have dirac delta function or discontinuous function,
    numerical integration does not work properly.

    This class calculates numerical integration with
    pre-calculation of stochastic weight.

    :param callable call_pricer:
    :param callable put_pricer:
    :param list analytic_funcs:
    :param list call_integrands:
    :param list put_integrands:
    """

    def __init__(self,
                 call_pricer,
                 put_pricer,
                 analytic_funcs=[],
                 call_integrands=[],
                 put_integrands=[]):
        """__init__
        """
        self.call_pricer = call_pricer
        self.put_pricer = put_pricer
        self.analytic_funcs = analytic_funcs
        self.call_integrands = call_integrands
        self.put_integrands = put_integrands

    def _calc_analytic_term(self, init_swap_rate):
        """_calc_analytic_term

        :param init_swap_rate:
        """
        return sum([f(init_swap_rate) for f in self.analytic_funcs])

    def _calc_integral(self, func, pricer, lower, upper):
        """_calc_integral
        calculates following equation:

        .. math::
            \int_{L{\mathrm{lower}}}^{L_{\mathrm{upper}}} f(x) g(x)\ dx,

        where
        :math:`f(\cdot)` is `func`,
        :math:`g(\cdot)` is `pricer`,
        :math:`L_{\mathrm{lower}}` is `lower`,
        :math:`L_{\mathrm{upper}}` is `upper`.

        :param callable func:
        :param callable pricer: call option pricer or put option pricer.
        :param float lower: lower bound of integral.
        :param float upper: upper bound of integral.
        :return: integral value.
        :rtype: float.
        """
        integrate = scipy.integrate
        result = integrate.quad(lambda x: func(x) * pricer(x), lower, upper)
        return result[0]

    def _calc_put_integral(self, init_swap_rate, min_put_range):
        """_calc_put_integral
        integrate function with put pricer.

        :param init_swap_rate:
        :param min_put_range:
        :return: integal value.
        :rtype: float.
        """
        return sum([self._calc_integral(f, self.put_pricer,
                                        min_put_range, init_swap_rate)
                    for f in self.put_integrands])

    def _calc_call_integral(self, init_swap_rate, max_call_range):
        """_calc_call_integral
        integrate function with put pricer.

        :param init_swap_rate:
        :param max_call_range:
        :return: integal value.
        :rtype: float.
        """
        return sum([self._calc_integral(f, self.call_pricer,
                                        init_swap_rate, max_call_range)
                    for f in self.call_integrands])

    def eval(self, init_swap_rate, min_put_range, max_call_range):
        """eval
        evaluates expectation by the replication method.

        :param init_swap_rate:
        :param min_put_range:
        :param max_call_range:
        :return: value of expectation.
        :rtype: float.
        """
        analytic_term = self._calc_analytic_term(init_swap_rate)
        put_integral_term = self._calc_put_integral(
            init_swap_rate, min_put_range)
        call_integral_term = self._calc_call_integral(
            init_swap_rate, max_call_range)

        return analytic_term + put_integral_term + call_integral_term
