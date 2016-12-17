#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import scipy.integrate
import numpy as np
import functools
from . import util


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


class Replication(object):
    """Replication
    Interface class to the class which uses replication method.
    """

    def eval(self):
        """eval
        evaluates value of derivative by replication methods.
        """
        raise NotImplementedError


class SimpleReplication(Replication):

    def __init__(self,
                 call_pricer,
                 put_pricer,
                 payoff_func,
                 payoff_fhess):
        self.call_pricer = call_pricer
        self.put_pricer = put_pricer
        self.payoff_func = payoff_func
        self.payoff_fhess = self.payoff_fhess

    def _first_integrand(self, strike):
        return self.put_pricer(strike) * self.weight(strike)

    def _second_integrand(self, strike):
        return self.call_pricer(strike) * self.weight(strike)

    def weight(self, strike):
        return self.payoff_fhess(strike)

    def eval(self, init_swap_rate):
        integrate = scipy.integrate

        first_term = self.payoff_func(init_swap_rate)
        second_term = integrate.quad(
            self._first_integrand, -np.inf, init_swap_rate)
        third_term = integrate.quad(
            self._second_integrand, init_swap_rate, np.inf)
        return first_term + second_term + third_term


class AnalyticReplication(Replication):

    def __init__(self,
                 call_pricer,
                 put_pricer,
                 analytic_funcs=[],
                 put_integrands=[],
                 call_integrands=[]):
        self.call_pricer = call_pricer
        self.put_pricer = put_pricer
        self.analytic_funcs = analytic_funcs
        self.put_integrands = put_integrands
        self.call_integrands = call_integrands

    def _calc_analytic_term(self, init_swap_rate):
        return sum([f(init_swap_rate) for f in self.analytic_funcs])

    def _calc_integral(self, func, pricer, lower, upper):
        integrate = scipy.integrate
        result = integrate.quad(lambda x: func(x) * pricer(x), lower, upper)
        return result[0]

    def _calc_put_integral(self, init_swap_rate, min_put_range):
        return sum([self._calc_integral(f, self.put_pricer,
                                        min_put_range, init_swap_rate)
                    for f in self.put_integrands])

    def _calc_call_integral(self, init_swap_rate, max_call_range):
        return sum([self._calc_integral(f, self.call_pricer,
                                        init_swap_rate, max_call_range)
                    for f in self.call_integrands])

    def eval(self, init_swap_rate, min_put_range, max_call_range):
        analytic_term = self._calc_analytic_term(init_swap_rate)
        put_integral_term = self._calc_put_integral(
            init_swap_rate, min_put_range)
        call_integral_term = self._calc_call_integral(
            init_swap_rate, max_call_range)

        return analytic_term + put_integral_term + call_integral_term
