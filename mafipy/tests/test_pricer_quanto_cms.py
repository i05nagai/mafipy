#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import mafipy.pricer_quanto_cms as target
import mafipy.analytic_formula as analytic_formula
import mafipy.math_formula as math_formula
import scipy.stats
import pytest
from pytest import approx


class TestPricerQuantoCms(object):

    # before all tests starts
    @classmethod
    def setup_class(cls):
        pass

    # after all tests finish
    @classmethod
    def teardown_class(cls):
        pass

    # before each test start
    def setup(self):
        pass

    # after each test finish
    def teardown(self):
        pass

    @pytest.mark.parametrize("underlying, strike, rate, maturity, vol, expect", [
        # maturity < 0 raise AssertionError
        (2.0, 1.0, 1.0, -1.0, 1.0, 1.0),
        # vol < 0 raise AssertionError
        (2.0, 1.0, 1.0, 1.0, -1.0, 1.0),
        (2.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    ])
    def test_make_pdf_black_scholes(
            self, underlying, strike, rate, maturity, vol, expect):
        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.make_pdf_black_scholes(
                    underlying, rate, maturity, vol)
        else:
            expect = analytic_formula.black_scholes_call_value_fhess_by_strike(
                underlying, strike, rate, maturity, vol)
            actual = target.make_pdf_black_scholes(
                underlying, rate, maturity, vol)(strike)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, expect", [
            # maturity < 0 raise AssertionError
            (2.0, 1.0, 1.0, -1.0, 1.0, 1.0),
            # vol < 0 raise AssertionError
            (2.0, 1.0, 1.0, 1.0, -1.0, 1.0),
            # otherwise
            (2.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        ])
    def test_make_pdf_fprime_black_scholes(
            self, underlying, strike, rate, maturity, vol, expect):
        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.make_pdf_fprime_black_scholes(
                    underlying, rate, maturity, vol)
        else:
            expect = analytic_formula.black_scholes_call_value_third_by_strike(
                underlying, strike, rate, maturity, vol)
            actual = target.make_pdf_fprime_black_scholes(
                underlying, rate, maturity, vol)(strike)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, expect", [
            # maturity < 0 raise AssertionError
            (2.0, 1.0, 1.0, -1.0, 1.0, 1.0),
            # vol < 0 raise AssertionError
            (2.0, 1.0, 1.0, 1.0, -1.0, 1.0),
            # otherwise
            (2.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        ])
    def test_make_cdf_black_scholes(
            self, underlying, strike, rate, maturity, vol, expect):
        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.make_cdf_black_scholes(
                    underlying, rate, maturity, vol)
        else:
            af = analytic_formula
            expect = (1.0 + af.black_scholes_call_value_fprime_by_strike(
                underlying, strike, rate, maturity, vol))
            actual = target.make_cdf_black_scholes(
                underlying, rate, maturity, vol)(strike)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "swap_rate", [
            (2.0),
        ])
    def test__calc_h(self, swap_rate):
        def swap_rate_cdf(swap_rate):
            return (swap_rate * 0.9) / swap_rate
        norm = scipy.stats.norm
        expect = norm.ppf(swap_rate_cdf(swap_rate))

        actual = target._calc_h(swap_rate_cdf, swap_rate)
        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "swap_rate", [
            (2.0),
        ])
    def test__calc_h_fprime(self, swap_rate):
        norm = scipy.stats.norm

        def swap_rate_cdf(swap_rate):
            return norm.cdf(swap_rate)

        def swap_rate_pdf(swap_rate):
            return norm.pdf(swap_rate)

        h = target._calc_h(swap_rate_cdf, swap_rate)

        expect = swap_rate_pdf(swap_rate) / norm.pdf(h)
        actual = target._calc_h_fprime(swap_rate_pdf, swap_rate, h)
        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "swap_rate", [
            (2.0),
        ])
    def test__calc_h_fhess(self, swap_rate):
        norm = scipy.stats.norm

        def swap_rate_cdf(s):
            return norm.cdf(s)

        def swap_rate_pdf(s):
            return norm.pdf(s)

        def swap_rate_pdf_fprime(s):
            return -s * norm.pdf(s)

        h = target._calc_h(swap_rate_cdf, swap_rate)
        h_fprime = target._calc_h_fprime(swap_rate_pdf, swap_rate, h)

        # expect
        term1 = swap_rate_pdf_fprime(swap_rate) * norm.pdf(h)
        term2 = (swap_rate_pdf(swap_rate)
                 * math_formula.norm_pdf_fprime(h) * h_fprime)
        denominator = norm.pdf(h) ** 2
        expect = (term1 - term2) / denominator
        # actual
        actual = target._calc_h_fhess(
            swap_rate_pdf_fprime, swap_rate_pdf, swap_rate, h, h_fprime)
        assert expect == approx(actual)


class TestSimpleQuantoCmsPricer(object):

    # before all tests starts
    @classmethod
    def setup_class(cls):
        pass

    # after all tests finish
    @classmethod
    def teardown_class(cls):
        pass

    # before each test start
    def setup(self):
        pass

    # after each test finish
    def teardown(self):
        pass

    def test_(self):
        pass
