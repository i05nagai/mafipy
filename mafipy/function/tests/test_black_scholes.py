#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pytest import approx
import math
import numpy as np
import pytest
import scipy.stats

import mafipy.function as function
import mafipy.function.black_scholes as target


class TestBlackScholes(object):

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

    # -------------------------------------------------------------------------
    # Black scholes european call/put
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "underlying, strike, vol, expect", [
            # underlying = 0
            (0.0, 2.0, 1.0, True),
            # underlying != 0 and strike < 0
            (1.0, -1.0, 1.0, True),
            # underlying != 0 and strike > 0 and vol < 0
            (1.0, 2.0, -1.0, True),
            # underlying != 0 and strike > 0 and vol > 0
            (1.0, 2.0, 1.0, False),
        ])
    def test__is_d1_or_d2_infinity(self, underlying, strike, vol, expect):
        actual = target._is_d1_or_d2_infinity(underlying, strike, vol)
        assert expect == actual

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol", [
            # underlying = strike
            (2.0, 2.0, 1.0, 1.0, 1.0),
            # underlying > strike
            (2.0, 1.0, 1.0, 1.0, 1.0),
            # underlying < strike
            (1.0, 2.0, 1.0, 1.0, 1.0),
        ])
    def test_func_d1(self, underlying, strike, rate, maturity, vol):
        expect_numerator = (math.log(underlying / strike)
                            + (rate + vol * vol * 0.5) * maturity)
        expect_denominator = vol * math.sqrt(maturity)
        expect = expect_numerator / expect_denominator
        actual = target.func_d1(underlying, strike, rate, maturity, vol)
        assert actual == approx(expect)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol", [
            # underlying = strike
            (2.0, 2.0, 1.0, 1.0, 1.0),
            # underlying > strike
            (2.0, 1.0, 1.0, 1.0, 1.0),
            # underlying < strike
            (1.0, 2.0, 1.0, 1.0, 1.0),
        ])
    def test_func_d2(self, underlying, strike, rate, maturity, vol):
        expect_numerator = (math.log(underlying / strike)
                            + (rate - vol * vol * 0.5) * maturity)
        expect_denominator = vol * math.sqrt(maturity)
        expect = expect_numerator / expect_denominator
        actual = target.func_d2(underlying, strike, rate, maturity, vol)
        assert actual == approx(expect)

    def test_d_fprime_by_strike(self):
        underlying = 2.0
        strike = 1.0
        rate = 1.0
        vol = 0.0001
        maturity = 1.0 / 365.0
        expect = -1.0 / (math.sqrt(maturity) * vol * strike)
        actual = target.d_fprime_by_strike(
            underlying, strike, rate, maturity, vol)
        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol",
        [
            (2.0, 1.0, 1.0, 1.0 / 365.0, 0.0001),
        ])
    def test_d_fhess_by_strike(
            self, underlying, strike, rate, maturity, vol):
        expect = 1.0 / (math.sqrt(maturity) * vol * strike * strike)
        actual = target.d_fhess_by_strike(
            underlying, strike, rate, maturity, vol)
        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, expect",
        [
            # underlying = strike
            (1.0, 1.0, 1.0, 1.0, 1.0, 0.678817974887),
            # underlying > strike
            (2.0, 1.0, 1.0, 1.0, 1.0, 1.646647107929),
            # underlying < strike
            (1.0, 2.0, 1.0, 1.0, 1.0, 0.478587969669),
        ])
    def test_black_scholes_call_formula(
            self, underlying, strike, rate, maturity, vol, expect):
        actual = target.black_scholes_call_formula(
            underlying, strike, rate, maturity, vol)
        assert actual == approx(expect)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol",
        [
            # underlying > strike
            (2.0, 1.0, 1.0, 1.0, 1.0),
        ])
    def test_black_scholes_put_formula(
            self, underlying, strike, rate, maturity, vol):
        call_value = target.black_scholes_call_formula(
            underlying, strike, rate, maturity, vol)
        discount = math.exp(-rate * maturity)
        expect = call_value - (underlying - strike * discount)
        actual = target.black_scholes_put_formula(
            underlying, strike, rate, maturity, vol)
        assert actual == approx(expect)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # maturity < 0
            (1.0, 1.0, 1.0, -1.0, 1.0, 0.0),
            # maturity = 0
            (1.0, 1.0, 1.0, 0.0, 1.0, 0.0),
            # underlying > 0, strike < 0
            (1.0, -1.0, 1.0, 1.0, 1.0, 0.0),
            # underlying < 0, strike > 0
            (-1.0, 1.0, 1.0, 1.0, 1.0, 0.0),
            # underlying < 0, stirke < 0
            (-1.0, -2.0, 1.0, 1.0, 1.0, 0.0),
            # underlying = 0
            (0.0, 1.1, 1.2, 1.3, 1.4, 0.2),
            # stirke = 0, underlying > 0
            (1.1, 0.0, 1.2, 1.3, 1.4, 0.2),
            # stirke = 0, underlying < 0
            (-1.1, 0.0, 1.2, 1.3, 1.4, 0.2),
            # vol < 0
            (2.0, 1.0, 1.0, 1.0, -1.0, 1.0),
            # today > 0
            (2.0, 1.0, 1.0, 1.0, -1.0, 1.0),
        ])
    def test_black_scholes_call_value(
            self, underlying, strike, rate, maturity, vol, today):
        time = maturity - today
        if vol <= 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_value(
                    underlying, strike, rate, maturity, vol, today)
        else:
            actual = target.black_scholes_call_value(
                underlying, strike, rate, maturity, vol, today)
            if maturity < 0 or np.isclose(maturity, 0.0):
                expect = 0.0
                assert expect == approx(actual)
            # never below strike
            elif underlying > 0.0 and strike < 0.0:
                expect = underlying - math.exp(-rate * time) * strike
                assert expect == approx(actual)
            # never beyond strike
            elif underlying < 0.0 and strike > 0.0:
                expect = 0.0
                assert expect == approx(actual)
            # underlying and strike are negative
            elif underlying < 0.0 and strike < 0.0:
                expect = (-1.0 + 2.0) + 0.478587969669
                assert expect == approx(actual)
            # underlying = 0
            elif np.isclose(underlying, 0.0):
                assert 0.0 == approx(actual)
            # strike = 0, underlying > 0
            elif np.isclose(strike, 0.0) and underlying > 0.0:
                expect = underlying * math.exp(-rate * today)
                assert expect == approx(actual)
            # strike = 0, underlying < 0
            elif np.isclose(strike, 0.0) and underlying < 0.0:
                expect = 0.0
                assert expect == approx(actual)
            else:
                expect = target.black_scholes_call_formula(
                    underlying, strike, rate, time, vol)
                assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # maturity < 0
            (1.0, 1.0, 1.0, -1.0, 1.0, 0.0),
            # maturity = 0
            (1.0, 1.0, 1.0, 0.0, 1.0, 0.0),
            # stirke = 0, underlying > 0
            (1.1, 0.0, 1.2, 1.3, 1.4, 0.2),
            # stirke = 0, underlying < 0
            (-1.1, 0.0, 1.2, 1.3, 1.4, 0.2),
            # underlying = 0, strike > 0
            (0.0, 1.1, 1.2, 1.3, 1.4, 0.2),
            # underlying = 0, strike < 0
            (0.0, -1.1, 1.2, 1.3, 1.4, 0.2),
            # otherwise
            (2.1, 1.2, 1.3, 1.4, 1.5, 0.5),
        ])
    def test_black_scholes_put_value(
            self, underlying, strike, rate, maturity, vol, today):
        time = maturity - today
        call_value = target.black_scholes_call_value(
            underlying, strike, rate, maturity, vol, today)
        discount = math.exp(-rate * time)
        expect = call_value - (underlying - discount * strike)
        actual = target.black_scholes_put_value(
            underlying, strike, rate, maturity, vol, today)
        if time < 0.0 or np.isclose(time, 0.0):
            expect = 0.0
        elif np.isclose(strike, 0.0) and underlying > 0.0:
            # max(-|S|, 0)
            expect = 0.0
        elif np.isclose(strike, 0.0) and underlying < 0.0:
            # max(|S|, 0)
            expect = underlying * math.exp(-rate * today)
        elif np.isclose(underlying, 0.0) and strike > 0.0:
            # max(K, 0)
            expect = strike * math.exp(-rate * time)
        elif np.isclose(underlying, 0.0) and strike < 0.0:
            # max(-|K|, 0)
            expect = 0

        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # underlying > strike
            (2.0, 1.0, 1.0, 1.0, 0.1, 0.0),
            # underlying == strike
            (1.0, 1.0, 1.0, 1.0, 0.1, 0.0),
            # underlying < strike
            (1.0, 2.0, 1.0, 1.0, 0.1, 0.0),
            # maturity < 0 raise AssertionError
            (1.0, 2.0, 1.0, -1.0, 0.1, 0.0),
            # vol < 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.0, -0.1, 0.0),
        ])
    def test_black_scholes_call_value_fprime_by_strike(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_value_fprime_by_strike(
                    underlying, strike, rate, maturity, vol)
        else:
            norm = scipy.stats.norm
            d1 = target.func_d1(underlying, strike, rate, maturity, vol)
            d2 = target.func_d2(underlying, strike, rate, maturity, vol)
            d_fprime = target.d_fprime_by_strike(
                underlying, strike, rate, maturity, vol)
            discount = math.exp(-rate * maturity)

            term1 = underlying * norm.pdf(d1) * d_fprime
            term2 = discount * norm.cdf(d2)
            term3 = discount * strike * norm.pdf(d2) * d_fprime
            expect = term1 - term2 - term3

            actual = target.black_scholes_call_value_fprime_by_strike(
                underlying, strike, rate, maturity, vol)

            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # vol <= 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.0, -0.1, 0.0),
            # maturity < 0
            (1.0, 2.0, 1.0, -1.0, 0.1, 0.0),
            # underlying > strike
            (2.0, 1.0, 1.0, 1.0, 0.1, 0.0),
            # underlying == strike
            (1.0, 1.0, 1.0, 1.0, 0.1, 0.0),
            # underlying < strike
            (1.0, 2.0, 1.0, 1.0, 0.1, 0.0),
        ])
    def test_black_scholes_call_value_fhess_by_strike(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_value_fhess_by_strike(
                    underlying, strike, rate, maturity, vol)
        elif maturity < 0.0:
            expect = 0.0
            actual = target.black_scholes_call_value_fhess_by_strike(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)
        else:
            norm = scipy.stats.norm
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            discount = math.exp(-rate * maturity)
            d2 = target.func_d2(underlying, strike, rate, maturity, vol)
            d2_fprime = target.d_fprime_by_strike(
                underlying, strike, rate, maturity, vol)
            d2_density = norm.pdf(d2)
            expect = -discount * d2_density * d2_fprime

            actual = target.black_scholes_call_value_fhess_by_strike(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # underlying > strike
            (2.0, 1.0, 1.0, 1.0, 0.1, 0.0),
            # underlying == strike
            (1.0, 1.0, 1.0, 1.0, 0.1, 0.0),
            # underlying < strike
            (1.0, 2.0, 1.0, 1.0, 0.1, 0.0),
            # maturity < 0 raise AssertionError
            (1.0, 2.0, 1.0, -1.0, 0.1, 0.0),
            # vol < 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.1, -0.1, 0.0),
        ])
    def test_black_scholes_call_value_third_by_strike(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_value_third_by_strike(
                    underlying, strike, rate, maturity, vol)
            return
        if maturity < 0.0:
            expect = 0.0
            actual = target.black_scholes_call_value_third_by_strike(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)
        else:
            norm = scipy.stats.norm
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            d2 = target.func_d2(underlying, strike, rate, maturity, vol)
            d2_density = norm.pdf(d2)
            d2_density_fprime = function.norm_pdf_fprime(d2)
            d_fprime = target.d_fprime_by_strike(
                underlying, strike, rate, maturity, vol)
            d_fhess = target.d_fhess_by_strike(
                underlying, strike, rate, maturity, vol)

            term1 = d2_density_fprime * d_fprime * d_fprime
            term2 = d2_density * d_fhess
            discount = math.exp(-rate * maturity)
            expect = -discount * (term1 + term2)

            actual = target.black_scholes_call_value_third_by_strike(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)

    # -------------------------------------------------------------------------
    # Black scholes greeks
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # vol < 0 raise AssertionError
            (1.1, 1.2, 0.2, 1.3, -0.1, 0.0),
            # maturity < 0, returns 0
            (1.1, 1.2, 0.2, -1.3, 0.1, 0.0),
            # otherwise
            (1.1, 1.2, 0.2, 1.3, 0.1, 0.0),
        ])
    def test_black_scholes_call_delta(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_delta(
                    underlying, strike, rate, maturity, vol)
            return
        elif maturity < 0.0:
            actual = target.black_scholes_call_delta(
                underlying, strike, rate, maturity, vol)
            assert 0.0 == approx(actual)
        else:
            shock = 1e-6
            value_plus = function.black_scholes_call_value(
                underlying + shock, strike, rate, maturity, vol)
            value_minus = function.black_scholes_call_value(
                underlying - shock, strike, rate, maturity, vol)
            expect = (value_plus - value_minus) / (2.0 * shock)

            actual = target.black_scholes_call_delta(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # vol < 0 raise AssertionError
            (1.1, 1.2, 0.2, 1.3, -0.1, 0.0),
            # maturity < 0 returns 0
            (1.1, 1.2, 0.2, -1.3, 0.1, 0.0),
            # otherwise
            (1.1, 1.2, 0.2, 1.3, 0.1, 0.0),
        ])
    def test_black_scholes_call_gamma(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_gamma(
                    underlying, strike, rate, maturity, vol)
        else:
            shock = 1e-6
            value_plus = function.black_scholes_call_delta(
                underlying + shock, strike, rate, maturity, vol)
            value_minus = function.black_scholes_call_delta(
                underlying - shock, strike, rate, maturity, vol)
            expect = (value_plus - value_minus) / (2.0 * shock)

            actual = target.black_scholes_call_gamma(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # vol < 0 raise AssertionError
            (1.1, 1.2, 0.3, 1.4, -0.1, 0.0),
            # maturity < 0 returns 0
            (1.1, 1.1, 0.2, -1.3, 0.1, 0.0),
            # otherwise
            (1.1, 1.2, 0.3, 1.4, 0.1, 0.0),
        ])
    def test_black_scholes_call_vega(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_vega(
                    underlying, strike, rate, maturity, vol)
        elif maturity < 0.0:
            actual = target.black_scholes_call_vega(
                underlying, strike, rate, maturity, vol)
            assert 0.0 == approx(actual)
        else:
            shock = 1e-6
            value_plus = function.black_scholes_call_value(
                underlying, strike, rate, maturity, vol + shock)
            value_minus = function.black_scholes_call_value(
                underlying, strike, rate, maturity, vol - shock)
            expect = (value_plus - value_minus) / (2.0 * shock)

            actual = target.black_scholes_call_vega(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol",
        [
            # vol < 0 raise AssertionError
            (1.1, 1.2, 0.3, 1.4, -0.1),
            # maturity <= 0 return 0
            (1.1, 1.2, 0.2, -1.3, 0.1),
            # otherwise
            (1.1, 1.2, 0.3, 1.4, 0.1),
        ])
    def test_black_scholes_call_volga(
            self, underlying, strike, rate, maturity, vol):

        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_volga(
                    underlying, strike, rate, maturity, vol)
        elif maturity <= 0.0:
            expect = 0
            actual = target.black_scholes_call_volga(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)
        else:
            shock = 1e-6
            value_plus = function.black_scholes_call_vega(
                underlying, strike, rate, maturity, vol + shock)
            value_minus = function.black_scholes_call_vega(
                underlying, strike, rate, maturity, vol - shock)
            expect = (value_plus - value_minus) / (2.0 * shock)

            actual = target.black_scholes_call_volga(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # maturity < 0 raise AssertionError
            (1.1, 1.2, 0.2, -1.3, 0.1, 0.0),
            # vol < 0 raise AssertionError
            (1.1, 1.2, 0.2, 1.3, -0.1, 0.0),
            # otherwise
            (1.1, 1.2, 0.2, 1.3, 0.1, 0.0),
        ])
    def test_black_scholes_call_theta(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_theta(
                    underlying, strike, rate, maturity, vol, today)
        else:
            shock = 1e-6
            value_plus = function.black_scholes_call_value(
                underlying, strike, rate, maturity, vol, today + shock)
            value_minus = function.black_scholes_call_value(
                underlying, strike, rate, maturity, vol, today - shock)
            expect = (value_plus - value_minus) / (2.0 * shock)

            actual = target.black_scholes_call_theta(
                underlying, strike, rate, maturity, vol, today)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # maturity < 0 raise AssertionError
            (1.1, 1.2, 0.2, -1.3, 0.1, 0.0),
            # vol < 0 raise AssertionError
            (1.1, 1.2, 0.2, 1.3, -0.1, 0.0),
            # otherwise
            (1.1, 1.2, 0.2, 1.3, 0.1, 0.0),
        ])
    def test_black_scholes_call_rho(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_rho(
                    underlying, strike, rate, maturity, vol, today)
        else:
            shock = 1e-6
            value_plus = function.black_scholes_call_value(
                underlying, strike, rate + shock, maturity, vol, today)
            value_minus = function.black_scholes_call_value(
                underlying, strike, rate - shock, maturity, vol, today)
            expect = (value_plus - value_minus) / (2.0 * shock)

            actual = target.black_scholes_call_rho(
                underlying, strike, rate, maturity, vol, today)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol",
        [
            # vol < 0 raise AssertionError
            (1.1, 1.2, 0.3, 1.4, -0.1),
            # maturity <= 0 return 0
            (1.1, 1.2, 0.2, -1.3, 0.1),
            # otherwise
            (1.1, 1.2, 0.3, 1.4, 0.1),
        ])
    def test_black_scholes_call_vega_fprime_by_strike(
            self, underlying, strike, rate, maturity, vol):

        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_vega_fprime_by_strike(
                    underlying, strike, rate, maturity, vol)
        elif maturity <= 0.0:
            expect = 0
            actual = target.black_scholes_call_vega_fprime_by_strike(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)
        else:
            shock = 1e-6
            value_plus = function.black_scholes_call_vega(
                underlying, strike + shock, rate, maturity, vol)
            value_minus = function.black_scholes_call_vega(
                underlying, strike - shock, rate, maturity, vol)
            expect = (value_plus - value_minus) / (2.0 * shock)

            actual = target.black_scholes_call_vega_fprime_by_strike(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)

    # -------------------------------------------------------------------------
    # Black scholes distributions
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol",
        [
            # vol < 0 raise AssertionError
            (1.1, 2.2, 1.3, 1.4, -0.1),
            # otherwise
            (1.1, 2.2, 1.3, 1.4, 0.1),
        ])
    def test_black_scholes_cdf(
            self, underlying, strike, rate, maturity, vol):

        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_cdf(
                    underlying, strike, rate, maturity, vol)
        else:
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            expect = (1.0
                      + function.black_scholes_call_value_fprime_by_strike(
                          underlying,
                          strike,
                          rate,
                          maturity,
                          vol) * math.exp(rate * maturity))

            actual = target.black_scholes_cdf(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)
