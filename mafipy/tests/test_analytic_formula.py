#!/bin/python
# -*- coding: utf-8 -*-


from __future__ import division
from pytest import approx
import mafipy.analytic_formula as target
import mafipy.math_formula
import math
import pytest
import scipy.stats


class TestAnalytic(object):

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
        expect = strike / (math.sqrt(maturity) * vol * underlying)
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
        expect = 1.0 / (math.sqrt(maturity) * vol * underlying)
        actual = target.d_fprime_by_strike(
            underlying, strike, rate, maturity, vol)
        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, expect",
        [
            # underlying = strike
            (1.0, 1.0, 1.0, 1.0, 1.0, 0.678817974887),
            # underlying > strike
            (2.0, 1.0, 1.0, 1.0, 1.0, 1.646647107929),
            # underlying < maturity
            (1.0, 2.0, 1.0, 1.0, 1.0, 0.478587969669),
            # maturity < 0
            (1.0, 1.0, 1.0, -1.0, 1.0, 0.0),
            # underlying < 0
            (-1.0, -2.0, 1.0, 1.0, 1.0, (-1.0 + 2.0) + 0.478587969669),
            # underlying = 0
            (0.0, 2.0, 1.0, 1.0, 1.0, 0.0),
            # strike < 0
            (1.0, -2.0, 1.0, 1.0, 1.0, 3.0),
            # vol < 0
            (2.0, 1.0, 1.0, 1.0, -1.0, 1.0),
        ])
    def test_calc_black_scholes_call_formula(
            self, underlying, strike, rate, maturity, vol, expect):
        actual = target.calc_black_scholes_call_formula(
            underlying, strike, rate, maturity, vol)
        assert actual == approx(expect)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol",
        [
            # underlying = strike
            (1.0, 1.0, 1.0, 1.0, 1.0),
            # underlying > strike
            (2.0, 1.0, 1.0, 1.0, 1.0),
            # underlying < maturity
            (1.0, 2.0, 1.0, 1.0, 1.0),
            # maturity < 0
            (1.0, 1.0, 1.0, -1.0, 1.0),
            # underlying < 0
            (-1.0, -2.0, 1.0, 1.0, 1.0),
            # underlying = 0
            (0.0, 2.0, 1.0, 1.0, 1.0),
            # strike < 0
            (1.0, -2.0, 1.0, 1.0, 1.0),
            # vol < 0
            (2.0, 1.0, 1.0, 1.0, -1.0),
        ])
    def test_calc_black_scholes_put_formula(
            self, underlying, strike, rate, maturity, vol):
        call_value = target.calc_black_scholes_call_formula(
            underlying, strike, rate, maturity, vol)
        discount = math.exp(-rate * maturity)
        expect = call_value - (underlying - strike * discount)
        actual = target.calc_black_scholes_put_formula(
            underlying, strike, rate, maturity, vol)
        assert actual == approx(expect)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            (2.0, 1.0, 1.0, 1.0, -1.0, 1.0),
        ])
    def test_calc_black_scholes_call_value(
            self, underlying, strike, rate, maturity, vol, today):
        expect = target.calc_black_scholes_call_formula(
            underlying, strike, rate, maturity - today, vol)
        actual = target.calc_black_scholes_call_value(
            underlying, strike, rate, maturity, vol, today)
        assert actual == approx(expect)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            (2.0, 1.0, 1.0, 1.0, -1.0, 1.0),
        ])
    def test_calc_black_scholes_put_value(
            self, underlying, strike, rate, maturity, vol, today):
        expect = target.calc_black_scholes_put_formula(
            underlying, strike, rate, maturity - today, vol)
        actual = target.calc_black_scholes_put_value(
            underlying, strike, rate, maturity, vol, today)
        assert actual == approx(expect)

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
    def test_black_scholes_call_value_fhess_by_strike(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_value_fhess_by_strike(
                    underlying, strike, rate, maturity, vol)
        else:
            norm = scipy.stats.norm
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            d1 = target.func_d1(underlying, strike, rate, maturity, vol)
            d2 = target.func_d2(underlying, strike, rate, maturity, vol)
            d1_density = norm.pdf(d1)
            d1_density_fprime = mafipy.math_formula.norm_pdf_fprime(d1)
            d2_density = norm.pdf(d2)
            d2_density_fprime = mafipy.math_formula.norm_pdf_fprime(d2)
            d_fprime = target.d_fprime_by_strike(
                underlying, strike, rate, maturity, vol)
            d_fhess = target.d_fhess_by_strike(
                underlying, strike, rate, maturity, vol)

            term1 = (underlying * d1_density_fprime * d_fprime * d_fprime)
            term2 = underlying * d1_density * d_fhess
            term3 = 2.0 * d2_density * d_fprime
            term4 = strike * d2_density_fprime * d_fprime * d_fprime
            term5 = strike * d2_density * d_fhess

            expect = term1 + term2 - term3 - term4 - term5

            actual = target.black_scholes_call_value_fhess_by_strike(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, maturity, alpha, beta, rho, nu, expect", [
        ])
    def test_calc_sabr_atm_implied_vol(
            self, underlying, maturity, alpha, beta, rho, nu, expect):
        underlying = 100.0
        maturity = 1.0
        alpha = 1.0
        beta = 1.0
        rho = 0.0
        nu = 1.0
        actual = target.calc_sabr_model_atm_implied_vol(
            underlying, maturity, alpha, beta, rho, nu)
        assert actual == expect


class TestBlackScholesPricerHelper:

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
        self.target_class = target.BlackScholesPricerHelper()
        pass

    # after each test finish
    def teardown(self):
        pass

    @pytest.mark.parametrize("underlying, rate, maturity, vol, today", [
        (1.0, 1.0, 0.1, 1.0, 0.1)
    ])
    def test_make_call_wrt_strike(
            self, underlying, rate, maturity, vol, today):
        expect_func = lambda strike: target.calc_black_scholes_call_value(
            underlying=underlying,
            strike=strike,
            rate=rate,
            maturity=maturity,
            vol=vol,
            today=today)
        actual_func = self.target_class.make_call_wrt_strike(
            underlying, rate, maturity, vol, today)
        assert type(expect_func) == type(actual_func)

    @pytest.mark.parametrize("underlying, rate, maturity, vol, today", [
        (1.0, 1.0, 0.1, 1.0, 0.1)
    ])
    def test_make_put_wrt_strike(
            self, underlying, rate, maturity, vol, today):
        expect_func = lambda strike: target.calc_black_scholes_put_value(
            underlying=underlying,
            strike=strike,
            rate=rate,
            maturity=maturity,
            vol=vol,
            today=today)
        actual_func = self.target_class.make_put_wrt_strike(
            underlying, rate, maturity, vol, today)
        assert type(expect_func) == type(actual_func)
