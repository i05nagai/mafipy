#!/bin/python
# -*- coding: utf-8 -*-


from __future__ import division
from pytest import approx
import numpy as np
import mafipy.analytic_formula as target
import mafipy.math_formula
import mafipy.tests.util as util
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
    def test_calc_black_scholes_call_formula(
            self, underlying, strike, rate, maturity, vol, expect):
        actual = target.calc_black_scholes_call_formula(
            underlying, strike, rate, maturity, vol)
        assert actual == approx(expect)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol",
        [
            # underlying > strike
            (2.0, 1.0, 1.0, 1.0, 1.0),
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
            # vol < 0
            (2.0, 1.0, 1.0, 1.0, -1.0, 1.0),
            # today > 0
            (2.0, 1.0, 1.0, 1.0, -1.0, 1.0),
        ])
    def test_calc_black_scholes_call_value(
            self, underlying, strike, rate, maturity, vol, today):
        if vol <= 0.0:
            with pytest.raises(AssertionError):
                actual = target.calc_black_scholes_call_value(
                    underlying, strike, rate, maturity, vol, today)
        else:
            actual = target.calc_black_scholes_call_value(
                underlying, strike, rate, maturity, vol, today)
            if maturity < 0 or np.isclose(maturity, 0.0):
                expect = 0.0
                assert expect == approx(actual)
            # never below strike
            elif underlying > 0.0 and strike < 0.0:
                expect = underlying - math.exp(-rate * maturity) * strike
                assert expect == approx(actual)
            # never beyond strike
            elif underlying < 0.0 and strike > 0.0:
                expect = 0.0
                assert expect == approx(actual)
            # underlying and strike are negative
            elif underlying < 0.0:
                expect = (-1.0 + 2.0) + 0.478587969669
                assert expect == approx(actual)
            else:
                expect = target.calc_black_scholes_call_formula(
                    underlying, strike, rate, maturity - today, vol)
                assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            (2.0, 1.0, 1.0, 1.0, 1.0, 0.5),
        ])
    def test_calc_black_scholes_put_value(
            self, underlying, strike, rate, maturity, vol, today):
        call_value = target.calc_black_scholes_call_value(
            underlying, strike, rate, maturity, vol, today)
        discount = math.exp(-rate * (maturity - today))
        expect = call_value - (underlying - discount * strike)
        actual = target.calc_black_scholes_put_value(
            underlying, strike, rate, maturity, vol, today)
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
        ])
    def test_black_scholes_call_value_third_by_strike(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_value_third_by_strike(
                    underlying, strike, rate, maturity, vol)
        else:
            norm = scipy.stats.norm
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            d2 = target.func_d2(underlying, strike, rate, maturity, vol)
            d2_density = norm.pdf(d2)
            d2_density_fprime = mafipy.math_formula.norm_pdf_fprime(d2)
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
    # Black payers/recievers swaption
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, swap_annuity, option_maturity, vol",
        [
            # vol < 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.0, -0.1),
            # maturity < 0
            (1.0, 2.0, 1.0, -1.0, 0.1),
            # maturity = 0
            (1.0, 1.0, 1.0, 0.0, 1.0),
            # underlying > 0, strike < 0
            (1.1, -1.3, 1.2, 1.1, 1.5),
            # underlying < 0, strike > 0
            (-1.1, 1.3, 1.2, 1.1, 1.5),
            # underlying < 0, stirke < 0
            (-1.1, -2.3, 1.2, 1.1, 1.5),
            # otherwise
            (1.0, 2.0, 1.0, 1.0, 0.1),
        ])
    def test_black_payers_swaption_value(self,
                                         init_swap_rate,
                                         option_strike,
                                         swap_annuity,
                                         option_maturity,
                                         vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_payers_swaption_value(
                    init_swap_rate,
                    option_strike,
                    swap_annuity,
                    option_maturity,
                    vol)
            return
        elif option_maturity < 0.0 or np.isclose(option_maturity, 0.0):
            expect = 0.0
        # never below strike
        elif init_swap_rate > 0.0 and option_strike < 0.0:
            expect = swap_annuity * (init_swap_rate - option_strike)
        # never beyond strike
        elif init_swap_rate < 0.0 and option_strike > 0.0:
            expect = 0.0
        # max(S(T)-K,0) = max((-K)-(-S(T)),0)
        elif init_swap_rate < 0.0 and option_strike < 0.0:
            option_value = target.calc_black_scholes_put_formula(
                -init_swap_rate, -option_strike, 0.0, option_maturity, vol)
            expect = swap_annuity * option_value
        else:
            value = target.calc_black_scholes_call_formula(
                init_swap_rate, option_strike, 0.0, option_maturity, vol)
            expect = swap_annuity * value

        actual = target.black_payers_swaption_value(
            init_swap_rate,
            option_strike,
            swap_annuity,
            option_maturity,
            vol)
        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, swap_annuity, option_maturity, vol",
        [
            # vol < 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.0, -0.1),
            # maturity < 0
            (1.1, 2.2, 1.3, -1.4, 0.1),
            # maturity = 0
            (1.1, 1.2, 1.3, 0.0, 1.5),
            # underlying > 0, strike < 0
            (1.1, -1.3, 1.2, 1.1, 1.5),
            # underlying < 0, strike > 0
            (-1.1, 1.3, 1.2, 1.1, 1.5),
            # underlying < 0, stirke < 0
            (-1.1, -2.3, 1.2, 1.1, 1.5),
            # otherwise
            (1.0, 2.0, 1.0, 1.0, 0.1),
        ])
    def test_black_receivers_swaption_value(self,
                                            init_swap_rate,
                                            option_strike,
                                            swap_annuity,
                                            option_maturity,
                                            vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_receivers_swaption_value(
                    init_swap_rate,
                    option_strike,
                    swap_annuity,
                    option_maturity,
                    vol)
            return
        elif option_maturity < 0.0 or np.isclose(option_maturity, 0.0):
            expect = 0.0
        # max(K-S(T),0) = 0
        elif init_swap_rate > 0.0 and option_strike < 0.0:
            expect = 0.0
        # max(K-S(T),0) = K + S(T)
        elif init_swap_rate < 0.0 and option_strike > 0.0:
            expect = swap_annuity * (option_strike - init_swap_rate)
        # max(K-S(T),0) = max((-S(T)) - (-K),0)
        elif init_swap_rate < 0.0 and option_strike < 0.0:
            option_value = target.calc_black_scholes_call_formula(
                -init_swap_rate, -option_strike, 0.0, option_maturity, vol)
            expect = swap_annuity * option_value
        else:
            value = target.calc_black_scholes_put_formula(
                init_swap_rate, option_strike, 0.0, option_maturity, vol)
            expect = swap_annuity * value

        actual = target.black_receivers_swaption_value(
            init_swap_rate,
            option_strike,
            swap_annuity,
            option_maturity,
            vol)
        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, swap_annuity, option_maturity, vol",
        [
            # vol < 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.0, -0.1),
            # maturity < 0
            (1.0, 2.0, 1.0, -1.0, 0.1),
            # otherwise
            (1.0, 2.0, 1.0, 1.0, 0.1),
        ])
    def test_black_payers_swaption_value_fprime_by_strike(self,
                                                          init_swap_rate,
                                                          option_strike,
                                                          swap_annuity,
                                                          option_maturity,
                                                          vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_payers_swaption_value_fprime_by_strike(
                    init_swap_rate,
                    option_strike,
                    swap_annuity,
                    option_maturity,
                    vol)
            return
        elif option_maturity < 0.0 or np.isclose(option_maturity, 0.0):
            expect = 0.0
        else:
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            value = target.black_scholes_call_value_fprime_by_strike(
                init_swap_rate, option_strike, 0.0, option_maturity, vol)
            expect = swap_annuity * value

        actual = target.black_payers_swaption_value_fprime_by_strike(
            init_swap_rate,
            option_strike,
            swap_annuity,
            option_maturity,
            vol)
        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, swap_annuity, option_maturity, vol",
        [
            # vol < 0 raise AssertionError
            (1.1, 2.2, 1.3, 1.4, -0.1),
            # maturity < 0
            (1.1, 2.2, 1.3, -1.4, 0.1),
            # maturity = 0
            (1.1, 1.2, 1.3, 0.0, 1.5),
            # underlying > 0, strike < 0
            (1.1, -1.3, 1.2, 1.1, 1.5),
            # underlying < 0, strike > 0
            (-1.1, 1.3, 1.2, 1.1, 1.5),
            # underlying < 0, stirke < 0
            (-1.1, -2.3, 1.2, 1.1, 1.5),
            # otherwise
            (1.1, 2.2, 1.3, 1.4, 0.1),
        ])
    def test_black_payers_swaption_value_fhess_by_strike(self,
                                                         init_swap_rate,
                                                         option_strike,
                                                         swap_annuity,
                                                         option_maturity,
                                                         vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_payers_swaption_value_fhess_by_strike(
                    init_swap_rate,
                    option_strike,
                    swap_annuity,
                    option_maturity,
                    vol)
            return
        elif option_maturity < 0.0 or np.isclose(option_maturity, 0.0):
            expect = 0.0
        elif init_swap_rate > 0.0 and option_strike < 0.0:
            expect = 0.0
        elif init_swap_rate < 0.0 and option_strike > 0.0:
            expect = 0.0
        elif init_swap_rate < 0.0 and option_strike < 0.0:
            value = target.black_scholes_call_value_fhess_by_strike(
                init_swap_rate,
                option_strike,
                0.0,
                option_maturity,
                vol)
            expect = swap_annuity * value
        else:
            value = target.black_scholes_call_value_fhess_by_strike(
                init_swap_rate,
                option_strike,
                0.0,
                option_maturity,
                vol)
            expect = swap_annuity * value

        actual = target.black_payers_swaption_value_fhess_by_strike(
            init_swap_rate,
            option_strike,
            swap_annuity,
            option_maturity,
            vol)
        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, swap_annuity, option_maturity, vol",
        [
            # vol < 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.0, -0.1),
            # maturity < 0
            (1.0, 2.0, 1.0, -1.0, 0.1),
            # otherwise
            (1.0, 2.0, 1.0, 1.0, 0.1),
        ])
    def test_black_payers_swaption_value_third_by_strike(self,
                                                         init_swap_rate,
                                                         option_strike,
                                                         swap_annuity,
                                                         option_maturity,
                                                         vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_payers_swaption_value_third_by_strike(
                    init_swap_rate,
                    option_strike,
                    swap_annuity,
                    option_maturity,
                    vol)
            return
        elif option_maturity < 0.0 or np.isclose(option_maturity, 0.0):
            expect = 0.0
        else:
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            value = target.black_scholes_call_value_third_by_strike(
                init_swap_rate, option_strike, 0.0, option_maturity, vol)
            expect = swap_annuity * value

        actual = target.black_payers_swaption_value_third_by_strike(
            init_swap_rate,
            option_strike,
            swap_annuity,
            option_maturity,
            vol)
        assert expect == approx(actual)

    # -------------------------------------------------------------------------
    # Black scholes greeks
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # maturity < 0 raise AssertionError
            (1.0, 2.0, 1.0, -1.0, 0.1, 0.0),
            # vol < 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.0, -0.1, 0.0),
            # otherwise
            (1.0, 2.0, 1.0, 1.0, 0.1, 0.0),
        ])
    def test_black_scholes_call_delta(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_delta(
                    underlying, strike, rate, maturity, vol)
        else:
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            d1 = target.func_d1(underlying, strike, rate, maturity, vol)
            expect = scipy.stats.norm.cdf(d1)

            actual = target.black_scholes_call_delta(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # maturity < 0 raise AssertionError
            (1.0, 2.0, 1.0, -1.0, 0.1, 0.0),
            # vol < 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.0, -0.1, 0.0),
            # otherwise
            (1.0, 2.0, 1.0, 1.0, 0.1, 0.0),
        ])
    def test_black_scholes_call_gamma(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_gamma(
                    underlying, strike, rate, maturity, vol)
        else:
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            d1 = target.func_d1(underlying, strike, rate, maturity, vol)
            denominator = (underlying ** 2) * vol * math.sqrt(maturity)
            expect = -scipy.stats.norm.pdf(d1) / denominator

            actual = target.black_scholes_call_gamma(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # maturity < 0 raise AssertionError
            (1.0, 2.0, 1.0, -1.0, 0.1, 0.0),
            # vol < 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.0, -0.1, 0.0),
            # otherwise
            (1.0, 2.0, 1.0, 1.0, 0.1, 0.0),
        ])
    def test_black_scholes_call_vega(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_vega(
                    underlying, strike, rate, maturity, vol)
        else:
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            d1 = target.func_d1(underlying, strike, rate, maturity, vol)
            expect = -(math.sqrt(maturity)
                       * underlying * scipy.stats.norm.pdf(d1))

            actual = target.black_scholes_call_vega(
                underlying, strike, rate, maturity, vol)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # maturity < 0 raise AssertionError
            (1.0, 2.0, 1.0, -1.0, 0.1, 0.0),
            # vol < 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.0, -0.1, 0.0),
            # otherwise
            (1.0, 2.0, 1.0, 1.0, 0.1, 0.0),
        ])
    def test_black_scholes_call_theta(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_theta(
                    underlying, strike, rate, maturity, vol, today)
        else:
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            norm = scipy.stats.norm
            time = maturity - today
            d1 = target.func_d1(underlying, strike, rate, time, vol)
            term1 = underlying * norm.pdf(d1) * vol / (2.0 * math.sqrt(time))
            d2 = target.func_d2(underlying, strike, rate, time, vol)
            term2 = rate * math.exp(-rate * time) * strike * norm.cdf(d2)
            expect = -term1 - term2

            actual = target.black_scholes_call_theta(
                underlying, strike, rate, maturity, vol, today)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, today",
        [
            # maturity < 0 raise AssertionError
            (1.0, 2.0, 1.0, -1.0, 0.1, 0.0),
            # vol < 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.0, -0.1, 0.0),
            # otherwise
            (1.0, 2.0, 1.0, 1.0, 0.1, 0.0),
        ])
    def test_black_scholes_call_rho(
            self, underlying, strike, rate, maturity, vol, today):

        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_scholes_call_rho(
                    underlying, strike, rate, maturity, vol, today)
        else:
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            norm = scipy.stats.norm
            time = maturity - today
            d2 = target.func_d2(underlying, strike, rate, time, vol)
            expect = time * math.exp(-rate * time) * strike * norm.cdf(d2)

            actual = target.black_scholes_call_rho(
                underlying, strike, rate, maturity, vol, today)
            assert expect == approx(actual)

    def test_calc_local_vol_model_implied_vol(self):
        underlying = 1.0
        strike = 0.0
        maturity = 1.0

        # local_vol_func returns 0
        def case1():
            def lv_func(u):
                return 0.0

            def lv_fhess(u):
                return 0.0
            expect = 0.0
            actual = target.calc_local_vol_model_implied_vol(
                underlying, strike, maturity, lv_func, lv_fhess)
            assert expect == actual
        case1()

        # otherwise
        def case2():

            def lv_func(u):
                return u * u

            def lv_fhess(u):
                return 2.0
            average_val = 0.5 * (underlying + strike)
            lv_val = lv_func(average_val)
            lv_val_fhess = lv_fhess(average_val)
            numerator = lv_val_fhess * ((underlying - strike) ** 2)
            denominator = 24.0 * lv_val
            expect = lv_val * (1.0 + numerator / denominator)

            actual = target.calc_local_vol_model_implied_vol(
                underlying, strike, maturity, lv_func, lv_fhess)

            assert expect == actual
        case2()

    # TODO: add more test cases
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_calc_sabr_implied_vol(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        one_minus_beta = 1.0 - beta
        one_minus_beta2 = one_minus_beta ** 2
        one_minus_beta4 = one_minus_beta ** 4
        log_val = math.log(underlying / strike)
        log_val2 = log_val ** 2
        log_val4 = log_val ** 4
        alpha2 = alpha ** 2
        # factor1
        factor11 = (underlying * strike) ** (one_minus_beta * 0.5)
        term11 = 1.0
        term12 = one_minus_beta2 * log_val2 / 24.0
        term13 = one_minus_beta4 * log_val4 / 1920.0
        factor12 = term11 + term12 + term13
        denominator1 = factor11 * factor12
        factor1 = alpha / denominator1
        # factor2
        z = factor11 * log_val * nu / alpha
        x_numerator = math.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho
        x_denominator = 1.0 - rho
        x = math.log(x_numerator / x_denominator)
        factor2 = z / x
        # factor3
        numerator31 = one_minus_beta2 * alpha2
        denominator31 = 24.0 * ((underlying * strike) ** one_minus_beta)
        term31 = numerator31 / denominator31
        numerator32 = rho * beta * nu * alpha
        denominator32 = 4.0 * ((underlying * strike) ** (one_minus_beta * 0.5))
        term32 = numerator32 / denominator32
        numerator33 = (2.0 - 3.0 * rho * rho) * nu * nu
        denominator33 = 24.0
        term33 = numerator33 / denominator33
        factor3 = 1 + (term31 + term32 + term33) * maturity

        expect = factor1 * factor2 * factor3

        # actual
        actual = target.calc_sabr_implied_vol(
            underlying, strike, maturity, alpha, beta, rho, nu)
        assert expect == actual

    # TODO: add more test cases
    @pytest.mark.parametrize(
        "underlying, maturity, alpha, beta, rho, nu", [
            (0.0357, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_calc_sabr_atm_implied_vol(
            self, underlying, maturity, alpha, beta, rho, nu):
        one_minus_beta = 1.0 - beta
        one_minus_beta2 = one_minus_beta ** 2
        alpha2 = alpha ** 2
        # factor1
        factor1 = alpha / (underlying ** one_minus_beta)
        # factor2
        numerator1 = one_minus_beta2 * alpha2
        denominator1 = 24.0 * underlying ** (2.0 * one_minus_beta)
        term1 = numerator1 / denominator1
        numerator2 = rho * beta * alpha * nu
        denominator2 = 4.0 * (underlying ** one_minus_beta)
        term2 = numerator2 / denominator2
        numerator3 = (2.0 - 3.0 * rho * rho) * nu * nu
        denominator3 = 24.0
        term3 = numerator3 / denominator3
        factor2 = 1.0 + (term1 + term2 + term3) * maturity
        # expect
        expect = factor1 * factor2
        # actual
        actual = target.calc_sabr_atm_implied_vol(
            underlying, maturity, alpha, beta, rho, nu)
        assert expect == actual


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


class TestBlackSwaptionPricerHelper(object):

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
        self.target = target.BlackSwaptionPricerHelper()
        pass

    # after each test finish
    def teardown(self):
        pass

    @pytest.mark.parametrize(
        "init_swap_rate, swap_annuity, option_maturity, vol", [
            (1.0, 1.0, 0.1, 1.0)
        ])
    def test_make_payers_swaption_strike(
            self, init_swap_rate, swap_annuity, option_maturity, vol):
        def expect_func(option_strike):
            return target.black_payers_swaption_value(
                init_swap_rate=init_swap_rate,
                option_strike=option_strike,
                swap_annuity=swap_annuity,
                option_maturity=option_maturity,
                vol=vol)
        actual_func = self.target.make_payers_swaption_wrt_strike(
            init_swap_rate, swap_annuity, option_maturity, vol)
        assert type(expect_func) == type(actual_func)

    @pytest.mark.parametrize(
        "init_swap_rate, swap_annuity, option_maturity, vol", [
            (1.0, 1.0, 0.1, 1.0)
        ])
    def test_make_receivers_swaption_wrt_strike(
            self, init_swap_rate, swap_annuity, option_maturity, vol):
        def expect_func(option_strike):
            return target.black_receivers_swaption_value(
                init_swap_rate=init_swap_rate,
                option_strike=option_strike,
                swap_annuity=swap_annuity,
                option_maturity=option_maturity,
                vol=vol)
        actual_func = self.target.make_receivers_swaption_wrt_strike(
            init_swap_rate, swap_annuity, option_maturity, vol)
        assert type(expect_func) == type(actual_func)


class TestModelAnalyticFormula(object):

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
        "underlying, strike, rate, maturity, vol, today", [
            util.get_real(6)
        ])
    def test_put_call_parity_black_scholes(
            self, underlying, strike, rate, maturity, vol, today):

        call_value = target.calc_black_scholes_call_value(
            underlying, strike, rate, maturity, vol, today)
        put_value = target.calc_black_scholes_put_value(
            underlying, strike, rate, maturity, vol, today)
        # forward
        time = maturity - today
        if time < 0.0:
            forward_value = 0.0
        else:
            discount = math.exp(-rate * time)
            forward_value = underlying - discount * strike

        assert forward_value == approx(call_value - put_value)
