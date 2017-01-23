#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from pytest import approx
import math
import pytest

from . import util
import mafipy.function
import mafipy.function.analytic_formula as target


class TestAnalyticFormula(object):

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

    def test_implied_vol_brenner_subrahmanyam(self):
        data = util.get_real(5)
        underlying = data[0]
        strike = data[1]
        rate = data[2]
        maturity = data[3]
        option_value = data[4]
        expect = math.sqrt(2.0 * math.pi / maturity) * option_value / strike
        actual = target.implied_vol_brenner_subrahmanyam(
            underlying, strike, rate, maturity, option_value)
        assert expect == approx(actual)

        # maturity = 0
        maturity = 0.0
        expect = 0.0
        actual = target.implied_vol_brenner_subrahmanyam(
            underlying, strike, rate, maturity, option_value)
        assert expect == approx(actual)

        # maturity < 0
        maturity = -1.0
        expect = 0.0
        actual = target.implied_vol_brenner_subrahmanyam(
            underlying, strike, rate, maturity, option_value)
        assert expect == approx(actual)

        # strike = 0
        strike = 0.0
        expect = 0.0
        actual = target.implied_vol_brenner_subrahmanyam(
            underlying, strike, rate, maturity, option_value)
        assert expect == approx(actual)

    def test_implied_vol_quadratic_approx(self):
        data = util.get_real(5)
        underlying = data[0]
        strike = data[1]
        rate = data[2]
        maturity = data[3]
        option_value = data[4]

        def case1():
            # expect
            discount_strike = math.exp(-rate * maturity) * strike
            moneyness_delta = underlying - discount_strike
            diff = option_value - moneyness_delta / 2.0
            moneyness_delta2 = moneyness_delta ** 2
            sqrt_inner = max(diff ** 2 - moneyness_delta2 / math.pi, 0.0)
            factor1 = diff + math.sqrt(sqrt_inner)
            factor2 = (math.sqrt(2.0 * math.pi / maturity)
                       / (underlying + discount_strike))
            expect = factor1 * factor2
            # actual
            actual = target.implied_vol_quadratic_approx(
                underlying, strike, rate, maturity, option_value)
            assert expect == approx(actual)
        case1()

        def case_maturity_is_not_positive():
            maturity = -1.0
            expect = 0.0
            actual = target.implied_vol_quadratic_approx(
                underlying, strike, rate, maturity, option_value)
            assert expect == approx(actual)
        case_maturity_is_not_positive()

        def case_maturity_is_zero():
            maturity = 0.0
            expect = 0.0
            actual = target.implied_vol_quadratic_approx(
                underlying, strike, rate, maturity, option_value)
            assert expect == approx(actual)
        case_maturity_is_zero()


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
        def expect_func(strike):
            return target.black_scholes_call_value(
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
        def expect_func(strike):
            target.black_scholes_put_value(
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

        call_value = mafipy.function.black_scholes_call_value(
            underlying, strike, rate, maturity, vol, today)
        put_value = mafipy.function.black_scholes_put_value(
            underlying, strike, rate, maturity, vol, today)
        # forward
        time = maturity - today
        if time < 0.0:
            forward_value = 0.0
        else:
            discount = math.exp(-rate * time)
            forward_value = underlying - discount * strike

        assert forward_value == approx(call_value - put_value)
