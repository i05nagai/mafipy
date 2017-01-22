#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from pytest import approx
import pytest

from . import util
import mafipy.function as target


class TestPayoff(object):

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
        "underlying, strike, gearing",
        [
            # underlying = strike
            (2.0, 2.0, 1.0),
            # underlying > strike
            (3.0, 2.0, 1.0),
            # underlying < strike
            (1.0, 2.0, 1.0),
            # gearing = 2
            (2.0, 1.0, 2.0),
        ])
    def test_payoff_call(self, underlying, strike, gearing):
        expect = gearing * max(underlying - strike, 0.0)
        actual = target.payoff_call(underlying, strike, gearing)
        assert(expect == approx(actual))

    @pytest.mark.parametrize(
        "underlying, strike, gearing",
        [
            # underlying = strike
            (2.0, 2.0, 1.0),
            # underlying > strike
            (3.0, 2.0, 1.0),
            # underlying < strike
            (1.0, 2.0, 1.0),
            # gearing = 2
            (2.0, 1.0, 2.0),
        ])
    def test_payoff_call_fprime(self, underlying, strike, gearing):
        expect = 0.0
        if underlying > strike:
            expect = gearing
        actual = target.payoff_call_fprime(underlying, strike, gearing)
        assert(expect == approx(actual))

    @pytest.mark.parametrize(
        "underlying, strike, gearing",
        [
            # underlying = strike
            (2.0, 2.0, 1.0),
            # underlying > strike
            (3.0, 2.0, 1.0),
            # underlying < strike
            (1.0, 2.0, 1.0),
            # gearing = 2
            (2.0, 1.0, 2.0),
        ])
    def test_payoff_put(self, underlying, strike, gearing):
        expect = gearing * max(strike - underlying, 0.0)
        actual = target.payoff_put(underlying, strike, gearing)
        assert(expect == approx(actual))

    @pytest.mark.parametrize(
        "underlying, strike, gearing",
        [
            # underlying = strike
            (2.0, 2.0, 1.0),
            # underlying > strike
            (3.0, 2.0, 1.0),
            # underlying < strike
            (1.0, 2.0, 1.0),
            # gearing = 2
            (2.0, 1.0, 2.0),
        ])
    def test_payoff_put_fprime(self, underlying, strike, gearing):
        expect = 0.0
        if underlying < strike:
            expect = -gearing
        actual = target.payoff_put_fprime(underlying, strike, gearing)
        assert(expect == approx(actual))

    @pytest.mark.parametrize(
        "underlying, lower_strike, upper_strike, gearing",
        [
            # underlying <= lower_strike,
            (1.0, 1.0, 2.0, 1.0),
            # lower_strike < underlying < upper_strike,
            (1.5, 1.0, 2.0, 1.0),
            # underlying >= upper_strike
            (2.0, 1.0, 2.0, 1.0),
            # lower_strike >= upper_strike
            (2.0, 1.0, 1.0, 1.0),
            # gearing = 2
            (1.5, 1.0, 2.0, 2.0),
        ])
    def test_payoff_bull_spread(self,
                                underlying,
                                lower_strike,
                                upper_strike,
                                gearing):
        expect = (target.payoff_call(underlying, lower_strike, gearing)
                  - target.payoff_call(underlying, upper_strike, gearing))
        if lower_strike >= upper_strike:
            expect = 0.0
        actual = target.payoff_bull_spread(
            underlying, lower_strike, upper_strike, gearing)
        assert(expect == approx(actual))

    @pytest.mark.parametrize(
        "underlying, lower_strike, upper_strike, gearing",
        [
            # underlying <= lower_strike,
            (1.0, 1.0, 2.0, 1.0),
            # lower_strike < underlying < upper_strike,
            (1.5, 1.0, 2.0, 1.0),
            # underlying >= upper_strike
            (2.0, 1.0, 2.0, 1.0),
            # lower_strike >= upper_strike
            (2.0, 1.0, 1.0, 1.0),
            # gearing = 2
            (1.5, 1.0, 2.0, 2.0),
        ])
    def test_payoff_bull_spread_fprime(self,
                                       underlying,
                                       lower_strike,
                                       upper_strike,
                                       gearing):
        expect = 0.0
        if lower_strike < underlying < upper_strike:
            expect = gearing
        actual = target.payoff_bull_spread_fprime(
            underlying, lower_strike, upper_strike, gearing)
        assert(expect == approx(actual))

    @pytest.mark.parametrize(
        "underlying, strike, gearing",
        [
            # underlying = strike
            (2.0, 2.0, 1.0),
            # underlying > strike
            (3.0, 2.0, 1.0),
            # underlying < strike
            (1.0, 2.0, 1.0),
            # gearing = 2
            (3.0, 2.0, 2.0),
        ])
    def test_payoff_straddle(self, underlying, strike, gearing):
        expect = (target.payoff_call(underlying, strike, gearing)
                  + target.payoff_put(underlying, strike, gearing))
        actual = target.payoff_straddle(underlying, strike, gearing)
        assert(expect == approx(actual))

    @pytest.mark.parametrize(
        "underlying, lower_strike, upper_strike, gearing",
        [
            # underlying <= lower_strike,
            (1.0, 1.0, 2.0, 1.0),
            # lower_strike < underlying < upper_strike,
            (1.5, 1.0, 2.0, 1.0),
            # underlying >= upper_strike
            (2.0, 1.0, 2.0, 1.0),
            # lower_strike >= upper_strike
            (2.0, 1.0, 1.0, 1.0),
            # gearing = 2
            (1.5, 1.0, 2.0, 2.0),
        ])
    def test_payoff_strangle(self,
                             underlying,
                             lower_strike,
                             upper_strike,
                             gearing):
        expect = (target.payoff_put(underlying, lower_strike, gearing)
                  + target.payoff_call(underlying, upper_strike, gearing))
        if lower_strike >= upper_strike:
            return 0.0
        actual = target.payoff_strangle(
            underlying, lower_strike, upper_strike, gearing)
        assert(expect == approx(actual))

    @pytest.mark.parametrize(
        "underlying, spot_price, spread, gearing",
        [
            # underlying <= spot_price - spread
            (1.0, 2.0, 1.0, 1.0),
            # spot_price - spread < underlying < spot_price
            (1.5, 2.0, 1.0, 1.0),
            # spot_price + spread > underlying > spot_price
            (2.5, 2.0, 1.0, 1.0),
            # underlying >= spot_price + spread
            (3.0, 2.0, 1.0, 1.0),
            # spread = 0
            (2.0, 2.0, 0.0, 1.0),
            # spread < 0
            (2.0, 1.0, -1.0, 1.0),
            # gearing = 2
            (1.5, 2.0, 1.0, 2.0),
        ])
    def test_payoff_butterfly_spread(self,
                                     underlying,
                                     spot_price,
                                     spread,
                                     gearing):
        expect = (target.payoff_call(underlying, spot_price - spread, gearing)
                  - 2.0 * target.payoff_call(underlying, spot_price, gearing)
                  + target.payoff_call(
                      underlying, spot_price + spread, gearing))
        if spread < 0.0:
            return 0.0
        actual = target.payoff_butterfly_spread(
            underlying, spot_price, spread, gearing)
        assert(expect == approx(actual))

    @pytest.mark.parametrize(
        "underlying, lower_strike, upper_strike, gearing",
        [
            # underlying <= lower_strike,
            (1.0, 1.0, 2.0, 1.0),
            # lower_strike < underlying < upper_strike,
            (1.5, 1.0, 2.0, 1.0),
            # underlying >= upper_strike
            (2.0, 1.0, 2.0, 1.0),
            # lower_strike >= upper_strike
            (2.0, 1.0, 1.0, 1.0),
            # gearing = 2
            (1.5, 1.0, 2.0, 2.0),
        ])
    def test_payoff_risk_riversal(self,
                                  underlying,
                                  lower_strike,
                                  upper_strike,
                                  gearing):
        expect = (-target.payoff_put(underlying, lower_strike, gearing)
                  + target.payoff_call(underlying, upper_strike, gearing))
        if lower_strike > upper_strike:
            return 0.0
        actual = target.payoff_risk_reversal(
            underlying, lower_strike, upper_strike, gearing)
        assert(expect == approx(actual))


class TestBullSpreadUnderlyingPayoffHelper(object):

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
        data = sorted(util.get_real(2))
        self.lower_strike = data[0]
        self.upper_strike = data[1]
        data = util.get_real()
        self.gearing = data[0]
        params = {
            "lower_strike": self.lower_strike,
            "upper_strike": self.upper_strike,
            "gearing": self.gearing
        }
        self.target = target.BullSpreadUnderlyingPayoffHelper(**params)

    # after each test finish
    def teardown(self):
        pass

    def test_make_func(self):

        def case1():
            swap_rate = self.lower_strike
            actual = self.target.make_func()(swap_rate)
            assert 0.0 == approx(actual)
        case1()

        def case2():
            swap_rate = self.upper_strike
            actual = self.target.make_func()(swap_rate)
            expect = (self.upper_strike - self.lower_strike) * self.gearing
            assert expect == approx(actual)
        case2()

        def case3():
            swap_rate = util.get_real()[0]
            actual = self.target.make_func()(swap_rate)
            expect = target.payoff_bull_spread(
                swap_rate, self.lower_strike, self.upper_strike, self.gearing)
            assert expect == approx(actual)
        case3()

    def test_make_fprime(self):
        swap_rate = util.get_real()[0]
        actual = self.target.make_func()(swap_rate)
        expect = target.payoff_bull_spread(
            swap_rate, self.lower_strike, self.upper_strike, self.gearing)
        assert expect == approx(actual)
