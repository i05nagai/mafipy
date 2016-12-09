#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import mafipy.pricer_quanto_cms as target
import mafipy.analytic_formula as analytic_formula
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
