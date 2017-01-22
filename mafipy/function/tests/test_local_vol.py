#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from pytest import approx

import mafipy.function.local_vol as target


class TestLocalVol(object):

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
            assert expect == approx(actual)
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

            assert expect == approx(actual)
        case2()
