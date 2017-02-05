#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from pytest import approx
import mafipy.function as function
import mafipy.function.black as target
import numpy as np
import pytest


class TestBlack(object):

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
            option_value = function.black_scholes_put_formula(
                -init_swap_rate, -option_strike, 0.0, option_maturity, vol)
            expect = swap_annuity * option_value
        else:
            value = function.black_scholes_call_formula(
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
            option_value = function.black_scholes_call_formula(
                -init_swap_rate, -option_strike, 0.0, option_maturity, vol)
            expect = swap_annuity * option_value
        else:
            value = function.black_scholes_put_formula(
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
            value = function.black_scholes_call_value_fprime_by_strike(
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
            value = function.black_scholes_call_value_fhess_by_strike(
                init_swap_rate,
                option_strike,
                0.0,
                option_maturity,
                vol)
            expect = swap_annuity * value
        else:
            value = function.black_scholes_call_value_fhess_by_strike(
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
            value = function.black_scholes_call_value_third_by_strike(
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
    # black payer's/reciever's swaption greeks
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, swap_annuity, option_maturity, vol",
        [
            # vol < 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.0, -0.1),
            # maturity < 0
            (1.0, 2.0, 1.0, -1.0, 0.1),
            # otherwise
            (1.1, 1.2, 1.3, 1.4, 0.1),
        ])
    def test_model_black_payers_swaption_delta(self,
                                               init_swap_rate,
                                               option_strike,
                                               swap_annuity,
                                               option_maturity,
                                               vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_payers_swaption_delta(
                    init_swap_rate,
                    option_strike,
                    swap_annuity,
                    option_maturity,
                    vol)
            return
        elif option_maturity < 0.0 or np.isclose(option_maturity, 0.0):
            expect = 0.0
            actual = target.black_payers_swaption_delta(
                init_swap_rate,
                option_strike,
                swap_annuity,
                option_maturity,
                vol)
            assert expect == approx(actual)
        else:
            shock = 1e-6
            value_plus = function.black_payers_swaption_value(
                init_swap_rate + shock, option_strike, swap_annuity,
                option_maturity, vol)
            value_minus = function.black_payers_swaption_value(
                init_swap_rate - shock, option_strike, swap_annuity,
                option_maturity, vol)
            expect = (value_plus - value_minus) / (2.0 * shock)

            actual = target.black_payers_swaption_delta(
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
            (1.1, 1.2, 1.3, 1.4, 0.1),
        ])
    def test_model_black_payers_swaption_vega(self,
                                              init_swap_rate,
                                              option_strike,
                                              swap_annuity,
                                              option_maturity,
                                              vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_payers_swaption_vega(
                    init_swap_rate,
                    option_strike,
                    swap_annuity,
                    option_maturity,
                    vol)
            return
        elif option_maturity < 0.0 or np.isclose(option_maturity, 0.0):
            expect = 0.0
            actual = target.black_payers_swaption_vega(
                init_swap_rate,
                option_strike,
                swap_annuity,
                option_maturity,
                vol)
            assert expect == approx(actual)
        else:
            shock = 1e-6
            value_plus = function.black_payers_swaption_value(
                init_swap_rate, option_strike, swap_annuity,
                option_maturity, vol + shock)
            value_minus = function.black_payers_swaption_value(
                init_swap_rate, option_strike, swap_annuity,
                option_maturity, vol - shock)
            expect = (value_plus - value_minus) / (2.0 * shock)

            actual = target.black_payers_swaption_vega(
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
            (1.1, 1.2, 1.3, 1.4, 0.1),
        ])
    def test_model_black_payers_swaption_volga(self,
                                               init_swap_rate,
                                               option_strike,
                                               swap_annuity,
                                               option_maturity,
                                               vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_payers_swaption_volga(
                    init_swap_rate,
                    option_strike,
                    swap_annuity,
                    option_maturity,
                    vol)
            return
        elif option_maturity < 0.0 or np.isclose(option_maturity, 0.0):
            expect = 0.0
            actual = target.black_payers_swaption_volga(
                init_swap_rate,
                option_strike,
                swap_annuity,
                option_maturity,
                vol)
            assert expect == approx(actual)
        else:
            shock = 1e-6
            value_plus = target.black_payers_swaption_vega(
                init_swap_rate, option_strike, swap_annuity,
                option_maturity, vol + shock)
            value_minus = target.black_payers_swaption_vega(
                init_swap_rate, option_strike, swap_annuity,
                option_maturity, vol - shock)
            volga_diff = (value_plus - value_minus) / (2.0 * shock)

            volga_analytic = target.black_payers_swaption_volga(
                init_swap_rate,
                option_strike,
                swap_annuity,
                option_maturity,
                vol)
            assert volga_diff == approx(volga_analytic)

    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, swap_annuity, option_maturity, vol",
        [
            # vol < 0 raise AssertionError
            (1.0, 2.0, 1.0, 1.0, -0.1),
            # maturity < 0
            (1.0, 2.0, 1.0, -1.0, 0.1),
            # otherwise
            (1.1, 1.2, 1.3, 1.4, 0.1),
        ])
    def test_model_black_payers_swaption_vega_fprime_by_strike(
            self, init_swap_rate, option_strike, swap_annuity,
            option_maturity, vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_payers_swaption_vega_fprime_by_strike(
                    init_swap_rate,
                    option_strike,
                    swap_annuity,
                    option_maturity,
                    vol)
            return
        elif option_maturity < 0.0 or np.isclose(option_maturity, 0.0):
            expect = 0.0
            actual = target.black_payers_swaption_vega_fprime_by_strike(
                init_swap_rate,
                option_strike,
                swap_annuity,
                option_maturity,
                vol)
            assert expect == approx(actual)
        else:
            shock = 1e-6
            value_plus = target.black_payers_swaption_vega(
                init_swap_rate, option_strike + shock, swap_annuity,
                option_maturity, vol)
            value_minus = target.black_payers_swaption_vega(
                init_swap_rate, option_strike - shock, swap_annuity,
                option_maturity, vol)
            diff = (value_plus - value_minus) / (2.0 * shock)

            analytic = target.black_payers_swaption_vega_fprime_by_strike(
                init_swap_rate,
                option_strike,
                swap_annuity,
                option_maturity,
                vol)
            assert diff == approx(analytic)

    # -------------------------------------------------------------------------
    # black payer's/reciever's swaption distribution
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, swap_annuity, option_maturity, vol",
        [
            # vol < 0 raise AssertionError
            (1.1, 2.1, 1.2, 1.3, -0.1),
            # otherwise
            (1.1, 2.1, 1.2, 1.3, 0.1),
        ])
    def test_black_payers_swaption_cdf(self,
                                       init_swap_rate,
                                       option_strike,
                                       swap_annuity,
                                       option_maturity,
                                       vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_swaption_cdf(
                    init_swap_rate,
                    option_strike,
                    swap_annuity,
                    option_maturity,
                    vol)
            return
        else:
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            expect = (1.0
                      + function.black_payers_swaption_value_fprime_by_strike(
                          init_swap_rate,
                          option_strike,
                          swap_annuity,
                          option_maturity,
                          vol) / swap_annuity)

            actual = target.black_swaption_cdf(
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
            (1.1, 2.1, 1.2, 1.3, -0.1),
            # otherwise
            (1.1, 2.1, 1.2, 1.3, 0.1),
        ])
    def test_black_payers_swaption_pdf(self,
                                       init_swap_rate,
                                       option_strike,
                                       swap_annuity,
                                       option_maturity,
                                       vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.black_swaption_pdf(
                    init_swap_rate,
                    option_strike,
                    swap_annuity,
                    option_maturity,
                    vol)
            return
        else:
            # double checking implimentation of formula
            # because it is a bit complicated to generate test cases
            expect = (function.black_payers_swaption_value_fhess_by_strike(
                init_swap_rate,
                option_strike,
                swap_annuity,
                option_maturity,
                vol) / swap_annuity)

            actual = target.black_swaption_pdf(
                init_swap_rate,
                option_strike,
                swap_annuity,
                option_maturity,
                vol)
            assert expect == approx(actual)
