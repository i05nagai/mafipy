#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from pytest import approx
import math
import pytest

import mafipy.function
import mafipy.function.sabr as target


class TestSabr(object):

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
    # SABR model
    # -------------------------------------------------------------------------
    # TODO: add more test cases
    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, swap_annuity, option_maturity,\
        alpha, beta, rho, nu", [
            (0.0357, 0.03, 1.0, 2.0, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_sabr_payers_swaption_value(
            self, init_swap_rate, option_strike, swap_annuity, option_maturity,
            alpha, beta, rho, nu):
        vol = target.sabr_implied_vol_hagan(
            init_swap_rate, option_strike, option_maturity,
            alpha, beta, rho, nu)
        expect = mafipy.function.black_payers_swaption_value(
            init_swap_rate, option_strike, swap_annuity, option_maturity, vol)

        # actual
        actual = target.sabr_payers_swaption_value(
            init_swap_rate, option_strike, swap_annuity, option_maturity,
            alpha, beta, rho, nu)
        assert expect == actual

    # TODO: add more test cases
    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, swap_annuity, option_maturity,\
        alpha, beta, rho, nu", [
            (0.0357, 0.03, 1.0, 2.0, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_sabr_receivers_swaption_value(
            self, init_swap_rate, option_strike, swap_annuity, option_maturity,
            alpha, beta, rho, nu):
        call_value = target.sabr_payers_swaption_value(
            init_swap_rate, option_strike, swap_annuity, option_maturity,
            alpha, beta, rho, nu)
        forward_value = swap_annuity * (init_swap_rate - option_strike)
        expect = call_value - forward_value

        # actual
        actual = target.sabr_receivers_swaption_value(
            init_swap_rate, option_strike, swap_annuity, option_maturity,
            alpha, beta, rho, nu)
        assert expect == actual

    # TODO: add more test cases
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_sabr_implied_vol_hagan(
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
        actual = target.sabr_implied_vol_hagan(
            underlying, strike, maturity, alpha, beta, rho, nu)
        assert expect == actual

    # TODO: add more test cases
    @pytest.mark.parametrize(
        "underlying, maturity, alpha, beta, rho, nu", [
            (0.0357, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_sabr_atm_implied_vol_hagan(
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
        actual = target.sabr_atm_implied_vol_hagan(
            underlying, maturity, alpha, beta, rho, nu)
        assert expect == actual

    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model_sabr_implied_vol_hagan(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        vol1 = target.sabr_implied_vol_hagan(
            underlying, strike, maturity, alpha, beta, rho, nu)

        A1 = target._sabr_implied_vol_hagan_A1(
            underlying, strike, maturity, alpha, beta, rho, nu)
        A2 = target._sabr_implied_vol_hagan_A2(
            underlying, strike, maturity, alpha, beta, rho, nu)
        A3 = target._sabr_implied_vol_hagan_A3(
            underlying, strike, maturity, alpha, beta, rho, nu)
        A4 = target._sabr_implied_vol_hagan_A4(
            underlying, strike, maturity, alpha, beta, rho, nu)
        vol2 = alpha * A2 * A4 / (A1 * A3)

        assert vol1 == approx(vol2)

    # A1 fprime by strike
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A1_fprime_by_strike(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-5

        A1_plus = target._sabr_implied_vol_hagan_A1(
            underlying, strike + shock, maturity, alpha, beta, rho, nu)
        A1_minus = target._sabr_implied_vol_hagan_A1(
            underlying, strike - shock, maturity, alpha, beta, rho, nu)
        A1_fprime = (A1_plus - A1_minus) / (2.0 * shock)

        A1_prime_analytic = target._sabr_implied_vol_hagan_A1_fprime_by_strike(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A1_prime_analytic == approx(A1_fprime)

    # A1 fhess by strike
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A1_fhess_by_strike(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-6

        A1_plus = target._sabr_implied_vol_hagan_A1_fprime_by_strike(
            underlying, strike + shock, maturity, alpha, beta, rho, nu)
        A1_minus = target._sabr_implied_vol_hagan_A1_fprime_by_strike(
            underlying, strike - shock, maturity, alpha, beta, rho, nu)
        A1_diff = (A1_plus - A1_minus) / (2.0 * shock)

        A1_analytic = target._sabr_implied_vol_hagan_A1_fhess_by_strike(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A1_analytic == approx(A1_diff, rel=5e-4)

    # A1 fprime by underlying
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A1_fprime_by_underlying(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-5

        A1_plus = target._sabr_implied_vol_hagan_A1(
            underlying + shock, strike, maturity, alpha, beta, rho, nu)
        A1_minus = target._sabr_implied_vol_hagan_A1(
            underlying - shock, strike, maturity, alpha, beta, rho, nu)
        A1_diff = (A1_plus - A1_minus) / (2.0 * shock)

        A1_analytic = target._sabr_implied_vol_hagan_A1_fprime_by_underlying(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A1_analytic == approx(A1_diff)

    # A1 fhess by underlying
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A1_fhess_by_underlying(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-6

        A1_plus = target._sabr_implied_vol_hagan_A1_fprime_by_underlying(
            underlying + shock, strike, maturity, alpha, beta, rho, nu)
        A1_minus = target._sabr_implied_vol_hagan_A1_fprime_by_underlying(
            underlying - shock, strike, maturity, alpha, beta, rho, nu)
        A1_diff = (A1_plus - A1_minus) / (2.0 * shock)

        A1_analytic = target._sabr_implied_vol_hagan_A1_fhess_by_underlying(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A1_analytic == approx(A1_diff, rel=5e-4)

    # A2 fprime by strike
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A2_fprime_by_strike(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-5

        A2_plus = target._sabr_implied_vol_hagan_A2(
            underlying, strike + shock, maturity, alpha, beta, rho, nu)
        A2_minus = target._sabr_implied_vol_hagan_A2(
            underlying, strike - shock, maturity, alpha, beta, rho, nu)
        A2_fprime = (A2_plus - A2_minus) / (2.0 * shock)

        A2_prime_analytic = target._sabr_implied_vol_hagan_A2_fprime_by_strike(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A2_prime_analytic == approx(A2_fprime)

    # A2 fhess by strike
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A2_fhess_by_strike(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-6

        A2_plus = target._sabr_implied_vol_hagan_A2_fprime_by_strike(
            underlying, strike + shock, maturity, alpha, beta, rho, nu)
        A2_minus = target._sabr_implied_vol_hagan_A2_fprime_by_strike(
            underlying, strike - shock, maturity, alpha, beta, rho, nu)
        A2_diff = (A2_plus - A2_minus) / (2.0 * shock)

        A2_analytic = target._sabr_implied_vol_hagan_A2_fhess_by_strike(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A2_analytic == approx(A2_diff)

    # A2 fprime by underlying
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A2_fprime_by_underlying(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-5

        A2_plus = target._sabr_implied_vol_hagan_A2(
            underlying + shock, strike, maturity, alpha, beta, rho, nu)
        A2_minus = target._sabr_implied_vol_hagan_A2(
            underlying - shock, strike, maturity, alpha, beta, rho, nu)
        A2_diff = (A2_plus - A2_minus) / (2.0 * shock)

        A2_analytic = target._sabr_implied_vol_hagan_A2_fprime_by_underlying(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A2_analytic == approx(A2_diff)

    # A2 fhess by underlying
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A2_fhess_by_underlying(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-6

        A2_plus = target._sabr_implied_vol_hagan_A2_fprime_by_underlying(
            underlying + shock, strike, maturity, alpha, beta, rho, nu)
        A2_minus = target._sabr_implied_vol_hagan_A2_fprime_by_underlying(
            underlying - shock, strike, maturity, alpha, beta, rho, nu)
        A2_diff = (A2_plus - A2_minus) / (2.0 * shock)

        A2_analytic = target._sabr_implied_vol_hagan_A2_fhess_by_underlying(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A2_analytic == approx(A2_diff)

    # A3 fprime by strike
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A3_fprime_by_strike(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-5

        A3_plus = target._sabr_implied_vol_hagan_A3(
            underlying, strike + shock, maturity, alpha, beta, rho, nu)
        A3_minus = target._sabr_implied_vol_hagan_A3(
            underlying, strike - shock, maturity, alpha, beta, rho, nu)
        A3_fprime = (A3_plus - A3_minus) / (2.0 * shock)

        A3_prime_analytic = target._sabr_implied_vol_hagan_A3_fprime_by_strike(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A3_prime_analytic == approx(A3_fprime)

    # A3 fhess by strike
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A3_fhess_by_strike(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-6

        A3_plus = target._sabr_implied_vol_hagan_A3_fprime_by_strike(
            underlying, strike + shock, maturity, alpha, beta, rho, nu)
        A3_minus = target._sabr_implied_vol_hagan_A3_fprime_by_strike(
            underlying, strike - shock, maturity, alpha, beta, rho, nu)
        A3_diff = (A3_plus - A3_minus) / (2.0 * shock)

        A3_analytic = target._sabr_implied_vol_hagan_A3_fhess_by_strike(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A3_analytic == approx(A3_diff)

    # A3 fprime by underlying
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A3_fprime_by_underlying(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-5

        A3_plus = target._sabr_implied_vol_hagan_A3(
            underlying + shock, strike, maturity, alpha, beta, rho, nu)
        A3_minus = target._sabr_implied_vol_hagan_A3(
            underlying - shock, strike, maturity, alpha, beta, rho, nu)
        A3_diff = (A3_plus - A3_minus) / (2.0 * shock)

        A3_analytic = target._sabr_implied_vol_hagan_A3_fprime_by_underlying(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A3_analytic == approx(A3_diff)

    # A3 fhess by underlying
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A3_fhess_by_underlying(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-6

        A3_plus = target._sabr_implied_vol_hagan_A3_fprime_by_underlying(
            underlying + shock, strike, maturity, alpha, beta, rho, nu)
        A3_minus = target._sabr_implied_vol_hagan_A3_fprime_by_underlying(
            underlying - shock, strike, maturity, alpha, beta, rho, nu)
        A3_diff = (A3_plus - A3_minus) / (2.0 * shock)

        A3_analytic = target._sabr_implied_vol_hagan_A3_fhess_by_underlying(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A3_analytic == approx(A3_diff)

    # A4 fprime by underlying
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A4_fprime_by_underlying(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-6

        A4_plus = target._sabr_implied_vol_hagan_A4(
            underlying + shock, strike, maturity, alpha, beta, rho, nu)
        A4_minus = target._sabr_implied_vol_hagan_A4(
            underlying - shock, strike, maturity, alpha, beta, rho, nu)
        A4_diff = (A4_plus - A4_minus) / (2.0 * shock)

        A4_analytic = target._sabr_implied_vol_hagan_A4_fprime_by_underlying(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A4_analytic == approx(A4_diff)

    # A4 fhess by strike
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A4_fhess_by_strike(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-6

        A4_plus = target._sabr_implied_vol_hagan_A4_fprime_by_strike(
            underlying, strike + shock, maturity, alpha, beta, rho, nu)
        A4_minus = target._sabr_implied_vol_hagan_A4_fprime_by_strike(
            underlying, strike - shock, maturity, alpha, beta, rho, nu)
        A4_diff = (A4_plus - A4_minus) / (2.0 * shock)

        A4_analytic = target._sabr_implied_vol_hagan_A4_fhess_by_strike(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A4_analytic == approx(A4_diff)

    # A4 fhess by underlying
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model__sabr_implied_vol_hagan_A4_fhess_by_underlying(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-6

        A4_plus = target._sabr_implied_vol_hagan_A4_fprime_by_underlying(
            underlying + shock, strike, maturity, alpha, beta, rho, nu)
        A4_minus = target._sabr_implied_vol_hagan_A4_fprime_by_underlying(
            underlying - shock, strike, maturity, alpha, beta, rho, nu)
        A4_diff = (A4_plus - A4_minus) / (2.0 * shock)

        A4_analytic = target._sabr_implied_vol_hagan_A4_fhess_by_underlying(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert A4_analytic == approx(A4_diff)

    # implied vol fprime
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model_sabr_implied_vol_hagan_fprime_by_strike(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-5
        vol1 = target.sabr_implied_vol_hagan(
            underlying, strike + shock, maturity, alpha, beta, rho, nu)
        vol2 = target.sabr_implied_vol_hagan(
            underlying, strike - shock, maturity, alpha, beta, rho, nu)
        vol_diff = (vol1 - vol2) / (2.0 * shock)

        vol_analytic = target.sabr_implied_vol_hagan_fprime_by_strike(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert vol_analytic == approx(vol_diff)

    # implied vol fhess
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model_sabr_implied_vol_hagan_fhess_by_strike(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-5
        vol_fprime1 = target.sabr_implied_vol_hagan_fprime_by_strike(
            underlying, strike + shock, maturity, alpha, beta, rho, nu)
        vol_fprime2 = target.sabr_implied_vol_hagan_fprime_by_strike(
            underlying, strike - shock, maturity, alpha, beta, rho, nu)
        vol_diff = (vol_fprime1 - vol_fprime2) / (2.0 * shock)

        vol_analytic = target.sabr_implied_vol_hagan_fhess_by_strike(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert vol_analytic == approx(vol_diff, rel=5e-5)

    # implied vol fprime by underlying
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model_sabr_implied_vol_hagan_fprime_by_underlying(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-5
        vol1 = target.sabr_implied_vol_hagan(
            underlying + shock, strike, maturity, alpha, beta, rho, nu)
        vol2 = target.sabr_implied_vol_hagan(
            underlying - shock, strike, maturity, alpha, beta, rho, nu)
        vol_diff = (vol1 - vol2) / (2.0 * shock)

        vol_analytic = target.sabr_implied_vol_hagan_fprime_by_underlying(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert vol_analytic == approx(vol_diff)

    # implied vol fhess by underlying
    @pytest.mark.parametrize(
        "underlying, strike, maturity, alpha, beta, rho, nu", [
            (0.0357, 0.03, 2, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model_sabr_implied_vol_hagan_fhess_by_underlying(
            self, underlying, strike, maturity, alpha, beta, rho, nu):
        shock = 1e-5
        vol1 = target.sabr_implied_vol_hagan_fprime_by_underlying(
            underlying + shock, strike, maturity, alpha, beta, rho, nu)
        vol2 = target.sabr_implied_vol_hagan_fprime_by_underlying(
            underlying - shock, strike, maturity, alpha, beta, rho, nu)
        vol_diff = (vol1 - vol2) / (2.0 * shock)

        vol_analytic = target.sabr_implied_vol_hagan_fhess_by_underlying(
            underlying, strike, maturity, alpha, beta, rho, nu)

        assert vol_analytic == approx(vol_diff, rel=5e-4)

    # -------------------------------------------------------------------------
    # SABR greeks
    # -------------------------------------------------------------------------
    # SABR payer's swaption delta
    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, swap_annuity, option_maturity,\
        alpha, beta, rho, nu", [
            (0.0357, 0.03, 1.0, 2.0, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model_sabr_payers_swaption_delta(
            self, init_swap_rate, option_strike, swap_annuity, option_maturity,
            alpha, beta, rho, nu):
        shock = 1e-6
        value_plus = target.sabr_payers_swaption_value(
            init_swap_rate + shock, option_strike, swap_annuity,
            option_maturity, alpha, beta, rho, nu)
        value_minus = target.sabr_payers_swaption_value(
            init_swap_rate - shock, option_strike, swap_annuity,
            option_maturity, alpha, beta, rho, nu)
        delta_diff = (value_plus - value_minus) / (2.0 * shock)

        delta_analytic = target.sabr_payers_swaption_delta(
            init_swap_rate, option_strike, swap_annuity, option_maturity,
            alpha, beta, rho, nu)

        assert delta_analytic == approx(delta_diff, rel=5e-3)

    # -------------------------------------------------------------------------
    # SABR distribution
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, option_maturity,\
        alpha, beta, rho, nu", [
            (0.0357, 0.03, 2.0, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model_sabr_cdf(
            self, init_swap_rate, option_strike, option_maturity,
            alpha, beta, rho, nu):
        shock = 1e-6
        value_plus = target.sabr_payers_swaption_value(
            init_swap_rate, option_strike + shock, 1.0,
            option_maturity, alpha, beta, rho, nu)
        value_minus = target.sabr_payers_swaption_value(
            init_swap_rate, option_strike - shock, 1.0,
            option_maturity, alpha, beta, rho, nu)
        value_diff = (value_plus - value_minus) / (2.0 * shock)
        cdf_diff = 1.0 + value_diff

        cdf_analytic = target.sabr_cdf(
            init_swap_rate, option_strike, option_maturity,
            alpha, beta, rho, nu)

        assert cdf_analytic == approx(cdf_diff)

    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, option_maturity,\
        alpha, beta, rho, nu", [
            (0.0357, 0.03, 2.0, 0.036, 0.5, -0.25, 0.35)
        ])
    def test_model_sabr_pdf(
            self, init_swap_rate, option_strike, option_maturity,
            alpha, beta, rho, nu):
        shock = 1e-6
        value_plus = target.sabr_cdf(
            init_swap_rate, option_strike + shock, option_maturity,
            alpha, beta, rho, nu)
        value_minus = target.sabr_cdf(
            init_swap_rate, option_strike - shock, option_maturity,
            alpha, beta, rho, nu)
        value_diff = (value_plus - value_minus) / (2.0 * shock)
        pdf_diff = value_diff

        pdf_analytic = target.sabr_pdf(
            init_swap_rate, option_strike, option_maturity,
            alpha, beta, rho, nu)

        assert pdf_analytic == approx(pdf_diff, rel=5e-5)
