#!/bin/python

from __future__ import division
from pytest import approx
import pytest
import mafipy.calibrator as target
import mafipy.analytic_formula as analytic_formula
import mafipy.tests.util as util


class TestModelCalibrator:

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

    @pytest.mark.parametrize("underlying, strike, rate, maturity, vol", [
        # at the money
        (2.1, 2.1, util.get(), util.get(), util.get()),
        # out of the money
        (1.1, 2.2, util.get(), util.get(), util.get()),
        # in the money
        (2.1, 1.9, util.get(), util.get(), util.get()),
    ])
    def test_black_scholes_implied_vol(
            self, underlying, strike, rate, maturity, vol):
        option_value = analytic_formula.black_scholes_call_formula(
            underlying, strike, rate, maturity, vol)
        implied_vol = target.black_scholes_implied_vol(
            underlying, strike, rate, maturity, option_value)
        assert implied_vol == approx(vol)

    @pytest.mark.parametrize(
        "init_swap_rate, option_strike, swap_annuity, option_maturity, vol", [
            # at the money
            (0.21, 0.21, util.get(), 1.0 / 12.0, util.get()),
            (0.21, 0.21, util.get(), 3.0 / 12.0, util.get()),
            (0.21, 0.21, util.get(), 6.0 / 12.0, util.get()),
            (0.21, 0.21, util.get(), 1.0, util.get()),
            (0.21, 0.21, util.get(), 2.0, util.get()),
            (0.21, 0.21, util.get(), 3.0, util.get()),
            (0.21, 0.21, util.get(), 4.0, util.get()),
            (0.21, 0.21, util.get(), 5.0, util.get()),
            (0.21, 0.21, util.get(), 6.0, util.get()),
            (0.21, 0.21, util.get(), 7.0, util.get()),
            (0.21, 0.21, util.get(), 8.0, util.get()),
            (0.21, 0.21, util.get(), 9.0, util.get()),
            (0.21, 0.21, util.get(), 10.0, util.get()),
            # out of the money
            (0.18, 0.22, util.get(), 1.0 / 12.0, util.get()),
            (0.08, 0.22, util.get(), 3.0 / 12.0, util.get()),
            (0.08, 0.22, util.get(), 6.0 / 12.0, util.get()),
            (0.08, 0.22, util.get(), 1.0, util.get()),
            (0.08, 0.22, util.get(), 2.0, util.get()),
            (0.08, 0.22, util.get(), 3.0, util.get()),
            (0.08, 0.22, util.get(), 4.0, util.get()),
            (0.08, 0.22, util.get(), 5.0, util.get()),
            (0.08, 0.22, util.get(), 6.0, util.get()),
            (0.08, 0.22, util.get(), 7.0, util.get()),
            (0.08, 0.22, util.get(), 8.0, util.get()),
            (0.08, 0.22, util.get(), 9.0, util.get()),
            (0.08, 0.22, util.get(), 10.0, util.get()),
            # in the money
            (0.21, 0.18, util.get(), 1.0 / 12.0, util.get()),
            (0.21, 0.08, util.get(), 3.0 / 12.0, util.get()),
            (0.21, 0.18, util.get(), 6.0 / 12.0, util.get()),
            (0.21, 0.08, util.get(), 1.0, util.get()),
            (0.21, 0.08, util.get(), 2.0, util.get()),
            (0.21, 0.08, util.get(), 3.0, util.get()),
            (0.21, 0.08, util.get(), 4.0, util.get()),
            (0.21, 0.08, util.get(), 5.0, util.get()),
            (0.21, 0.08, util.get(), 6.0, util.get()),
            (0.21, 0.08, util.get(), 7.0, util.get()),
            (0.21, 0.08, util.get(), 8.0, util.get()),
            (0.21, 0.08, util.get(), 9.0, util.get()),
            (0.21, 0.08, util.get(), 10., util.get()),
        ])
    def test_black_swaption_implied_vol(self,
                                        init_swap_rate,
                                        option_strike,
                                        swap_annuity,
                                        option_maturity,
                                        vol):
        option_value = analytic_formula.black_payers_swaption_value(
            init_swap_rate, option_strike, swap_annuity, option_maturity, vol)
        implied_vol = target.black_swaption_implied_vol(init_swap_rate,
                                                        option_strike,
                                                        swap_annuity,
                                                        option_maturity,
                                                        option_value)
        assert implied_vol == approx(vol)

    def test_sabr_calibration_simple(self):
        market_vols = [45.6, 41.6, 37.9, 36.6, 37.8, 39.2, 40.0]
        market_vols = [vol / 100.0 for vol in market_vols]
        market_strikes = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        market_strikes = [strike / 100.0 for strike in market_strikes]
        option_maturity = 3.0

        # beta 0
        expect_beta = 0.0
        alpha, beta, rho, nu = target.sabr_caibration_simple(market_vols,
                                                             market_strikes,
                                                             option_maturity,
                                                             expect_beta,
                                                             init_alpha=1.0,
                                                             init_nu=0.4)
        assert 0.0135724377 == approx(alpha)
        assert expect_beta == approx(beta)
        assert 0.8879962646 == approx(rho)
        assert 0.3376939549 == approx(nu)

        # beta 0.5
        expect_beta = 0.5
        alpha, beta, rho, nu = target.sabr_caibration_simple(market_vols,
                                                             market_strikes,
                                                             option_maturity,
                                                             expect_beta,
                                                             init_alpha=1.0,
                                                             init_rho=0.4,
                                                             init_nu=1.0)
        assert 0.072998468816 == approx(alpha)
        assert expect_beta == approx(beta)
        assert -0.002287560228 == approx(rho)
        assert 1e-08 == approx(nu)

        # beta 1.0
        expect_beta = 1.0
        alpha, beta, rho, nu = target.sabr_caibration_simple(market_vols,
                                                             market_strikes,
                                                             option_maturity,
                                                             expect_beta)
        assert 0.3242845824 == approx(alpha)
        assert expect_beta == approx(beta)
        assert -0.041674869 == approx(rho)
        assert 0.779416715 == approx(nu)

    def test_sabr_calibration_west(self):
        market_vols = [45.6, 41.6, 37.9, 36.6, 37.8, 39.2, 40.0]
        market_vols = [vol / 100.0 for vol in market_vols]
        market_strikes = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
        market_strikes = [strike / 100.0 for strike in market_strikes]
        option_maturity = 3.0

        # beta 0
        expect_beta = 0.0
        alpha, beta, rho, nu = target.sabr_caibration_west(market_vols,
                                                           market_strikes,
                                                           option_maturity,
                                                           expect_beta)
        assert 0.01120935448488606 == approx(alpha)
        assert expect_beta == approx(beta)
        assert 0.42563951834424474 == approx(rho)
        assert 0.84492709640795993 == approx(nu)

        # beta 0.5
        expect_beta = 0.5
        alpha, beta, rho, nu = target.sabr_caibration_west(market_vols,
                                                           market_strikes,
                                                           option_maturity,
                                                           expect_beta)
        assert 0.05847588694407545 == approx(alpha)
        assert expect_beta == approx(beta)
        assert 0.20565391641295228 == approx(rho)
        assert 0.79689209024466956 == approx(nu)

        # beta 1.0
        expect_beta = 1.0
        alpha, beta, rho, nu = target.sabr_caibration_west(market_vols,
                                                           market_strikes,
                                                           option_maturity,
                                                           expect_beta,
                                                           init_rho=0.3,
                                                           init_nu=0.8)
        assert 0.31548238677601786 == approx(alpha)
        assert expect_beta == approx(beta)
        assert -0.030890193169276017 == approx(rho)
        assert 0.81566639230647286 == approx(nu)
