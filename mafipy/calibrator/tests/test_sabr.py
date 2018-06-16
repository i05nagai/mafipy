#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from pytest import approx
import pytest

from . import util
import mafipy.calibrator as calibrator
import mafipy.calibrator.sabr as target
import mafipy.function


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
            (0.12, 0.22, util.get(), 6.0 / 12.0, util.get()),
            (0.08, 0.22, util.get(), 1.0, util.get()),
            (0.08, 0.22, util.get(), 2.0, util.get()),
            (0.08, 0.22, util.get(), 3.0, util.get()),
            (0.08, 0.22, util.get(), 4.0, util.get()),
            (0.08, 0.22, util.get(), 5.0, util.get()),
            (0.12, 0.22, util.get(), 6.0, util.get()),
            (0.08, 0.22, util.get(), 7.0, util.get()),
            (0.08, 0.22, util.get(), 8.0, util.get()),
            (0.08, 0.22, util.get(), 9.0, util.get()),
            (0.08, 0.22, util.get(), 10.0, util.get()),
            # in the money
            (0.21, 0.18, util.get(), 1.0 / 12.0, util.get()),
            (0.21, 0.10, util.get(), 3.0 / 12.0, util.get()),
            (0.21, 0.18, util.get(), 6.0 / 12.0, util.get()),
            (0.21, 0.12, util.get(), 1.0, util.get()),
            (0.21, 0.18, util.get(), 2.0, util.get()),
            (0.21, 0.18, util.get(), 3.0, util.get()),
            (0.21, 0.12, util.get(), 4.0, util.get()),
            (0.21, 0.08, util.get(), 5.0, util.get()),
            (0.21, 0.10, util.get(), 6.0, util.get()),
            (0.21, 0.08, util.get(), 7.0, util.get()),
            (0.21, 0.08, util.get(), 8.0, util.get()),
            (0.21, 0.08, util.get(), 9.0, util.get()),
            (0.21, 0.08, util.get(), 10.0, util.get()),
        ])
    def test_black_swaption_implied_vol(self,
                                        init_swap_rate,
                                        option_strike,
                                        swap_annuity,
                                        option_maturity,
                                        vol):
        option_value = mafipy.function.black_payers_swaption_value(
            init_swap_rate, option_strike, swap_annuity, option_maturity, vol)
        implied_vol = calibrator.black_swaption_implied_vol(init_swap_rate,
                                                            option_strike,
                                                            swap_annuity,
                                                            option_maturity,
                                                            option_value)
        assert implied_vol == approx(vol, rel=1e-6)

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
        assert 0.0116352114 == approx(alpha)
        assert expect_beta == approx(beta)
        assert 0.4495084250 == approx(rho)
        assert 0.7902295128 == approx(nu)

        # beta 0.5
        expect_beta = 0.5
        alpha, beta, rho, nu = target.sabr_caibration_simple(
            market_vols,
            market_strikes,
            option_maturity,
            expect_beta,
            init_alpha=1.0,
            init_rho=0.2,
            init_nu=1.0,
            nu_lower_bound=1e-8,
            tol=1e-32)
        # for travis CI tests
        assert 0.0729991374 == approx(alpha, rel=1e-4)
        assert expect_beta == approx(beta)
        assert -0.3051180437 == approx(rho, rel=5e-3)
        assert 2.5609685118e-05 == approx(nu, rel=5e-3)

        # beta 1.0
        expect_beta = 1.0
        alpha, beta, rho, nu = target.sabr_caibration_simple(market_vols,
                                                             market_strikes,
                                                             option_maturity,
                                                             expect_beta)
        assert 0.3242804455 == approx(alpha)
        assert expect_beta == approx(beta)
        assert -0.0416661971 == approx(rho)
        assert 0.7794312489 == approx(nu)

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
        assert 0.01120946664130403 == approx(alpha, rel=5e-4)
        assert expect_beta == approx(beta)
        # 0.42567236979991263 - 0.4256399449986689
        # 0.42567236979991263 * 3.2424801243713386e-05 = 1.3802341985702631e-05
        # https://travis-ci.org/i05nagai/mafipy/jobs/388131298
        assert 0.42567236979991263 == approx(rho, rel=5e-4)
        # 0.84491343183536094 - 0.8449269034163132
        # 0.84491343183536094 * -1.3471580952217899e-05 = -1.1382319694586305e-05
        # https://travis-ci.org/i05nagai/mafipy/jobs/392970990
        assert 0.84491343183536094 == approx(nu, rel=5e-4)

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
