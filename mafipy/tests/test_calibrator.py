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
