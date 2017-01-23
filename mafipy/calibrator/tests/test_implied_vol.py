#!/bin/python

from __future__ import division, print_function, absolute_import
from pytest import approx
import pytest

from . import util
import mafipy.calibrator.implied_vol as target
import mafipy.function as function


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
        (2.1, 1.85, util.get(), util.get(), util.get()),
    ])
    def test_black_scholes_implied_vol(
            self, underlying, strike, rate, maturity, vol):
        option_value = function.black_scholes_call_formula(
            underlying, strike, rate, maturity, vol)
        implied_vol = target.black_scholes_implied_vol(
            underlying, strike, rate, maturity, option_value)
        assert implied_vol == approx(vol, rel=5e-4)

