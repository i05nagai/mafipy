#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from pytest import approx
import math
import numpy as np
import pytest
import scipy.stats

from . import util
import mafipy.function as function
import mafipy.replication.replication_method as replication_method
import mafipy.replication._quanto_cms_forward_fx as _fx
import mafipy.replication.pricer_quanto_cms as target


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

    # -------------------------------------------------------------------------
    # Black swaption model
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "init_swap_rate, strike, swap_annuity, option_maturity, vol", [
            # vol < 0 raise AssertionError
            (2.0, 1.0, 3.0, 1.0, -1.0),
            # otherwise
            (2.0, 1.0, 3.0, 1.0, 1.0),
        ])
    def test_make_pdf_black_swaption(self,
                                     init_swap_rate,
                                     strike,
                                     swap_annuity,
                                     option_maturity,
                                     vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.make_pdf_black_swaption(
                    init_swap_rate, swap_annuity, option_maturity, vol)
        else:
            expect = function.black_payers_swaption_value_fhess_by_strike(
                init_swap_rate, strike, 1.0, option_maturity, vol)
            actual = target.make_pdf_black_swaption(
                init_swap_rate, swap_annuity, option_maturity, vol)(strike)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "init_swap_rate, strike, swap_annuity, option_maturity, vol", [
            # vol < 0 raise AssertionError
            (2.0, 1.0, 3.0, 1.0, -1.0),
            # otherwise
            (2.0, 1.0, 3.0, 1.0, 1.0),
        ])
    def test_make_pdf_fprime_black_swaption(self,
                                            init_swap_rate,
                                            strike,
                                            swap_annuity,
                                            option_maturity,
                                            vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.make_pdf_fprime_black_swaption(
                    init_swap_rate, swap_annuity, option_maturity, vol)
        else:
            expect = function.black_payers_swaption_value_third_by_strike(
                init_swap_rate, strike, 1.0, option_maturity, vol)
            actual = target.make_pdf_fprime_black_swaption(
                init_swap_rate, swap_annuity, option_maturity, vol)(strike)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "init_swap_rate, strike, swap_annuity, option_maturity, vol", [
            # vol < 0 raise AssertionError
            (2.0, 1.0, 3.0, 1.0, -1.0),
            # otherwise
            (2.0, 1.0, 3.0, 1.0, 1.0),
        ])
    def test_make_cdf_black_swaption(self,
                                     init_swap_rate,
                                     strike,
                                     swap_annuity,
                                     option_maturity,
                                     vol):
        # raise AssertionError
        if vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.make_cdf_black_swaption(
                    init_swap_rate, swap_annuity, option_maturity, vol)
        else:
            expect = (1.0
                      + function.black_payers_swaption_value_fprime_by_strike(
                          init_swap_rate, strike, 1.0, option_maturity, vol))
            actual = target.make_cdf_black_swaption(
                init_swap_rate, swap_annuity, option_maturity, vol)(strike)
            assert expect == approx(actual)

    # -------------------------------------------------------------------------
    # Black scholes model
    # -------------------------------------------------------------------------
    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, expect", [
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
            expect = function.black_scholes_call_value_fhess_by_strike(
                underlying, strike, rate, maturity, vol)
            actual = target.make_pdf_black_scholes(
                underlying, rate, maturity, vol)(strike)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, expect", [
            # maturity < 0 raise AssertionError
            (2.0, 1.0, 1.0, -1.0, 1.0, 1.0),
            # vol < 0 raise AssertionError
            (2.0, 1.0, 1.0, 1.0, -1.0, 1.0),
            # otherwise
            (2.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        ])
    def test_make_pdf_fprime_black_scholes(
            self, underlying, strike, rate, maturity, vol, expect):
        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.make_pdf_fprime_black_scholes(
                    underlying, rate, maturity, vol)
        else:
            expect = function.black_scholes_call_value_third_by_strike(
                underlying, strike, rate, maturity, vol)
            actual = target.make_pdf_fprime_black_scholes(
                underlying, rate, maturity, vol)(strike)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "underlying, strike, rate, maturity, vol, expect", [
            # maturity < 0 raise AssertionError
            (2.0, 1.0, 1.0, -1.0, 1.0, 1.0),
            # vol < 0 raise AssertionError
            (2.0, 1.0, 1.0, 1.0, -1.0, 1.0),
            # otherwise
            (2.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        ])
    def test_make_cdf_black_scholes(
            self, underlying, strike, rate, maturity, vol, expect):
        # raise AssertionError
        if maturity < 0.0 or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target.make_cdf_black_scholes(
                    underlying, rate, maturity, vol)
        else:
            expect = (1.0 + function.black_scholes_call_value_fprime_by_strike(
                underlying, strike, rate, maturity, vol))
            actual = target.make_cdf_black_scholes(
                underlying, rate, maturity, vol)(strike)
            assert expect == approx(actual)


class Test_SimpleQuantoCmsLinearCallHelper(object):

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
        # data
        data = util.get_real(13)
        self.alpha0 = data[0]
        self.alpha1 = data[1]
        self.payoff_strike = data[3]
        self.payoff_gearing = data[4]
        self.maturity = data[5]
        self.forward_fx_vol = data[6]
        self.forward_fx_corr = data[7]
        self.init_swap_rate = data[8]
        self.rate = data[9]
        self.swap_rate_vol = data[10]
        self.today = 0.0
        self.init_swap_rate = data[11]
        self.rate = data[12]
        # ordered data
        data = sorted(util.get_real(3))
        self.init_swap_rate = data[1]
        self.min_put_range = data[0]
        self.max_put_range = data[1]
        self.min_call_range = data[1]
        self.max_call_range = data[2]
        # set
        self._set_target()

    # after each test finish
    def teardown(self):
        pass

    def _set_target(self):
        # annuity mapping params
        annuity_mapping_params = {
            "alpha0": self.alpha0,
            "alpha1": self.alpha1
        }
        annuity_mapping_helper = (
            replication_method.LinearAnnuityMappingFuncHelper(
                **annuity_mapping_params))
        self.annuity_mapping_func = annuity_mapping_helper.make_func()
        self.annuity_mapping_fprime = annuity_mapping_helper.make_fprime()
        self.annuity_mapping_fhess = annuity_mapping_helper.make_fhess()
        # payoff helper
        self.payoff_params = {
            "strike": self.payoff_strike,
            "gearing": self.payoff_gearing,
        }
        call_payoff_helper = function.CallUnderlyingPayoffHelper(
            **self.payoff_params)
        self.payoff_func = call_payoff_helper.make_func()
        self.payoff_fprime = call_payoff_helper.make_fprime()
        # foward fx diffusion
        forward_fx_diffusion_params = {
            "time": self.maturity,
            "vol": self.forward_fx_vol,
            "corr": self.forward_fx_corr,
            "swap_rate_cdf": self._swap_rate_cdf,
            "swap_rate_pdf": self._swap_rate_pdf,
            "swap_rate_pdf_fprime": self._swap_rate_pdf_fprime,
            "is_inverse": False
        }
        forward_fx_diffusion_helper = _fx._ForwardFxDiffusionHelper(
            **forward_fx_diffusion_params)
        self.forward_fx_diffusion = forward_fx_diffusion_helper.make_func()
        self.forward_fx_diffusion_fprime = (
            forward_fx_diffusion_helper.make_fprime())
        self.forward_fx_diffusion_fhess = (
            forward_fx_diffusion_helper.make_fhess())
        # pricer
        bs_pricer = function.BlackScholesPricerHelper()
        call_pricer_params = {
            "underlying": self.init_swap_rate,
            "rate": self.rate,
            "maturity": self.maturity,
            "vol": self.swap_rate_vol,
            "today": self.today,
        }
        self.call_pricer = bs_pricer.make_call_wrt_strike(**call_pricer_params)
        put_pricer_params = {
            "underlying": self.init_swap_rate,
            "rate": self.rate,
            "maturity": self.maturity,
            "vol": self.swap_rate_vol,
            "today": 0.0,
        }
        self.put_pricer = bs_pricer.make_put_wrt_strike(**put_pricer_params)
        # target
        self.target = target._SimpleQuantoCmsLinearCallHelper(
            annuity_mapping_helper,
            call_payoff_helper,
            forward_fx_diffusion_helper,
            self.call_pricer,
            self.put_pricer,
            self.payoff_strike,
            self.payoff_gearing,
            self.min_put_range,
            self.max_put_range,
            self.min_call_range,
            self.max_call_range)

    def _swap_rate_cdf(self, swap_rate):
        return scipy.stats.norm.cdf(swap_rate)

    def _swap_rate_pdf(self, swap_rate):
        return scipy.stats.norm.pdf(swap_rate)

    def _swap_rate_pdf_fprime(self, swap_rate):
        return function.norm_pdf_fprime(swap_rate)

    def _calc_func1(self, swap_rate):
        return (self.payoff_func(swap_rate)
                * self.annuity_mapping_func(swap_rate)
                * self.forward_fx_diffusion_fhess(swap_rate))

    def _calc_func2(self, swap_rate):
        return (2.0
                * self.payoff_fprime(swap_rate)
                * self.annuity_mapping_fprime(swap_rate)
                * self.forward_fx_diffusion(swap_rate))

    def _calc_func3(self, swap_rate):
        return (2.0
                * self.payoff_fprime(swap_rate)
                * self.annuity_mapping_func(swap_rate)
                * self.forward_fx_diffusion_fprime(swap_rate))

    def _calc_func4(self, swap_rate):
        return (2.0
                * self.payoff_func(swap_rate)
                * self.annuity_mapping_fprime(swap_rate)
                * self.forward_fx_diffusion_fprime(swap_rate))

    def test__make_numerator_call_integrands(self):
        data = util.get_real(1)
        swap_rate = data[0]
        funcs = self.target._make_numerator_integrands()
        # size
        assert 4 == len(funcs)
        # func1
        actual = funcs[0](swap_rate)
        expect = self._calc_func1(swap_rate)
        assert expect == approx(actual)
        # func2
        actual = funcs[1](swap_rate)
        expect = self._calc_func2(swap_rate)
        assert expect == approx(actual)
        # func3
        actual = funcs[2](swap_rate)
        expect = self._calc_func3(swap_rate)
        assert expect == approx(actual)
        # func4
        actual = funcs[3](swap_rate)
        expect = self._calc_func4(swap_rate)
        assert expect == approx(actual)

    def test_make_numerator_call_integrands(self):
        # just call _make_numerator_call_integrands()
        # so that no needs to test
        pass

    def test_make_numerator_put_integrands(self):
        # just call _make_numerator_call_integrands()
        # so that no needs to test
        pass

    def test_make_numerator_analytic_funcs(self):
        data = util.get_real(1)
        init_swap_rate = data[0]
        self.min_put_range = 1e-10
        self.max_put_range = init_swap_rate
        self.min_call_range = init_swap_rate
        self.max_call_range = init_swap_rate * 2.0

        def case1():
            # min_put_range < strike < max_put_range = min_call_range
            # min_call_range < max_put_range
            self.payoff_strike = (self.min_put_range
                                  + self.max_put_range) * 0.5
            self._set_target()
            funcs = self.target.make_numerator_analytic_funcs()

            # size
            assert 2 == len(funcs)
            # func1
            actual = funcs[0](init_swap_rate)
            expect = (self.payoff_func(init_swap_rate)
                      * self.annuity_mapping_func(init_swap_rate)
                      * self.forward_fx_diffusion(init_swap_rate))
            assert expect == approx(actual)
            # put term
            # p * g'' * a * chi
            actual = funcs[1](init_swap_rate)
            expect = (self.put_pricer(self.payoff_strike)
                      * self.payoff_gearing
                      * self.annuity_mapping_func(self.payoff_strike)
                      * self.forward_fx_diffusion(self.payoff_strike))
            assert expect == approx(actual)
        case1()

        # min_put_range < max_put_range = min_call_range
        # min_call_range < strike < max_call_range
        def case2():
            self.payoff_strike = (self.min_call_range
                                  + self.max_call_range) * 0.5
            self._set_target()
            funcs = self.target.make_numerator_analytic_funcs()
            # size
            assert 2 == len(funcs)
            # func1
            actual = funcs[0](init_swap_rate)
            expect = (self.payoff_func(init_swap_rate)
                      * self.annuity_mapping_func(init_swap_rate)
                      * self.forward_fx_diffusion(init_swap_rate))
            assert expect == approx(actual)
            # call term
            # c * g'' * a * chi
            actual = funcs[1](init_swap_rate)
            expect = (self.call_pricer(self.payoff_strike)
                      * self.payoff_gearing
                      * self.annuity_mapping_func(self.payoff_strike)
                      * self.forward_fx_diffusion(self.payoff_strike))
            assert expect == approx(actual)
        case2()


class Test_SimpleQuantoCmsLinearBullSpreadHelper(object):

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
        # data
        data = util.get_real(7)
        self.alpha0 = data[0]
        self.alpha1 = data[1]
        self.payoff_gearing = data[2]
        self.maturity = data[3]
        self.forward_fx_vol = data[4]
        self.forward_fx_corr = data[5]
        self.swap_rate_vol = data[6]
        # ordered data
        data = sorted(util.get_real(2))
        self.payoff_lower_strike = data[0]
        self.payoff_upper_strike = data[1]
        data = sorted(util.get_real(3))
        self.init_swap_rate = data[1]
        self.min_put_range = data[0]
        self.max_put_range = data[1]
        self.min_call_range = data[1]
        self.max_call_range = data[2]
        # set
        self._set_target()

    # after each test finish
    def teardown(self):
        pass

    def _set_target(self):
        # annuity mapping params
        self.annuity_mapping_params = {
            "alpha0": self.alpha0,
            "alpha1": self.alpha1
        }
        self.annuity_mapping_helper = (
            replication_method.LinearAnnuityMappingFuncHelper(
                **self.annuity_mapping_params))
        self.annuity_mapping_func = self.annuity_mapping_helper.make_func()
        self.annuity_mapping_fprime = self.annuity_mapping_helper.make_fprime()
        self.annuity_mapping_fhess = self.annuity_mapping_helper.make_fhess()
        # payoff helper
        self.payoff_params = {
            "lower_strike": self.payoff_lower_strike,
            "upper_strike": self.payoff_upper_strike,
            "gearing": self.payoff_gearing,
        }
        call_payoff_helper = function.BullSpreadUnderlyingPayoffHelper(
            **self.payoff_params)
        self.payoff_func = call_payoff_helper.make_func()
        self.payoff_fprime = call_payoff_helper.make_fprime()
        # foward fx diffusion
        self.forward_fx_diffusion_params = {
            "time": self.maturity,
            "vol": self.forward_fx_vol,
            "corr": self.forward_fx_corr,
            "swap_rate_cdf": self._swap_rate_cdf,
            "swap_rate_pdf": self._swap_rate_pdf,
            "swap_rate_pdf_fprime": self._swap_rate_pdf_fprime,
            "is_inverse": False
        }
        self.forward_fx_diffusion_helper = _fx._ForwardFxDiffusionHelper(
            **self.forward_fx_diffusion_params)
        self.forward_fx_diffusion = (
            self.forward_fx_diffusion_helper.make_func())
        self.forward_fx_diffusion_fprime = (
            self.forward_fx_diffusion_helper.make_fprime())
        self.forward_fx_diffusion_fhess = (
            self.forward_fx_diffusion_helper.make_fhess())
        # pricer
        bs_pricer = function.BlackSwaptionPricerHelper()
        call_pricer_params = {
            "init_swap_rate": self.init_swap_rate,
            "swap_annuity": 1.0,
            "option_maturity": self.maturity,
            "vol": self.swap_rate_vol,
        }
        self.call_pricer = bs_pricer.make_payers_swaption_wrt_strike(
            **call_pricer_params)
        put_pricer_params = {
            "init_swap_rate": self.init_swap_rate,
            "swap_annuity": 1.0,
            "option_maturity": self.maturity,
            "vol": self.swap_rate_vol,
        }
        self.put_pricer = bs_pricer.make_receivers_swaption_wrt_strike(
            **put_pricer_params)
        # target
        self.target = target._SimpleQuantoCmsLinearBullSpreadHelper(
            self.annuity_mapping_helper,
            call_payoff_helper,
            self.forward_fx_diffusion_helper,
            self.call_pricer,
            self.put_pricer,
            self.payoff_lower_strike,
            self.payoff_upper_strike,
            self.payoff_gearing,
            self.min_put_range,
            self.max_put_range,
            self.min_call_range,
            self.max_call_range)

    def _swap_rate_cdf(self, swap_rate):
        return scipy.stats.norm.cdf(swap_rate)

    def _swap_rate_pdf(self, swap_rate):
        return scipy.stats.norm.pdf(swap_rate)

    def _swap_rate_pdf_fprime(self, swap_rate):
        return function.norm_pdf_fprime(swap_rate)

    def _calc_func1(self, swap_rate):
        return (self.payoff_func(swap_rate)
                * self.annuity_mapping_func(swap_rate)
                * self.forward_fx_diffusion_fhess(swap_rate))

    def _calc_func2(self, swap_rate):
        return (2.0
                * self.payoff_fprime(swap_rate)
                * self.annuity_mapping_fprime(swap_rate)
                * self.forward_fx_diffusion(swap_rate))

    def _calc_func3(self, swap_rate):
        return (2.0
                * self.payoff_fprime(swap_rate)
                * self.annuity_mapping_func(swap_rate)
                * self.forward_fx_diffusion_fprime(swap_rate))

    def _calc_func4(self, swap_rate):
        return (2.0
                * self.payoff_func(swap_rate)
                * self.annuity_mapping_fprime(swap_rate)
                * self.forward_fx_diffusion_fprime(swap_rate))

    def test__make_numerator_call_integrands(self):
        data = util.get_real(1)
        swap_rate = data[0]
        funcs = self.target._make_numerator_integrands()
        # size
        assert 4 == len(funcs)
        # func1
        actual = funcs[0](swap_rate)
        expect = self._calc_func1(swap_rate)
        assert expect == approx(actual)
        # func2
        actual = funcs[1](swap_rate)
        expect = self._calc_func2(swap_rate)
        assert expect == approx(actual)
        # func3
        actual = funcs[2](swap_rate)
        expect = self._calc_func3(swap_rate)
        assert expect == approx(actual)
        # func4
        actual = funcs[3](swap_rate)
        expect = self._calc_func4(swap_rate)
        assert expect == approx(actual)

    def test_make_numerator_call_integrands(self):
        # just call _make_numerator_call_integrands()
        # so that no needs to test
        pass

    def test_make_numerator_put_integrands(self):
        # just call _make_numerator_call_integrands()
        # so that no needs to test
        pass

    def _analytic_func1(self, init_swap_rate):
        return (self.payoff_func(init_swap_rate)
                * self.annuity_mapping_func(init_swap_rate)
                * self.forward_fx_diffusion(init_swap_rate))

    # put term1
    def _analytic_func21(self, init_swap_rate):
        return (self.put_pricer(self.payoff_lower_strike)
                * self.payoff_gearing
                * self.annuity_mapping_func(self.payoff_lower_strike)
                * self.forward_fx_diffusion(self.payoff_lower_strike))

    # put term2
    def _analytic_func22(self, init_swap_rate):
        return -(self.put_pricer(self.payoff_upper_strike)
                 * self.payoff_gearing
                 * self.annuity_mapping_func(self.payoff_upper_strike)
                 * self.forward_fx_diffusion(self.payoff_upper_strike))

    # call term1
    def _analytic_func31(self, init_swap_rate):
        return (self.call_pricer(self.payoff_lower_strike)
                * self.payoff_gearing
                * self.annuity_mapping_func(self.payoff_lower_strike)
                * self.forward_fx_diffusion(self.payoff_lower_strike))

    # call term2
    def _analytic_func32(self, init_swap_rate):
        return -(self.call_pricer(self.payoff_upper_strike)
                 * self.payoff_gearing
                 * self.annuity_mapping_func(self.payoff_upper_strike)
                 * self.forward_fx_diffusion(self.payoff_upper_strike))

    def test_make_numerator_analytic_funcs(self):
        data = util.get_real(1)
        init_swap_rate = data[0]

        def check(expect_list, actual_list, init_swap_rate):
            # size
            assert len(expect_list) == len(actual_list)
            # funcs
            for expect, actual in zip(expect_list, actual_list):
                assert expect(init_swap_rate) == approx(actual(init_swap_rate))

        def case1():
            # min_put_range < lower_strike <= upper_strike <max_put_range
            # min_call_range < max_call_range
            self.payoff_lower_strike = (self.max_put_range
                                        + self.min_put_range) * 0.5
            self.payoff_upper_strike = (self.max_put_range
                                        + self.min_put_range) * 0.5
            self._set_target()
            funcs = self.target.make_numerator_analytic_funcs()

            expect_list = [self._analytic_func1,
                           self._analytic_func21,
                           self._analytic_func22]
            check(expect_list, funcs, init_swap_rate)
        case1()

        # min_put_range < lower_strike < max_put_range
        # min_callrange < upper_strike < max_call_range
        def case2():
            self.payoff_lower_strike = (self.max_put_range
                                        + self.min_put_range) * 0.5
            self.payoff_upper_strike = (self.max_call_range
                                        + self.min_call_range) * 0.5
            self._set_target()
            funcs = self.target.make_numerator_analytic_funcs()
            expect_list = [self._analytic_func1,
                           self._analytic_func21,
                           self._analytic_func32]
            check(expect_list, funcs, init_swap_rate)
        case2()

        # min_put_range < lower_strike < max_put_range
        # min_call_range <  max_call_range < upper_strike
        def case3():
            self.payoff_lower_strike = (self.max_put_range
                                        + self.min_put_range) * 0.5
            self.payoff_upper_strike = self.max_call_range * 1.1
            self._set_target()
            funcs = self.target.make_numerator_analytic_funcs()
            expect_list = [self._analytic_func1,
                           self._analytic_func21]
            check(expect_list, funcs, init_swap_rate)
        case3()

        # min_put_range < max_put_range
        # min_call_range < lower_strike <= upper_strike < max_call_range
        def case4():
            self.payoff_lower_strike = (self.max_call_range
                                        + self.min_call_range) * 0.5
            self.payoff_upper_strike = (self.max_call_range
                                        + self.min_call_range) * 0.5
            self._set_target()
            funcs = self.target.make_numerator_analytic_funcs()
            expect_list = [self._analytic_func1,
                           self._analytic_func31,
                           self._analytic_func32]
            check(expect_list, funcs, init_swap_rate)
        case4()

        # min_put_range <= max_put_range
        # min_call_range <= lower_strike <= max_call_range <= upper_strike
        def case5():
            self.payoff_lower_strike = (self.max_call_range
                                        + self.min_call_range) * 0.5
            self.payoff_upper_strike = self.max_call_range * 1.1
            self._set_target()
            funcs = self.target.make_numerator_analytic_funcs()
            expect_list = [self._analytic_func1,
                           self._analytic_func31]
            check(expect_list, funcs, init_swap_rate)
        case5()

    def test_model__replicate_numerator(self):
        # correlation = 0, alpha0 = 0, payoff is bull spread
        # min_put_range <= lower_strike <= max_put_rage = min_call_range
        # max_put_range = min_call_range <= upper_strike <= max_call_range
        def case2():
            self.forward_fx_corr = 0.0
            self.alpha0 = 0.0

            self.min_put_range = 1e-10
            self.max_put_range = self.init_swap_rate
            self.min_call_range = self.init_swap_rate
            self.max_call_range = self.init_swap_rate * 3.0
            self.payoff_lower_strike = (self.min_put_range
                                        + self.max_put_range) * 0.5
            self.payoff_upper_strike = (self.min_call_range
                                        + self.max_call_range) * 0.5
            self._set_target()
            forward_fx_factor = math.exp(0.5
                                         * (self.forward_fx_vol ** 2)
                                         * self.maturity)
            call_value_lower = function.black_payers_swaption_value(
                self.init_swap_rate,
                self.payoff_lower_strike,
                1.0,
                self.maturity,
                self.swap_rate_vol)
            call_value_upper = function.black_payers_swaption_value(
                self.init_swap_rate,
                self.payoff_upper_strike,
                1.0,
                self.maturity,
                self.swap_rate_vol)
            expect = ((call_value_lower - call_value_upper)
                      * forward_fx_factor * self.alpha1 * self.payoff_gearing)
            actual = target._replicate_numerator(
                self.init_swap_rate,
                self.target,
                self.call_pricer,
                self.put_pricer,
                self.min_put_range,
                self.max_call_range)
            assert expect == approx(actual)
        case2()

    def test__replicate_denominator(self):
        # correlation = 0
        self.forward_fx_corr = 0.0
        self.min_put_range = 1e-10
        self.max_put_range = self.init_swap_rate
        self.min_call_range = self.init_swap_rate
        self.max_call_range = self.init_swap_rate * 3.0
        self._set_target()
        forward_fx_factor = math.exp(0.5
                                     * (self.forward_fx_vol ** 2)
                                     * self.maturity)
        expect = (self.alpha0 * self.init_swap_rate
                  + self.alpha1) * forward_fx_factor
        actual = target._replicate_denominator(
            self.init_swap_rate,
            self.call_pricer,
            self.put_pricer,
            self.annuity_mapping_helper,
            self.forward_fx_diffusion_helper,
            self.min_put_range,
            self.max_call_range)
        assert expect == approx(actual)

    def test_model_linear_tsr_bull_spread_quanto_cms(self):
        """
        model is redeuced to the following equation when
        :math:`alpha_{0}=0`, :math:`\\rho_{X}=0`.

        .. math::
            \mathrm{E}^{A}
            \left[
                g(S(T))
            \\right].
        """
        self.alpha0 = 0.0
        self.forward_fx_corr = 0.0

        self.gearing = 1.0 / self.init_swap_rate
        self.min_put_range = 1e-10
        self.max_put_range = self.init_swap_rate
        self.min_call_range = self.init_swap_rate
        self.max_call_range = self.init_swap_rate * 2.0
        self.payoff_lower_strike = (self.min_put_range
                                    + self.max_put_range) * 0.5
        self.payoff_upper_strike = self.init_swap_rate * 1.5
        self._set_target()
        actual = target.replicate(
            self.init_swap_rate,
            1.0,
            self.call_pricer,
            self.put_pricer,
            "bull_spread",
            self.payoff_params,
            self.forward_fx_diffusion_params,
            "linear",
            self.annuity_mapping_params,
            self.min_put_range,
            self.max_call_range)
        # expect
        call_value_lower_strike = function.black_payers_swaption_value(
            self.init_swap_rate,
            self.payoff_lower_strike,
            1.0,
            self.maturity,
            self.swap_rate_vol)
        call_value_upper_strike = function.black_payers_swaption_value(
            self.init_swap_rate,
            self.payoff_upper_strike,
            1.0,
            self.maturity,
            self.swap_rate_vol)
        expect = ((call_value_lower_strike - call_value_upper_strike)
                  * self.payoff_gearing)
        assert expect == approx(actual)


class TestModelPricerQuantoCMS(object):

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
        "underlying, rate, maturity, vol", [
            util.get_real(4)
        ])
    def test_black_scholes_pdf_atm(
            self, underlying, rate, maturity, vol):
        integrate = scipy.integrate
        # atm
        payoff_strike = underlying
        pdf = target.make_pdf_black_scholes(underlying, rate, maturity, vol)

        # call value
        def call_integrand(strike):
            return max(strike - payoff_strike, 0.0) * pdf(strike)
        call_value_integral = integrate.quad(
            call_integrand, payoff_strike, np.inf)[0]
        call_value_analytic = function.black_scholes_call_value(
            underlying, payoff_strike, rate, maturity, vol)
        assert call_value_integral == approx(call_value_analytic)

        # put value
        def put_integrand(strike):
            return max(payoff_strike - strike, 0.0) * pdf(strike)
        put_value_integral = integrate.quad(
            put_integrand, 0.0, payoff_strike)[0]
        put_value_analytic = function.black_scholes_put_value(
            underlying, payoff_strike, rate, maturity, vol)
        assert put_value_integral == approx(put_value_analytic)

    @pytest.mark.parametrize(
        "underlying, rate, maturity, vol", [
            util.get_real(4)
        ])
    def test_black_scholes_pdf_otm(
            self, underlying, rate, maturity, vol):
        integrate = scipy.integrate

        def case_call():
            # strike
            payoff_strike = underlying * 0.5
            pdf = target.make_pdf_black_scholes(
                underlying, rate, maturity, vol)

            def integrand(strike):
                return max(strike - payoff_strike, 0.0) * pdf(strike)
            value_integral = integrate.quad(
                integrand, payoff_strike, np.inf)[0]
            value_analytic = function.black_scholes_call_value(
                underlying, payoff_strike, rate, maturity, vol)
            assert value_integral == approx(value_analytic)
        case_call()

        # put value
        def case_put():
            # strike
            payoff_strike = underlying + (1.0 - underlying) * 0.5
            pdf = target.make_pdf_black_scholes(
                underlying, rate, maturity, vol)

            def integrand(strike):
                return max(payoff_strike - strike, 0.0) * pdf(strike)
            value_integral = integrate.quad(
                integrand, 0.0, payoff_strike)[0]
            value_analytic = function.black_scholes_put_value(
                underlying, payoff_strike, rate, maturity, vol)
            assert value_integral == approx(value_analytic)
        case_put()

    @pytest.mark.parametrize(
        "underlying, rate, maturity, vol", [
            util.get_real(4)
        ])
    def test_black_scholes_pdf_itm(
            self, underlying, rate, maturity, vol):
        integrate = scipy.integrate

        def case_call():
            # strike
            payoff_strike = underlying + (1.0 - underlying) * 0.5
            pdf = target.make_pdf_black_scholes(
                underlying, rate, maturity, vol)

            def integrand(strike):
                return max(strike - payoff_strike, 0.0) * pdf(strike)
            value_integral = integrate.quad(
                integrand, payoff_strike, np.inf)[0]
            value_analytic = function.black_scholes_call_value(
                underlying, payoff_strike, rate, maturity, vol)
            assert value_integral == approx(value_analytic)
        case_call()

        # put value
        def case_put():
            # strike
            payoff_strike = underlying * 0.5
            pdf = target.make_pdf_black_scholes(
                underlying, rate, maturity, vol)

            def integrand(strike):
                return max(payoff_strike - strike, 0.0) * pdf(strike)
            value_integral = integrate.quad(
                integrand, 0.0, payoff_strike)[0]
            value_analytic = function.black_scholes_put_value(
                underlying, payoff_strike, rate, maturity, vol)
            assert value_integral == approx(value_analytic)
        case_put()
