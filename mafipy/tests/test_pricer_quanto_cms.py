#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

from pytest import approx
import mafipy.analytic_formula as analytic_formula
import mafipy.math_formula as math_formula
import mafipy.payoff as payoff
import mafipy.pricer_quanto_cms as target
import mafipy.replication as replication
import mafipy.tests.util as util
import math
import pytest
import scipy.stats


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
            expect = analytic_formula.black_scholes_call_value_third_by_strike(
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
            af = analytic_formula
            expect = (1.0 + af.black_scholes_call_value_fprime_by_strike(
                underlying, strike, rate, maturity, vol))
            actual = target.make_cdf_black_scholes(
                underlying, rate, maturity, vol)(strike)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "swap_rate", [
            (2.0),
        ])
    def test__calc_h(self, swap_rate):
        def swap_rate_cdf(swap_rate):
            return (swap_rate * 0.9) / swap_rate
        norm = scipy.stats.norm
        expect = norm.ppf(swap_rate_cdf(swap_rate))

        actual = target._calc_h(swap_rate_cdf, swap_rate)
        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "swap_rate", [
            (2.0),
        ])
    def test__calc_h_fprime(self, swap_rate):
        norm = scipy.stats.norm

        def swap_rate_cdf(swap_rate):
            return norm.cdf(swap_rate)

        def swap_rate_pdf(swap_rate):
            return norm.pdf(swap_rate)

        h = target._calc_h(swap_rate_cdf, swap_rate)

        expect = swap_rate_pdf(swap_rate) / norm.pdf(h)
        actual = target._calc_h_fprime(swap_rate_pdf, swap_rate, h)
        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "swap_rate", [
            (2.0),
        ])
    def test__calc_h_fhess(self, swap_rate):
        norm = scipy.stats.norm

        def swap_rate_cdf(s):
            return norm.cdf(s)

        def swap_rate_pdf(s):
            return norm.pdf(s)

        def swap_rate_pdf_fprime(s):
            return -s * norm.pdf(s)

        h = target._calc_h(swap_rate_cdf, swap_rate)
        h_fprime = target._calc_h_fprime(swap_rate_pdf, swap_rate, h)

        # expect
        term1 = swap_rate_pdf_fprime(swap_rate) * norm.pdf(h)
        term2 = (swap_rate_pdf(swap_rate)
                 * math_formula.norm_pdf_fprime(h) * h_fprime)
        denominator = norm.pdf(h) ** 2
        expect = (term1 - term2) / denominator
        # actual
        actual = target._calc_h_fhess(
            swap_rate_pdf_fprime, swap_rate_pdf, swap_rate, h, h_fprime)
        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "swap_rate, time, vol, corr", [
            # vol < 0.0 raise AssertionError
            (2.0, 1.0, -1.0, 0.1),
            # corr < -1.0 raise AssertionError
            (2.0, 1.0, 1.0, -1.1),
            # corr > -1.0 raise AssertionError
            (2.0, 1.0, 1.0, 1.1),
            # otherwise
            (2.0, 1.0, 1.0, 0.1),
        ])
    def test_forward_fx_diffusion(self,
                                  swap_rate,
                                  time,
                                  vol,
                                  corr):
        norm = scipy.stats.norm

        def swap_rate_cdf(s):
            return norm.cdf(s)

        def swap_rate_pdf(s):
            return norm.pdf(s)

        def swap_rate_pdf_fprime(s):
            return -s * norm.pdf(s)

        h = target._calc_h(swap_rate_cdf, swap_rate)

        # raise AssertionError
        if -1.0 > corr or 1.0 < corr or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target._forward_fx_diffusion(swap_rate,
                                                      time,
                                                      vol,
                                                      corr,
                                                      swap_rate_cdf,
                                                      swap_rate_pdf,
                                                      swap_rate_pdf_fprime)
        else:
            # expect
            term1 = corr * vol * h
            term2 = (1 - corr * corr) * vol * vol * time * 0.5
            expect = math.exp(term1 + term2)
            # actual
            actual = target._forward_fx_diffusion(swap_rate,
                                                  time,
                                                  vol,
                                                  corr,
                                                  swap_rate_cdf,
                                                  swap_rate_pdf,
                                                  swap_rate_pdf_fprime)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "swap_rate, time, vol, corr", [
            # vol < 0.0 raise AssertionError
            (2.0, 1.0, -1.0, 0.1),
            # corr < -1.0 raise AssertionError
            (2.0, 1.0, 1.0, -1.1),
            # corr > -1.0 raise AssertionError
            (2.0, 1.0, 1.0, 1.1),
            # otherwise
            (2.0, 1.0, 1.0, 0.1),
        ])
    def test_forward_fx_diffusion_fprime(self,
                                         swap_rate,
                                         time,
                                         vol,
                                         corr):
        norm = scipy.stats.norm

        def swap_rate_cdf(s):
            return norm.cdf(s)

        def swap_rate_pdf(s):
            return norm.pdf(s)

        def swap_rate_pdf_fprime(s):
            return -s * norm.pdf(s)

        h = target._calc_h(swap_rate_cdf, swap_rate)

        # raise AssertionError
        if -1.0 > corr or 1.0 < corr or vol < 0.0:
            with pytest.raises(AssertionError):
                actual = target._forward_fx_diffusion_fprime(
                    swap_rate, time, vol, corr,
                    swap_rate_cdf, swap_rate_pdf, swap_rate_pdf_fprime)
        else:
            # expect
            forward_fx_diffusion = target._forward_fx_diffusion(
                swap_rate,
                time,
                vol,
                corr,
                swap_rate_cdf,
                swap_rate_pdf,
                swap_rate_pdf_fprime)
            h_fprime = target._calc_h_fprime(swap_rate_pdf, swap_rate, h)
            expect = (forward_fx_diffusion
                      * corr * vol * math.sqrt(time) * h_fprime)
            # actual
            actual = target._forward_fx_diffusion_fprime(swap_rate,
                                                         time,
                                                         vol,
                                                         corr,
                                                         swap_rate_cdf,
                                                         swap_rate_pdf,
                                                         swap_rate_pdf_fprime)
            assert expect == approx(actual)

    @pytest.mark.parametrize(
        "swap_rate, time, vol, corr", [
            # time < 0.0 raise AssertionError
            (2.0, -1.0, 1.0, 0.1),
            # vol < 0.0 raise AssertionError
            (2.0, 1.0, -1.0, 0.1),
            # corr < -1.0 raise AssertionError
            (2.0, 1.0, 1.0, -1.1),
            # corr > -1.0 raise AssertionError
            (2.0, 1.0, 1.0, 1.1),
            # otherwise
            (2.0, 1.0, 1.0, 0.1),
        ])
    def test_forward_fx_diffusion_fhess(self,
                                        swap_rate,
                                        time,
                                        vol,
                                        corr):
        norm = scipy.stats.norm

        def swap_rate_cdf(s):
            return norm.cdf(s)

        def swap_rate_pdf(s):
            return norm.pdf(s)

        def swap_rate_pdf_fprime(s):
            return -s * norm.pdf(s)

        # raise AssertionError
        if (time < 0.0
                or -1.0 > corr
                or 1.0 < corr
                or vol < 0.0):
            with pytest.raises(AssertionError):
                actual = target._forward_fx_diffusion_fhess(
                    swap_rate, time, vol, corr,
                    swap_rate_cdf, swap_rate_pdf, swap_rate_pdf_fprime)
        else:
            # expect
            h = target._calc_h(swap_rate_cdf, swap_rate)
            h_fprime = target._calc_h_fprime(swap_rate_pdf, swap_rate, h)
            h_fhess = target._calc_h_fhess(
                swap_rate_pdf_fprime, swap_rate_pdf, swap_rate, h, h_fprime)
            forward_fx_diffusion = target._forward_fx_diffusion(
                swap_rate,
                time,
                vol,
                corr,
                swap_rate_cdf,
                swap_rate_pdf,
                swap_rate_pdf_fprime)
            forward_fx_diffusion_fprime = target._forward_fx_diffusion_fprime(
                swap_rate,
                time,
                vol,
                corr,
                swap_rate_cdf,
                swap_rate_pdf,
                swap_rate_pdf_fprime)
            term1 = forward_fx_diffusion * h_fhess
            term2 = h_fprime * forward_fx_diffusion_fprime
            expect = (term1 + term2) * corr * vol * math.sqrt(time)
            # actual
            actual = target._forward_fx_diffusion_fhess(swap_rate,
                                                        time,
                                                        vol,
                                                        corr,
                                                        swap_rate_cdf,
                                                        swap_rate_pdf,
                                                        swap_rate_pdf_fprime)
            assert expect == approx(actual)


class Test_ForwardFxDiffusionHelper(object):

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
        data = util.get_real(3)
        self.time = data[0]
        self.vol = data[1]
        self.corr = data[2]
        self.target = target._ForwardFxDiffusionHelper(
            self.time,
            self.vol,
            self.corr,
            self._swap_rate_cdf,
            self._swap_rate_pdf,
            self._swap_rate_pdf_fprime)

    # after each test finish
    def teardown(self):
        pass

    def _swap_rate_cdf(self, s):
        return scipy.stats.norm.cdf(s)

    def _swap_rate_pdf(self, s):
        return scipy.stats.norm.pdf(s)

    def _swap_rate_pdf_fprime(self, s):
        return -s * scipy.stats.norm.pdf(s)

    @pytest.mark.parametrize("swap_rate", [
        util.get_real_t(1)[0],
    ])
    def test_make_func(self, swap_rate):
        expect = target._forward_fx_diffusion(
            swap_rate,
            self.time,
            self.vol,
            self.corr,
            self._swap_rate_cdf,
            self._swap_rate_pdf,
            self._swap_rate_pdf_fprime)
        actual = self.target.make_func()(swap_rate)
        assert expect == approx(actual)

    @pytest.mark.parametrize("swap_rate", [
        util.get_real_t(1)[0],
    ])
    def test_make_fprime(self, swap_rate):
        expect = target._forward_fx_diffusion_fprime(
            swap_rate,
            self.time,
            self.vol,
            self.corr,
            self._swap_rate_cdf,
            self._swap_rate_pdf,
            self._swap_rate_pdf_fprime)
        actual = self.target.make_fprime()(swap_rate)
        assert expect == approx(actual)

    @pytest.mark.parametrize("swap_rate", [
        util.get_real_t(1)[0],
    ])
    def test_make_fhess(self, swap_rate):
        expect = target._forward_fx_diffusion_fhess(
            swap_rate,
            self.time,
            self.vol,
            self.corr,
            self._swap_rate_cdf,
            self._swap_rate_pdf,
            self._swap_rate_pdf_fprime)
        actual = self.target.make_fhess()(swap_rate)
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
        data = util.get_real(14)
        self.alpha0 = data[0]
        self.alpha1 = data[1]
        self.payoff_underlying = data[3]
        self.gearing = data[4]
        self.maturity = data[5]
        self.forward_fx_vol = data[6]
        self.forward_fx_corr = data[7]
        self.init_swap_rate = data[8]
        self.rate = data[9]
        self.swap_rate_vol = data[10]
        self.today = 0.0
        self.init_swap_rate = data[11]
        self.rate = data[12]
        self.payoff_strike = data[13]
        # ordered data
        data = sorted(util.get_real(2))
        min_put_range = data[0]
        max_put_range = data[1]
        data = sorted(util.get_real(2))
        min_call_range = data[0]
        max_call_range = data[1]
        # set
        self._set_target(min_put_range,
                         max_put_range,
                         min_call_range,
                         max_call_range)

    # after each test finish
    def teardown(self):
        pass

    def _set_target(self,
                    min_put_range,
                    max_put_range,
                    min_call_range,
                    max_call_range):
        # range
        self.min_put_range = min_put_range
        self.max_put_range = max_put_range
        self.min_call_range = min_call_range
        self.max_call_range = max_call_range
        # annuity mapping params
        annuity_mapping_params = {
            "alpha0": self.alpha0,
            "alpha1": self.alpha1
        }
        annuity_mapping_helper = replication.LinearAnnuityMappingFuncHelper(
            **annuity_mapping_params)
        self.annuity_mapping_func = annuity_mapping_helper.make_func()
        self.annuity_mapping_fprime = annuity_mapping_helper.make_fprime()
        self.annuity_mapping_fhess = annuity_mapping_helper.make_fhess()
        # payoff helper
        self.payoff_params = {
            "underlying": self.payoff_underlying,
            "gearing": self.gearing,
        }
        call_payoff_helper = payoff.CallStrikePayoffHelper(
            **self.payoff_params)
        self.payoff_func = call_payoff_helper.make_func()
        self.payoff_fprime = call_payoff_helper.make_fprime()
        self.payoff_fhess = call_payoff_helper.make_fhess()
        # foward fx diffusion
        forward_fx_diffusion_params = {
            "time": self.maturity,
            "vol": self.forward_fx_vol,
            "corr": self.forward_fx_corr,
            "swap_rate_cdf": self._swap_rate_cdf,
            "swap_rate_pdf": self._swap_rate_pdf,
            "swap_rate_pdf_fprime": self._swap_rate_pdf_fprime
        }
        forward_fx_diffusion_helper = target._ForwardFxDiffusionHelper(
            **forward_fx_diffusion_params)
        self.forward_fx_diffusion = forward_fx_diffusion_helper.make_func()
        self.forward_fx_diffusion_fprime = (
            forward_fx_diffusion_helper.make_fprime())
        self.forward_fx_diffusion_fhess = (
            forward_fx_diffusion_helper.make_fhess())
        # pricer
        bs_pricer = analytic_formula.BlackScholesPricerHelper()
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
            min_put_range,
            max_put_range,
            min_call_range,
            max_call_range)

    def _swap_rate_cdf(self, swap_rate):
        return scipy.stats.norm.cdf(swap_rate)

    def _swap_rate_pdf(self, swap_rate):
        return scipy.stats.norm.pdf(swap_rate)

    def _swap_rate_pdf_fprime(self, swap_rate):
        return math_formula.norm_pdf_fprime(swap_rate)

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

        def case1():
            min_put_range = self.payoff_strike / 2.0
            max_put_range = self.payoff_strike * 1.01
            min_call_range = self.payoff_strike / 2.0
            max_call_range = self.payoff_strike * 1.01
            self._set_target(min_put_range,
                             max_put_range,
                             min_call_range,
                             max_call_range)
            funcs = self.target.make_numerator_analytic_funcs()

            # size
            assert 3 == len(funcs)
            # func1
            actual = funcs[0](init_swap_rate)
            expect = (self.payoff_func(init_swap_rate)
                      * self.annuity_mapping_func(init_swap_rate)
                      * self.forward_fx_diffusion(init_swap_rate))
            assert expect == approx(actual)
            # func2
            actual = funcs[1](init_swap_rate)
            expect = (self.put_pricer(self.payoff_strike)
                      * self.annuity_mapping_func(self.payoff_strike)
                      * self.forward_fx_diffusion(self.payoff_strike))
            assert expect == approx(actual)
            # func3
            actual = funcs[2](init_swap_rate)
            expect = (self.call_pricer(self.payoff_strike)
                      * self.annuity_mapping_func(self.payoff_strike)
                      * self.forward_fx_diffusion(self.payoff_strike))
            assert expect == approx(actual)
        case1()

        # min_put_range < strike < max_put_range
        # strike < min_call_range < max_call_range
        def case2():
            min_put_range = self.payoff_strike / 2.0
            max_put_range = self.payoff_strike * 1.01
            min_call_range = self.payoff_strike * 1.005
            max_call_range = self.payoff_strike * 1.01
            self._set_target(min_put_range,
                             max_put_range,
                             min_call_range,
                             max_call_range)
            funcs = self.target.make_numerator_analytic_funcs()
            # size
            assert 2 == len(funcs)
            # func1
            actual = funcs[0](init_swap_rate)
            expect = (self.payoff_func(init_swap_rate)
                      * self.annuity_mapping_func(init_swap_rate)
                      * self.forward_fx_diffusion(init_swap_rate))
            assert expect == approx(actual)
            # func2
            actual = funcs[1](init_swap_rate)
            expect = (self.put_pricer(self.payoff_strike)
                      * self.annuity_mapping_func(self.payoff_strike)
                      * self.forward_fx_diffusion(self.payoff_strike))
            assert expect == approx(actual)
        case2()

        # strike < min_put_range < max_put_range
        # min_call_range < strike < max_call_range
        def case3():
            min_put_range = self.payoff_strike * 1.005
            max_put_range = self.payoff_strike * 1.01
            min_call_range = self.payoff_strike / 2.0
            max_call_range = self.payoff_strike * 1.01
            self._set_target(min_put_range,
                             max_put_range,
                             min_call_range,
                             max_call_range)
            funcs = self.target.make_numerator_analytic_funcs()
            # size
            assert 2 == len(funcs)
            # func1
            actual = funcs[0](init_swap_rate)
            expect = (self.payoff_func(init_swap_rate)
                      * self.annuity_mapping_func(init_swap_rate)
                      * self.forward_fx_diffusion(init_swap_rate))
            assert expect == approx(actual)
            # func3
            actual = funcs[1](init_swap_rate)
            expect = (self.call_pricer(self.payoff_strike)
                      * self.annuity_mapping_func(self.payoff_strike)
                      * self.forward_fx_diffusion(self.payoff_strike))
            assert expect == approx(actual)
        case3()

        # strike < min_put_range < max_put_range
        # strike < min_call_range < max_call_range
        def case4():
            min_put_range = self.payoff_strike * 1.005
            max_put_range = self.payoff_strike * 1.01
            min_call_range = self.payoff_strike * 1.005
            max_call_range = self.payoff_strike * 1.01
            self._set_target(min_put_range,
                             max_put_range,
                             min_call_range,
                             max_call_range)
            funcs = self.target.make_numerator_analytic_funcs()
            # size
            assert 1 == len(funcs)
            # func1
            actual = funcs[0](init_swap_rate)
            expect = (self.payoff_func(init_swap_rate)
                      * self.annuity_mapping_func(init_swap_rate)
                      * self.forward_fx_diffusion(init_swap_rate))
            assert expect == approx(actual)
        case4()

    def test__make_denominator_integrands(self):
        data = util.get_real(1)
        swap_rate = data[0]
        funcs = self.target._make_denominator_integrands()
        # size
        assert 3 == len(funcs)
        # func1
        actual = funcs[0](swap_rate)
        expect = (self.annuity_mapping_fhess(swap_rate)
                  * self.forward_fx_diffusion(swap_rate))
        assert expect == approx(actual)
        # func2
        actual = funcs[1](swap_rate)
        expect = (self.annuity_mapping_func(swap_rate)
                  * self.forward_fx_diffusion_fhess(swap_rate))
        assert expect == approx(actual)
        # func3
        actual = funcs[2](swap_rate)
        expect = (2.0
                  * self.annuity_mapping_fprime(swap_rate)
                  * self.forward_fx_diffusion_fprime(swap_rate))
        assert expect == approx(actual)

    def test_make_denominator_call_integrands(self):
        # just call _make_denominator_call_integrands()
        # so that no needs to test
        pass

    def test_make_denominator_put_integrands(self):
        # just call _make_denominator_call_integrands()
        # so that no needs to test
        pass

    def test_make_denominator_analytic_func(self):
        data = util.get_real(1)
        init_swap_rate = data[0]
        funcs = self.target.make_denominator_analytic_funcs()
        # size
        assert 1 == len(funcs)
        # func1
        actual = funcs[0](init_swap_rate)
        expect = (self.annuity_mapping_func(init_swap_rate)
                  * self.forward_fx_diffusion(init_swap_rate))
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

    def test_eval(self):
        pass
