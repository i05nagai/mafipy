#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from pytest import approx
import math
import pytest
import scipy.stats

from . import util
import mafipy.function as function
from mafipy.replication import _quanto_cms_forward_fx as target


class Test_QuantoCmsForwardFx(object):

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
                 * function.norm_pdf_fprime(h) * h_fprime)
        denominator = norm.pdf(h) ** 2
        expect = (term1 - term2) / denominator
        # actual
        actual = target._calc_h_fhess(
            swap_rate_pdf_fprime, swap_rate_pdf, swap_rate, h, h_fprime)
        assert expect == approx(actual)

    @pytest.mark.parametrize(
        "swap_rate, time, vol, corr", [
            # is_inverse=False
            # vol < 0.0 raise AssertionError
            (2.1, 1.2, -1.3, 0.1),
            # corr < -1.0 raise AssertionError
            (2.1, 1.2, 1.3, -1.1),
            # corr > -1.0 raise AssertionError
            (2.1, 1.2, 1.3, 1.1),
            # otherwise
            (2.1, 1.2, 1.3, 0.1),
        ])
    def test_forward_fx_diffusion_false(
            self, swap_rate, time, vol, corr):
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
                                                      swap_rate_pdf_fprime,
                                                      False)
            return
        else:
            # expect
            term1 = corr * vol * math.sqrt(time) * h
            term2 = (1 - corr * corr) * vol * vol * time * 0.5
            expect = math.exp(term1 + term2)
            # actual
            actual = target._forward_fx_diffusion(swap_rate,
                                                  time,
                                                  vol,
                                                  corr,
                                                  swap_rate_cdf,
                                                  swap_rate_pdf,
                                                  swap_rate_pdf_fprime,
                                                  False)
            assert expect == approx(actual)

    # is_inverse = False
    @pytest.mark.parametrize(
        "swap_rate, time, vol, corr", [
            # vol < 0.0 raise AssertionError
            (2.1, 1.2, -1.3, 0.1),
            # corr < -1.0 raise AssertionError
            (2.1, 1.2, 1.3, -1.1),
            # corr > -1.0 raise AssertionError
            (2.1, 1.2, 1.3, 1.1),
            # otherwise
            (2.1, 1.2, 1.3, 0.1),
        ])
    def test_forward_fx_diffusion_true(
            self, swap_rate, time, vol, corr):
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
                                                      swap_rate_pdf_fprime,
                                                      True)
            return
        else:
            # expect
            term1 = corr * vol * math.sqrt(time) * h
            term2 = (1.0 - corr * corr) * vol * vol * time * 0.5
            expect = math.exp(-term1 + term2)
            # actual
            actual = target._forward_fx_diffusion(swap_rate,
                                                  time,
                                                  vol,
                                                  corr,
                                                  swap_rate_cdf,
                                                  swap_rate_pdf,
                                                  swap_rate_pdf_fprime,
                                                  True)
            assert expect == approx(actual)

    # is_inverse=True
    @pytest.mark.parametrize(
        "swap_rate, time, vol, corr", [
            # vol < 0.0 raise AssertionError
            (2.1, 1.2, -1.3, 0.1),
            # corr < -1.0 raise AssertionError
            (2.1, 1.2, 1.3, -1.1),
            # corr > -1.0 raise AssertionError
            (2.1, 1.2, 1.3, 1.1),
            # otherwise
            (2.1, 1.2, 1.3, 0.1),
        ])
    def test_forward_fx_diffusion_fprime_false(
            self, swap_rate, time, vol, corr):
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
                    swap_rate_cdf, swap_rate_pdf, swap_rate_pdf_fprime, False)
        else:
            # expect
            forward_fx_diffusion = target._forward_fx_diffusion(
                swap_rate,
                time,
                vol,
                corr,
                swap_rate_cdf,
                swap_rate_pdf,
                swap_rate_pdf_fprime,
                False)
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
                                                         swap_rate_pdf_fprime,
                                                         False)
            assert expect == approx(actual)

    # is_inverse=True
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
    def test_forward_fx_diffusion_fprime_true(
            self, swap_rate, time, vol, corr):
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
                    swap_rate_cdf, swap_rate_pdf, swap_rate_pdf_fprime, True)
        else:
            # expect
            forward_fx_diffusion = target._forward_fx_diffusion(
                swap_rate,
                time,
                vol,
                corr,
                swap_rate_cdf,
                swap_rate_pdf,
                swap_rate_pdf_fprime,
                True)
            h_fprime = target._calc_h_fprime(swap_rate_pdf, swap_rate, h)
            expect = -(forward_fx_diffusion
                       * corr * vol * math.sqrt(time) * h_fprime)
            # actual
            actual = target._forward_fx_diffusion_fprime(swap_rate,
                                                         time,
                                                         vol,
                                                         corr,
                                                         swap_rate_cdf,
                                                         swap_rate_pdf,
                                                         swap_rate_pdf_fprime,
                                                         True)
            assert expect == approx(actual)

    # is_inverse = False
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
    def test_forward_fx_diffusion_fhess_false(
            self, swap_rate, time, vol, corr):
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
                    swap_rate_cdf, swap_rate_pdf, swap_rate_pdf_fprime, False)
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
                swap_rate_pdf_fprime,
                False)
            forward_fx_diffusion_fprime = target._forward_fx_diffusion_fprime(
                swap_rate,
                time,
                vol,
                corr,
                swap_rate_cdf,
                swap_rate_pdf,
                swap_rate_pdf_fprime,
                False)
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
                                                        swap_rate_pdf_fprime,
                                                        False)
            assert expect == approx(actual)

    # is_inverse = True
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
    def test_forward_fx_diffusion_fhess_true(self,
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
                    swap_rate_cdf, swap_rate_pdf, swap_rate_pdf_fprime, True)
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
                swap_rate_pdf_fprime,
                True)
            factor = corr * vol * math.sqrt(time)
            term1 = h_fhess
            term2 = h_fprime * h_fprime * factor
            expect = (-term1 + term2) * factor * forward_fx_diffusion
            # actual
            actual = target._forward_fx_diffusion_fhess(swap_rate,
                                                        time,
                                                        vol,
                                                        corr,
                                                        swap_rate_cdf,
                                                        swap_rate_pdf,
                                                        swap_rate_pdf_fprime,
                                                        True)
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
        data_bool = util.get_bool()
        self.time = data[0]
        self.vol = data[1]
        self.corr = data[2]
        self.is_inverse = data_bool[0]
        self.target = target._ForwardFxDiffusionHelper(
            self.time,
            self.vol,
            self.corr,
            self._swap_rate_cdf,
            self._swap_rate_pdf,
            self._swap_rate_pdf_fprime,
            self.is_inverse)

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
            self._swap_rate_pdf_fprime,
            self.is_inverse)
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
            self._swap_rate_pdf_fprime,
            self.is_inverse)
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
            self._swap_rate_pdf_fprime,
            self.is_inverse)
        actual = self.target.make_fhess()(swap_rate)
        assert expect == approx(actual)
