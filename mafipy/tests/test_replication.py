#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import mafipy.replication as target
import pytest
from mafipy.tests import util
from pytest import approx


class TestReplication:

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

    @pytest.mark.parametrize("underlying, alpha0, alpha1, expect", [
        (1.434, 0.43, 0.51, 1.434 * 0.43 + 0.51)
    ])
    def test_linear_annuity_mapping_func(
            self, underlying, alpha0, alpha1, expect):
        actual = target.linear_annuity_mapping_func(underlying, alpha0, alpha1)
        assert expect == approx(actual)

    @pytest.mark.parametrize("underlying, alpha0, alpha1, expect", [
        (1.434, 0.43, 0.51, 0.43)
    ])
    def test_linear_annuity_mapping_fprime(
            self, underlying, alpha0, alpha1, expect):
        actual = target.linear_annuity_mapping_fprime(
            underlying, alpha0, alpha1)
        assert expect == approx(actual)

    @pytest.mark.parametrize("underlying, alpha0, alpha1, expect", [
        (1.434, 0.43, 0.51, 0.0)
    ])
    def test_linear_annuity_mapping_fhess(
            self, underlying, alpha0, alpha1, expect):
        actual = target.linear_annuity_mapping_fhess(
            underlying, alpha0, alpha1)
        assert expect == approx(actual)


class TestLinearAnnuityMappingFuncHelper(object):

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
        data = util.get_real(2)
        self.alpha0 = data[0]
        self.alpha1 = data[1]
        params = {
            "alpha0": self.alpha0,
            "alpha1": self.alpha1,
        }
        # target
        self.target = target.LinearAnnuityMappingFuncHelper(**params)

    # after each test finish
    def teardown(self):
        pass

    def test_make_func(self):
        swap_rate = util.get_real()[0]
        expect = target.linear_annuity_mapping_func(
            swap_rate, self.alpha0, self.alpha1)
        actual = self.target.make_func()(swap_rate)
        assert expect == approx(actual)

    def test_make_fprime(self):
        swap_rate = util.get_real()[0]
        expect = target.linear_annuity_mapping_fprime(
            swap_rate, self.alpha0, self.alpha1)
        actual = self.target.make_fprime()(swap_rate)
        assert expect == approx(actual)

    def test_make_fhess(self):
        swap_rate = util.get_real()[0]
        expect = target.linear_annuity_mapping_fhess(
            swap_rate, self.alpha0, self.alpha1)
        actual = self.target.make_fhess()(swap_rate)
        assert expect == approx(actual)


class TestAnalyticReplication(object):

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
        self.call_pricer = self._func1
        self.put_pricer = self._func1
        self.analytic_funcs = [self._func1]
        self.call_integrands = [self._func1]
        self.put_integrands = [self._func1]
        self.target = target.AnalyticReplication(self.call_pricer,
                                                 self.put_pricer,
                                                 self.analytic_funcs,
                                                 self.call_integrands,
                                                 self.put_integrands)

    # after each test finish
    def teardown(self):
        pass

    def _func1(self, x):
        return x

    def _integrate_func1(self, x):
        return 0.5 * x * x

    def test_eval(self):

        # integral range is 0
        def case1():
            init_swap_rate = util.get_real()[0]
            min_put_range = init_swap_rate
            max_call_range = init_swap_rate
            actual = self.target.eval(
                init_swap_rate, min_put_range, max_call_range)
            expect = init_swap_rate
            assert expect == approx(actual)
        case1()

        # normal
        def case2():
            data = sorted(util.get_real(3))
            init_swap_rate = data[1]
            min_put_range = data[0]
            max_call_range = data[2]
            actual = self.target.eval(
                init_swap_rate, min_put_range, max_call_range)
            # expect
            expect_analytic = init_swap_rate
            # put_pricer * term
            expect_put = (((init_swap_rate ** 3.0) - (min_put_range ** 3.0))
                          / 3.0)
            expect_call = (((max_call_range ** 3.0) - (init_swap_rate ** 3.0))
                           / 3.0)
            expect = expect_analytic + expect_put + expect_call
            assert expect == approx(actual)
        case2()
