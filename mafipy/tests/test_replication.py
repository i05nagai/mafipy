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
