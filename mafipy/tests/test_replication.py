#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division

import mafipy.replication as target
import pytest
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
