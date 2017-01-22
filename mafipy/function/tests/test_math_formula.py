#!/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
from pytest import approx
import pytest

import mafipy.function as target


class TestMathFormula:

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

    @pytest.mark.parametrize("x, expect", [
        # x = 0
        (0.0, 0.0),
        # x != 0
        (1.43, -1.43 * 0.14350455054006242),
    ])
    def test_norm_pdf_fprime(self, x, expect):
        actual = target.norm_pdf_fprime(x)
        assert expect == approx(actual)

    @pytest.mark.parametrize("x, expect", [
        # x = 1
        (1.0, 0.0),
        # x != 1
        (1.43, (1.43 * 1.43 - 1.0) * 0.14350455054006242),
    ])
    def test_norm_pdf_fhess(self, x, expect):
        actual = target.norm_pdf_fhess(x)
        assert expect == approx(actual)
