from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mafipy.math.qmc.sobol as target


def test_make_sobol():
    # dim1
    def case1():
        generator = target.make_sobol(1)
        expect = [
            [0.0],
            [0.5],
            [0.75],
            [0.25],
            [0.375],
            [0.875],
            [0.625],
            [0.125],
        ]
        actual = [generator.next() for i in range(len(expect))]
        assert expect == actual
    case1()

    # dim2
    def case2():
        generator = target.make_sobol(2)
        expect = [
            [0.0, 0.0],
            [0.5, 0.5],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.375, 0.125],
            [0.875, 0.625],
            [0.625, 0.375],
            [0.125, 0.875],
        ]
        actual = [generator.next() for i in range(len(expect))]
        assert expect == actual
    case2()
