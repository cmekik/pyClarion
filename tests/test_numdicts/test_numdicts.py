from itertools import product
import pyClarion.numdicts as nd

import unittest
import math
import itertools

from pyClarion.numdicts.numdicts import GradientTape, NumDict
from pyClarion.numdicts.ops import *


def linspace(a, b):
    r = 4  # represents how many divisions
    a = a*r
    b = b*r
    for i, j in itertools.product(tuple(range(a, b)), tuple(range(a, b))):
        yield i/r, j/r


class TestNumdictsAddition(unittest.TestCase):
    def test_addition_basic_functionality_defaults(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality for defaults
            d1 = nd.NumDict(default=i)
            d2 = nd.NumDict(default=j)
            self.assertAlmostEqual((d1+d2).default, d1.default+d2.default)
            d3 = nd.NumDict(default=3.0)
            self.assertAlmostEqual((d2+d1+d3).default, (d3+d1+d2).default)
            self.assertAlmostEqual((d3+d1+d2).default, (d1+d2+d3).default)

    def test_addition_basic_functionality_keys(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality for elements
            d1 = nd.NumDict({1: i, 2: (i+j)})
            d2 = nd.NumDict({1: j, 2: (i-j)})
            d3 = d1+d2
            self.assertAlmostEqual(d3[1], d1[1]+d2[1])
            self.assertAlmostEqual(d3[2], d1[2]+d2[2])

    def test_addition_basic_functionality_mixed(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality with mismatched keys
            d1 = nd.NumDict({1: i}, default=i)
            d2 = nd.NumDict({1: j, 2: j}, default=0)
            d3 = d1+d2
            self.assertAlmostEqual(d3[1], d1[1]+d2[1])
            self.assertAlmostEqual(d3[2], d1.default+d2[2])
            self.assertAlmostEqual(d3.default, d1.default+d2.default)

    def test_addition_basic_functionality_nodefault(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality with no default and a default
            d1 = nd.NumDict({1: i, 3: i}, default=i)
            d2 = nd.NumDict({1: j, 2: j})
            with self.assertRaises(KeyError):
                d1+d2

    def test_addition_differentiation(self):
        for i, j in linspace(-10, 10):
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for defaults
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d4 = d1 + d2
            d4, grads = tape.gradients(d4, (d1, d2))
            self.assertAlmostEqual(grads[0].default, 1.0)
            self.assertAlmostEqual(grads[1].default, 1.0)
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                d3 = d1+d2
            d3, grads = tape.gradients(d3, (d1, d2))
            self.assertAlmostEqual(grads[0][1], 1.0)
            self.assertAlmostEqual(grads[0][2], 1.0)
            self.assertAlmostEqual(grads[1][1], 1.0)
            self.assertAlmostEqual(grads[1][2], 1.0)


class TestNumdictsSubtraction(unittest.TestCase):
    def test_subtraction_basic_functionality_defaults(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality for default
            d1 = nd.NumDict(default=i)
            d2 = nd.NumDict(default=j)
            self.assertAlmostEqual((d1-d2).default, d1.default-d2.default)
            self.assertAlmostEqual((d2-d1).default, d2.default-d1.default)

    def test_subtraction_basic_functionality_keys(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality for elements
            d1 = nd.NumDict({1: i, 2: (i+j)})
            d2 = nd.NumDict({1: j, 2: (i-j)})
            d3 = d1-d2
            self.assertAlmostEqual(d3[1], d1[1]-d2[1])
            self.assertAlmostEqual(d3[2], d1[2]-d2[2])

    def test_subtraction_basic_functionality_mixed(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality with mismatched keys
            d1 = nd.NumDict({1: i}, default=i)
            d2 = nd.NumDict({1: j, 2: j}, default=0)
            d3 = d1-d2
            self.assertAlmostEqual(d3[1], d1[1]-d2[1])
            self.assertAlmostEqual(d3[2], d1.default-d2[2])
            self.assertAlmostEqual(d3.default, d1.default-d2.default)

    def test_subtraction_basic_functionality_nodefault(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality with no default and a default
            d1 = nd.NumDict({1: i, 3: i}, default=i)
            d2 = nd.NumDict({1: j, 2: j})
            with self.assertRaises(KeyError):
                d1-d2

    def test_subtraction_differentiation(self):
        for i, j in linspace(-10, 10):
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for defaults
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d4 = d1 - d2
            d4, grads = tape.gradients(d4, (d1, d2))
            self.assertAlmostEqual(grads[0].default, 1.0)
            self.assertAlmostEqual(grads[1].default, -1.0)
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                d3 = d1-d2
            d3, grads = tape.gradients(d3, (d1, d2))
            self.assertAlmostEqual(grads[0][1], 1.0)
            self.assertAlmostEqual(grads[0][2], 1.0)
            self.assertAlmostEqual(grads[1][1], -1.0)
            self.assertAlmostEqual(grads[1][2], -1.0)


class TestNumdictsMultiplication(unittest.TestCase):
    def test_multiplication_basic_functionality_default(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality for default
            d1 = nd.NumDict(default=i)
            d2 = nd.NumDict(default=j)
            self.assertAlmostEqual((d1*d2).default, d1.default*d2.default)
            # testing multiplying by 0
            d0 = nd.NumDict(default=0.0)
            self.assertAlmostEqual((d2*d0).default, 0.0)
            self.assertAlmostEqual((d1*d0).default, 0.0)
            # testing communative
            d3 = nd.NumDict(default=3.0)
            self.assertAlmostEqual((d2*d1*d3).default, (d3*d1*d2).default)
            self.assertAlmostEqual((d1*d2*d3).default, (d1*d2*d3).default)

    def test_multiplication_basic_functionality_keys(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality for elements
            d1 = nd.NumDict({1: i, 2: (i+j)})
            d2 = nd.NumDict({1: j, 2: (i-j)})
            d3 = d1*d2
            self.assertAlmostEqual(d3[1], d1[1]*d2[1])
            self.assertAlmostEqual(d3[2], d1[2]*d2[2])

    def test_multiplication_basic_functionality_mixed(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality with mismatched keys
            d1 = nd.NumDict({1: i}, default=i)
            d2 = nd.NumDict({1: j, 2: j}, default=0)
            d3 = d1*d2
            self.assertAlmostEqual(d3[1], d1[1]*d2[1])
            self.assertAlmostEqual(d3[2], d1.default*d2[2])
            self.assertAlmostEqual(d3.default, d1.default*d2.default)

    def test_multiplication_basic_functionality_nodefault(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality with no default and a default
            d1 = nd.NumDict({1: i, 3: i}, default=i)
            d2 = nd.NumDict({1: j, 2: j})
            with self.assertRaises(KeyError):
                d1*d2

    def test_multiplication_differentiation(self):
        for i, j in linspace(-10, 10):
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d3 = d1*d2
            d3, grads = tape.gradients(d3, (d1, d2))
            self.assertAlmostEqual(grads[0].default, d2.default)
            self.assertAlmostEqual(grads[1].default, d1.default)
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for multiple NumDicts
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d3 = d1*d1*d2
            d3, grads = tape.gradients(d3, (d1, d2))
            self.assertAlmostEqual(grads[0].default, d2.default*d1.default*2)
            self.assertAlmostEqual(grads[1].default, d1.default*d1.default)
            tape = nd.GradientTape()
            with tape:
                # testing differentiation with non NumDict elements
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d3 = d1*d2
                d4 = i*d3
            d3, grads = tape.gradients(d4, (d1, d2))
            self.assertAlmostEqual(grads[0].default, i*d2.default)
            self.assertAlmostEqual(grads[1].default, i*d1.default)
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                d3 = d1*d2
            d3, grads = tape.gradients(d3, (d1, d2))
            self.assertAlmostEqual(grads[0][1], d2[1])
            self.assertAlmostEqual(grads[0][2], d2[2])
            self.assertAlmostEqual(grads[1][1], d1[1])
            self.assertAlmostEqual(grads[1][2], d1[2])

    def test_productRule(self):
        for i, j in linspace(-10, 10):
            t1 = nd.GradientTape()
            with t1:
                # testing differentiation for default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d3 = nd.NumDict(default=i+j)
                d4 = (d1+d2)*d3
            d4, grad = t1.gradients(d4, (d1, d2, d3))
            self.assertAlmostEqual(grad[0].default, d3.default)
            self.assertAlmostEqual(grad[1].default, d3.default)
            self.assertAlmostEqual(grad[2].default, d1.default+d2.default)


class TestNumdictDivision(unittest.TestCase):
    def test_truediv_and_rtruediv_basic_functionality_defaults(self):
        for i, j in linspace(-10, 10):
            tape = nd.GradientTape()
            with tape:
                # testing basic functionality for default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                if(d1.default*d2.default == 0):
                    with self.assertRaises(ZeroDivisionError):
                        if(d1.default == 0):
                            d2/d1
                        else:
                            d1/d2
                    with self.assertRaises(ZeroDivisionError):
                        if(d1.default == 0):
                            NumDict.__truediv__(d2, d1)
                        else:
                            NumDict.__truediv__(d1, d2)
                    with self.assertRaises(ZeroDivisionError):
                        if(d1.default == 0):
                            NumDict.__rtruediv__(d1, d2)
                        else:
                            NumDict.__rtruediv__(d2, d1)
                else:
                    self.assertAlmostEqual(
                        (d1/d2).default, d1.default/d2.default)
                    self.assertAlmostEqual(NumDict.__truediv__(
                        d1, d2).default, (d1/d2).default)
                    self.assertAlmostEqual(NumDict.__truediv__(
                        d1, d2).default, NumDict.__rtruediv__(d2, d1).default)

    def test_truediv_and_rtruediv_basic_functionality_keys(self):
        for i, j in linspace(-10, 10):
            tape = nd.GradientTape()
            with tape:
                # testing basic functionality for elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                if(d1[1]*d1[2]*d2[1]*d2[2] == 0):
                    with self.assertRaises(ZeroDivisionError):
                        if(d1[1]*d1[2] == 0):
                            d2/d1
                        else:
                            d1/d2
                    with self.assertRaises(ZeroDivisionError):
                        if(d1[1]*d1[2] == 0):
                            NumDict.__truediv__(d2, d1)
                        else:
                            NumDict.__truediv__(d1, d2)
                    with self.assertRaises(ZeroDivisionError):
                        if(d1[1]*d1[2] == 0):
                            NumDict.__rtruediv__(d1, d2)
                        else:
                            NumDict.__rtruediv__(d2, d1)
                else:
                    d3 = d1/d2
                    self.assertAlmostEqual(d3[1], d1[1]/d2[1])
                    self.assertAlmostEqual(d3[2], d1[2]/d2[2])
                    d3 = NumDict.__truediv__(d1, d2)
                    self.assertAlmostEqual(d3[1], (d1/d2)[1])
                    self.assertAlmostEqual(d3[2], (d1/d2)[2])
                    self.assertAlmostEqual(NumDict.__truediv__(
                        d1, d2), NumDict.__rtruediv__(d2, d1))

    def test_truediv_and_rtruediv_basic_functionality_mixed(self):
        for i, j in linspace(-10, 10):
            tape = nd.GradientTape()
            with tape:
                # testing basic functionality with mismatched keys
                d1 = nd.NumDict({1: i}, default=i)
                d2 = nd.NumDict({1: j, 2: j}, default=1)
                if(d1[1]*d1.default*d2[1]*d2[2] == 0):
                    with self.assertRaises(ZeroDivisionError):
                        if(d1[1]*d1[2] == 0):
                            d2/d1
                        else:
                            d1/d2
                    with self.assertRaises(ZeroDivisionError):
                        if(d1[1]*d1[2] == 0):
                            NumDict.__truediv__(d2, d1)
                        else:
                            NumDict.__truediv__(d1, d2)
                    with self.assertRaises(ZeroDivisionError):
                        if(d1[1]*d1[2] == 0):
                            NumDict.__rtruediv__(d1, d2)
                        else:
                            NumDict.__rtruediv__(d2, d1)
                else:
                    d3 = d1/d2
                    self.assertAlmostEqual(d3[1], d1[1]/d2[1])
                    self.assertAlmostEqual(d3[2], d1.default/d2[2])
                    d3 = NumDict.__truediv__(d1, d2)
                    self.assertAlmostEqual(d3[1], (d1/d2)[1])
                    self.assertAlmostEqual(d3[2], (d1/d2)[2])
                    self.assertAlmostEqual(d3.default, d1.default/d2.default)
                    self.assertAlmostEqual(NumDict.__truediv__(
                        d1, d2), NumDict.__rtruediv__(d2, d1))

    def test_truediv_and_rtruediv_basic_functionality_nodefault(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality with mismatched keys
            d1 = nd.NumDict({1: i, 3: i}, default=i)
            d2 = nd.NumDict({1: j, 2: j})
            if(d1[1]*d1.default*d2[1]*d2[2] != 0):
                with self.assertRaises(KeyError):
                    d1/d2

    def test_truediv_and_rtruediv_differentiation(self):
        for i, j in linspace(-10, 10):
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for truediv and default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                if(d2.default != 0):
                    d3 = NumDict.__truediv__(d1, d2)
            if(d2.default != 0):
                d3, grads = tape.gradients(d3, (d1, d2))
                self.assertAlmostEqual(grads[0].default, 1/d2.default)
                self.assertAlmostEqual(grads[1].default,
                                       (-d1.default)/(d2.default**2))
            with tape:
                # testing differentiation for rtruediv and default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                if(d1.default != 0):
                    d4 = NumDict.__rtruediv__(d1, d2)
            if(d1.default != 0):
                d4, grads = tape.gradients(d4, (d1, d2))
                self.assertAlmostEqual(grads[0].default,
                                       (-d2.default)/(d1.default**2))
                self.assertAlmostEqual(grads[1].default, 1/d1.default)
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for truediv and elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                if(d2[1]*d2[2] != 0):
                    d3 = NumDict.__truediv__(d1, d2)
            if(d2[1]*d2[2] != 0):
                d3, grads = tape.gradients(d3, (d1, d2))
                self.assertAlmostEqual(grads[0][1], 1/d2[1])
                self.assertAlmostEqual(grads[0][2], 1/d2[2])
                self.assertAlmostEqual(grads[1][1], (-d1[1])/(d2[1]**2))
                self.assertAlmostEqual(grads[1][2], (-d1[2])/(d2[2]**2))
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for rtruediv and elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                if(d1[1]*d1[2] != 0):
                    d3 = NumDict.__rtruediv__(d1, d2)
            if(d1[1]*d1[2] != 0):
                d3, grads = tape.gradients(d3, (d1, d2))
                self.assertAlmostEqual(grads[0][1], (-d2[1])/(d1[1]**2))
                self.assertAlmostEqual(grads[0][2], (-d2[2])/(d1[2]**2))
                self.assertAlmostEqual(grads[1][1], 1/d1[1])
                self.assertAlmostEqual(grads[1][2], 1/d1[2])


class TestNumdictsPower(unittest.TestCase):
    def test_pow_and_rpow_basic_functionality_defaults(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality for default
            d1 = nd.NumDict(default=i)
            d2 = nd.NumDict(default=j)
            if(d1.default >= 0) or (d1.default < 0 and d2.default.is_integer()):
                if (d1.default != 0 or d2.default >= 0):
                    self.assertAlmostEqual(
                        (d1 ** d2).default, d1.default ** d2.default)
                    self.assertAlmostEqual(NumDict.__pow__(
                        d1, d2).default, (d1**d2).default)
                    self.assertAlmostEqual(NumDict.__pow__(
                        d1, d2).default, NumDict.__rpow__(d2, d1).default)
                else:
                    with self.assertRaises(ZeroDivisionError):
                        self.assertAlmostEqual(
                            (d1 ** d2).default, d1.default ** d2.default)
                        self.assertAlmostEqual(NumDict.__pow__(
                            d1, d2).default, (d1**d2).default)
                        self.assertAlmostEqual(NumDict.__pow__(
                            d1, d2).default, NumDict.__rpow__(d2, d1).default)

    def test_pow_and_rpow_basic_functionality_keys(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality for elements
            d1 = nd.NumDict({1: i, 2: (i+j)})
            d2 = nd.NumDict({1: j, 2: (i-j)})
            if (d1[1] >= 0 and d1[2] >= 0) or ((d1[1] < 0 and d2[1].is_integer()) and (d1[2] < 0 and d2[2].is_integer())):
                d3 = d1 ** d2
                self.assertAlmostEqual(d3[1], d1[1] ** d2[1])
                self.assertAlmostEqual(d3[2], d1[2] ** d2[2])
                self.assertAlmostEqual(NumDict.__pow__(d1, d2)[1], d3[1])
                self.assertAlmostEqual(NumDict.__pow__(d1, d2)[2], d3[2])
                self.assertAlmostEqual(NumDict.__pow__(d1, d2)[1],
                                       NumDict.__rpow__(d2, d1)[1])
                self.assertAlmostEqual(NumDict.__pow__(d1, d2)[2],
                                       NumDict.__rpow__(d2, d1)[2])

    def test_pow_and_rpow_basic_functionality_mixed(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality for elements
            d1 = nd.NumDict({1: i}, default=i)
            d2 = nd.NumDict({1: j, 2: j}, default=1)
            if (d1[1] >= 0 and d1.default > 0) or ((d1[1] < 0 and d2[1].is_integer()) and (d1.default < 0 and d2[2].is_integer())):
                d3 = d1 ** d2
                self.assertAlmostEqual(d3[1], d1[1] ** d2[1])
                self.assertAlmostEqual(d3[2], d1[2] ** d2[2])
                self.assertAlmostEqual(d3.default, d1.default**d2.default)
                self.assertAlmostEqual(
                    NumDict.__pow__(d1, d2)[1], d3[1])
                self.assertAlmostEqual(
                    NumDict.__pow__(d1, d2)[2], d3[2])
                self.assertAlmostEqual(NumDict.__pow__(d1, d2)[1],
                                       NumDict.__rpow__(d2, d1)[1])
                self.assertAlmostEqual(NumDict.__pow__(d1, d2)[2],
                                       NumDict.__rpow__(d2, d1)[2])

    def test_pow_and_rpow_basic_functionality_nodefault(self):
        for i, j in linspace(-10, 10):
            # testing basic functionality with mismatched keys
            d1 = nd.NumDict({1: i, 3: i}, default=i)
            d2 = nd.NumDict({1: j, 2: j})
            if (d1[1] >= 0 and d1.default > 0) or ((d1[1] < 0 and d2[1].is_integer()) and (d1.default < 0 and d2[2].is_integer())):
                with self.assertRaises(KeyError):
                    d1**d2

    def test_pow_and_rpow_differentiation(self):
        for i, j in linspace(-1, 10):
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for default with normal operator
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                if(d1.default >= 0) or (d1.default < 0 and d2.default.is_integer()):
                    if(d1.default != 0 and d2.default != 0):
                        d3 = d1 ** d2
            if(d1.default >= 0) or (d1.default < 0 and d2.default.is_integer()):
                if(d1.default != 0 and d2.default != 0):
                    d3, grads1 = tape.gradients(d3, (d1, d2))
                    self.assertAlmostEqual(
                        grads1[0].default, (d1.default**(d2.default-1))*d2.default)
                    if(d1.default < 0):
                        self.assertTrue(math.isnan(grads1[1].default))
                    else:
                        self.assertAlmostEqual(grads1[1].default, d1.default **
                                               d2.default*math.log(d1.default))
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for default with __pow__
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                if(d1.default >= 0) or (d1.default < 0 and d2.default.is_integer()):
                    if(d1.default != 0 and d2.default != 0):
                        d3 = NumDict.__pow__(d1, d2)
            if(d1.default >= 0) or (d1.default < 0 and d2.default.is_integer()):
                if(d1.default != 0 and d2.default != 0):
                    d3, grads2 = tape.gradients(d3, (d1, d2))
                    self.assertAlmostEqual(grads2[0].default,
                                           d1.default**(d2.default-1)*d2.default)
                    if(d1.default < 0):
                        self.assertTrue(math.isnan(grads2[1].default))
                    else:
                        self.assertAlmostEqual(grads2[1].default, d1.default **
                                               d2.default*math.log(d1.default))
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for default with __rpow__
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                if(d1.default >= 0) or (d1.default < 0 and d2.default.is_integer()):
                    if(d1.default != 0 and d2.default != 0):
                        d3 = NumDict.__rpow__(d2, d1)
            if(d1.default >= 0) or (d1.default < 0 and d2.default.is_integer()):
                if(d1.default != 0 and d2.default != 0):
                    d3, grads3 = tape.gradients(d3, (d1, d2))
                    self.assertAlmostEqual(grads3[0].default,
                                           d1.default**(d2.default-1)*d2.default)
                    if(d1.default < 0):
                        self.assertTrue(math.isnan(grads3[1].default))
                    else:
                        self.assertAlmostEqual(grads3[1].default, d1.default **
                                               d2.default*math.log(d1.default))
                    # verifying that all differentiation is equal to one another
            if(math.isnan(grads1[0].default) or math.isnan(grads1[0].default)
               or math.isnan(grads2[0].default) or math.isnan(grads2[0].default)
               or math.isnan(grads3[0].default) or math.isnan(grads3[1].default)):
                self.assertTrue(math.isnan(grads1[0].default) == math.isnan(
                    grads2[0].default) == math.isnan(grads3[0].default))
                self.assertTrue(math.isnan(grads1[1].default) == math.isnan(
                    grads2[1].default) == math.isnan(grads3[1].default))
            else:
                self.assertAlmostEqual(grads1[0].default, grads2[0].default)
                self.assertAlmostEqual(grads1[1].default, grads2[1].default)
                self.assertAlmostEqual(grads2[0].default, grads3[0].default)
                self.assertAlmostEqual(grads2[1].default, grads3[1].default)
            with tape:
                # testing differentiation for elements with normal operator
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                if (d1[1] >= 0 and d1[2] >= 0) or ((d1[1] < 0 and d2[1].is_integer()) and (d1[2] < 0 and d2[2].is_integer())):
                    d3 = d1 ** d2
            if (d1[1] > 0 and d1[2] > 0) or ((d1[1] < 0 and d2[1].is_integer()) and (d1[2] < 0 and d2[2].is_integer())):
                d3, grads = tape.gradients(d3, (d1, d2))
                self.assertAlmostEqual(
                    grads[0][1], (d1[1]**(d2[1]-1))*d2[1])
                if(d1[1] < 0):
                    self.assertTrue(math.isnan(grads[1][1]))
                else:
                    self.assertAlmostEqual(grads[1][1], d1[1] **
                                           d2[1]*math.log(d1[1]))
                self.assertAlmostEqual(
                    grads[0][2], (d1[2]**(d2[2]-1))*d2[2])
                if(d1[2] < 0):
                    self.assertTrue(math.isnan(grads[1][2]))
                else:
                    self.assertAlmostEqual(grads[1][2], d1[2] **
                                           d2[2]*math.log(d1[2]))


class TestNumdictsOpsThreshold(unittest.TestCase):
    def test_threshold_default(self):
        d = nd.NumDict(default=5)
        with GradientTape() as t:
            #d1 = sum_by(d,keyfunc=[1,2])
            d1 = threshold(d, th=3)
        self.assertAlmostEqual(d1.default, 5)
        d1, g = t.gradients(d1, d)
        self.assertAlmostEqual(g.default, 0)
        d = nd.NumDict(default=2)
        with GradientTape() as t:
            d1 = threshold(d, th=3)
        self.assertTrue(d1.default == None)

    def test_threshold_keys(self):
        d = nd.NumDict(data={1: 1, 2: 2, 3: 3, 4: 4, 5: 5})
        with GradientTape() as t:
            d1 = threshold(d, th=3)
        for i in range(5):
            if(i <= 3):
                self.assertTrue(d1.get(i) == None)
            else:
                self.assertAlmostEqual(d1[i], i)
        d1, g = t.gradients(d1, d)
        self.assertAlmostEqual(g.get(1), 0)
        self.assertAlmostEqual(g.get(2), 0)
        self.assertAlmostEqual(g.get(3), 0)
        self.assertAlmostEqual(g.get(4), 1)
        self.assertAlmostEqual(g.get(5), 1)

    def test_threshold_mixed(self):
        d = nd.NumDict(default=5, data={1: 1, 2: 2, 3: 3, 4: 4, 5: 5})
        with GradientTape() as t:
            d1 = threshold(d, th=3)
        self.assertAlmostEqual(d1.default, 5)
        for i in range(5):
            if(i <= 3):
                self.assertTrue(d1[i] == d1.default)
            else:
                self.assertAlmostEqual(d1[i], i)
        d = nd.NumDict(default=2, data={1: 1, 2: 2, 3: 3, 4: 4, 5: 5})
        with GradientTape() as t:
            d1 = threshold(d, th=3)
            d1 = d1*1
        self.assertTrue(d1.default == None)
        for i in range(5):
            if(i <= 3):
                self.assertTrue(d1.get(i) == None)
            else:
                self.assertAlmostEqual(d1[i], i)
        d1, g = t.gradients(d1, d)
        self.assertAlmostEqual(g.get(1), 0)
        self.assertAlmostEqual(g.get(2), 0)
        self.assertAlmostEqual(g.get(3), 0)
        self.assertAlmostEqual(g.get(4), 1)
        self.assertAlmostEqual(g.get(5), 1)


class TestNumdictsOpsClip(unittest.TestCase):
    def test_clip_keys(self):
        d = nd.NumDict(data={1: 1, 2: 2, 3: 3, 4: 4, 5: 5})
        with GradientTape() as t:
            d1 = clip(d, 2, 4)
        for i in range(1, 5):
            if(i < 2):
                self.assertAlmostEqual(d1[i], 2)
            elif(i > 4):
                self.assertAlmostEqual(d1[i], 4)
            else:
                self.assertAlmostEqual(d1[i], i)
        d1, g = t.gradients(d1, d)
        for i in range(1, 5):
            if(i <= 2):
                self.assertAlmostEqual(g[i], 0)
            elif(i >= 4):
                self.assertAlmostEqual(g[i], 0)
            else:
                self.assertAlmostEqual(g[i], 1)


class TestNumdictsOpsKeepDrop(unittest.TestCase):
    def dummyfunc(self, f):
        if(f % 2 == 0):
            return True
        else:
            return False

    def test_keep_keys(self):
        d = nd.NumDict(data={1: 1, 2: 2, 3: 3, 4: 4, 5: 5})
        testCollection = {1: True}
        with GradientTape() as t:
            d1 = keep(d, self.dummyfunc, testCollection)
        for i in range(1, 5):
            if(i % 2 == 0 or i == 1):
                self.assertAlmostEqual(d1[i], i)
            else:
                self.assertTrue(d1.get(i) == None)
        d1, g = t.gradients(d1, d)
        for i in range(1, 5):
            if(i % 2 == 0 or i == 1):
                self.assertAlmostEqual(g[i], 1)
            else:
                self.assertAlmostEqual(g.get(i), 0)

    def test_drop_keys(self):
        d = nd.NumDict(data={1: 1, 2: 2, 3: 3, 4: 4, 5: 5})
        testCollection = {1: True}
        with GradientTape() as t:
            d1 = drop(d, self.dummyfunc, testCollection)
        print(d1)  
        for i in range(1, 5):
            if(i % 2 == 0 or i == 1):
                self.assertTrue(d1.get(i) == None)
            else:
                self.assertAlmostEqual(d1[i], i)
        d1, g = t.gradients(d1, d)
        for i in range(1, 5):
            if(i % 2 == 0 or i == 1):
                self.assertAlmostEqual(g.get(i), 0)
            else:
                self.assertAlmostEqual(g[i], 1)


class TestNumdictsNested(unittest.TestCase):
    def test_persistent(self):
        for i, j in linspace(-10, 10):
            t1 = nd.GradientTape(persistent=True)
            with t1:
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d3 = d1*d1*d2*d2
            d4, g1 = t1.gradients(d3, (d1, d2), forward=False)
            self.assertAlmostEqual(
                g1[0].default, 2*d1.default*d2.default*d2.default)
            self.assertAlmostEqual(
                g1[1].default, 2*d2.default*d1.default*d1.default)
            d4, g1 = t1.gradients(d3, (d2, d1), forward=False)
            self.assertAlmostEqual(
                g1[1].default, 2*d1.default*d2.default*d2.default)
            self.assertAlmostEqual(
                g1[0].default, 2*d2.default*d1.default*d1.default)

    """def test_2nested_numdicts_defualts(self):#TODO fix this test
        for i, j in linspace(-10, 10):
            t1 = nd.GradientTape(persistent=True, allowNesting=True)
            t2 = nd.GradientTape(allowNesting=True)
            with t1:
                # testing differentiation for default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                with t2:
                    d3 = d1*d1*d2*d2
                d3, g1 = t2.gradients(
                    d3, (d1, d2), forward=False)  # first derivative
                self.assertAlmostEqual(
                    g1[0].default, 2*d2.default*d2.default*d1.default)
                self.assertAlmostEqual(
                    g1[1].default, 2*d1.default*d1.default*d2.default)
            # second derivative with respect to d1
            g3, g2 = t1.gradients(g1[0], (d1, d2), forward=False)
            x = nd.NumDict()
            y = nd.NumDict()

            with t1:
                with t2:
                    result = (x**2)*(y**2)
                result, g1 = t2.gradients(result, (x, y))  # first derivative
            # derivative of first derivative with respect to x
            result2, g2 = t1.gradients(g1[0], (x, y))
            # derivative of first derivative with respect to y
            result3, g3 = t1.gradients(g1[1], (x, y))

            self.assertAlmostEqual(g3);
            self.assertAlmostEqual(
                g1[0].default, 2*d2.default*d2.default*d1.default)
            self.assertAlmostEqual(
                g1[1].default, 2*d1.default*d1.default*d2.default)
            # second derivative with respect to d1
            g3, g2 = t1.gradients(g1[0], (d1, d2), forward=False)
            self.assertAlmostEqual(g2[0].default, 2*d2.default*d2.default)
            self.assertAlmostEqual(g2[1].default, 4*d2.default*d1.default)
            # second derivative with respect to d2
            # this is incorrect for some reason
            g3, g2 = t1.gradients(g1[1], (d1, d2), forward=False)
            self.assertAlmostEqual(g2[0].default, 2*d1.default*d1.default)
            self.assertAlmostEqual(g2[1].default, 4*d1.default*d2.default)"""

    def test_3nested_numdicts_defaults(self):
        for i, j in linspace(-10, 10):
            with GradientTape(allowNesting=True) as t1:
                # testing differentiation for default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d3 = nd.NumDict(default=i+j)
                with GradientTape(allowNesting=True) as t2:
                    with GradientTape(allowNesting=True) as t3:
                        d4 = d1*d2*d3
                    d4, g1 = t3.gradients(d4, (d1, d2, d3))
                    self.assertAlmostEqual(
                        g1[0].default, d3.default*d2.default)
                    self.assertAlmostEqual(
                        g1[1].default, d1.default*d3.default)
                    self.assertAlmostEqual(
                        g1[2].default, d1.default*d2.default)
                g1, g2 = t2.gradients(g1[0], (d1, d2, d3))
                self.assertAlmostEqual(g2[0].default, 0)
                self.assertAlmostEqual(g2[1].default, d3.default)
                self.assertAlmostEqual(g2[2].default, d2.default)
            g2, g3 = t1.gradients(g2[0], (d1, d2, d3))
            self.assertAlmostEqual(g3[0].default, 0)
            self.assertAlmostEqual(g3[1].default, 0)
            self.assertAlmostEqual(g3[2].default, 0)

    def test_2nested_numdicts_keys(self):
        for i, j in linspace(-10, 10):
            t1 = nd.GradientTape()
            t2 = nd.GradientTape(allowNesting=True)
            with t1:
                # testing differentiation for default
                d1 = nd.NumDict({1: i, 2: i+j})
                d2 = nd.NumDict({1: j, 2: j-i})
                with t2:
                    d3 = d1*d1+d2*d2
                d3, g1 = t2.gradients(d3, (d1, d2))
                self.assertAlmostEqual(g1[0][1], 2*d1[1])
                self.assertAlmostEqual(g1[1][1], 2*d2[1])
                self.assertAlmostEqual(g1[0][2], 2*d1[2])
                self.assertAlmostEqual(g1[1][2], 2*d2[2])
            g1, g2 = t1.gradients(g1[0], (d1, d2))
            self.assertAlmostEqual(g2[0][1], 2)
            self.assertAlmostEqual(g2[0][2], 2)
            self.assertAlmostEqual(g2[1][1], 0)
            self.assertAlmostEqual(g2[1][2], 0)

    def test_2nested_numdicts_mixed(self):
        for i, j in linspace(-10, 10):
            t1 = nd.GradientTape()
            t2 = nd.GradientTape(allowNesting=True)
            with t1:
                # testing differentiation for default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict({1: j, 2: j-i})
                with t2:
                    d3 = d1*d1+d2*d2
                d3, g1 = t2.gradients(d3, (d1, d2))
                self.assertAlmostEqual(g1[0][1], 2*d1.default)
                self.assertAlmostEqual(g1[1][1], 2*d2[1])
                self.assertAlmostEqual(g1[0][2], 2*d1.default)
                self.assertAlmostEqual(g1[1][2], 2*d2[2])
            g1, g2 = t1.gradients(g1[0], (d1, d2))
            self.assertAlmostEqual(g2[0][1], 2)
            self.assertAlmostEqual(g2[0][2], 2)
            self.assertAlmostEqual(g2[1][1], 0)
            self.assertAlmostEqual(g2[1][2], 0)

    """def test_3nested_numdicts_productRule(self): #TODO fix this test
        for i, j in linspace(-10, 10):
            t1 = nd.GradientTape()
            t2 = nd.GradientTape(allowNesting=True)
            t3 = nd.GradientTape(allowNesting=True)
            with t1:
                # testing differentiation for default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d3 = nd.NumDict(default=i+j)
                with t2:
                    with t3:
                        d4 = (d1+d2)*d3  # something go broke here not sure why
                        # d4 = d1*d3+d2*d3
                    d4, g1 = t3.gradients(d4, (d1, d2, d3))
                    self.assertAlmostEqual(g1[0].default, d3.default)
                    self.assertAlmostEqual(g1[1].default, d3.default)
                    self.assertAlmostEqual(
                        g1[2].default, d1.default+d2.default)
                g1, g2 = t2.gradients(g1[2], (d1, d2, d3))
                self.assertAlmostEqual(g2[0].default, 1)
                self.assertAlmostEqual(g2[1].default, 1)
    """
