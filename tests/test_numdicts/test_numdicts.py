import pyClarion.numdicts as nd

import unittest
import math

from pyClarion.numdicts.numdicts import NumDict


class TestNumdicts(unittest.TestCase):
    def sequenceGenerator(self, a, b):  # TODO fix/improve
        a = a*4
        b = b*4
        for i in range(a, b):
            for j in range(a, b):
                yield (i/4, j/4)

    def test_addition_basic_functionality(self):
        for i, j in self.sequenceGenerator(-10, 10):
            tape = nd.GradientTape()
            with tape:
                # testing basic functionality for defaults
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                self.assertEqual((d1+d2).default, d1.default+d2.default)
                d3 = nd.NumDict(default=3.0)
                self.assertEqual((d2+d1+d3).default, (d3+d1+d2).default)
                self.assertEqual((d3+d1+d2).default, (d1+d2+d3).default)
            with tape:
                # testing basic functionality for elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                d3 = d1+d2
                self.assertEqual(d3[1], d1[1]+d2[1])
                self.assertEqual(d3[2], d1[2]+d2[2])

    def test_addition_differentiation(self):
        for i, j in self.sequenceGenerator(-10, 10):
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for defaults
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d4 = d1 + d2
            d4, grads = tape.gradients(d4, (d1, d2))
            self.assertEqual(grads[0].default, 1.0)
            self.assertEqual(grads[1].default, 1.0)
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                d3 = d1+d2
            d3, grads = tape.gradients(d3, (d1, d2))
            self.assertEqual(grads[0][1], 1.0)
            self.assertEqual(grads[0][2], 1.0)
            self.assertEqual(grads[1][1], 1.0)
            self.assertEqual(grads[1][2], 1.0)

    def test_subtraction_basic_functionality(self):
        for i, j in self.sequenceGenerator(-10, 10):
            tape = nd.GradientTape()
            with tape:
                # testing basic functionality for default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                self.assertEqual((d1-d2).default, d1.default-d2.default)
                self.assertEqual((d2-d1).default, d2.default-d1.default)
            tape = nd.GradientTape()
            with tape:
                # testing basic functionality for elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                d3 = d1-d2
                self.assertEqual(d3[1], d1[1]-d2[1])
                self.assertEqual(d3[2], d1[2]-d2[2])

    def test_subtraction_differentiation(self):
        for i, j in self.sequenceGenerator(-10, 10):
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for defaults
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d4 = d1 - d2
            d4, grads = tape.gradients(d4, (d1, d2))
            self.assertEqual(grads[0].default, 1.0)
            self.assertEqual(grads[1].default, -1.0)
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                d3 = d1-d2
            d3, grads = tape.gradients(d3, (d1, d2))
            self.assertEqual(grads[0][1], 1.0)
            self.assertEqual(grads[0][2], 1.0)
            self.assertEqual(grads[1][1], -1.0)
            self.assertEqual(grads[1][2], -1.0)

    def test_truediv_and_rtruediv_basic_functionality(self):
        for i, j in self.sequenceGenerator(-10, 10):
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
                    self.assertEqual(
                        (d1/d2).default, d1.default/d2.default)
                    self.assertEqual(NumDict.__truediv__(
                        d1, d2).default, (d1/d2).default)
                    self.assertEqual(NumDict.__truediv__(
                        d1, d2).default, NumDict.__rtruediv__(d2, d1).default)
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
                    self.assertEqual(d3[1], d1[1]/d2[1])
                    self.assertEqual(d3[2], d1[2]/d2[2])
                    d3 = NumDict.__truediv__(d1, d2)
                    self.assertEqual(d3[1], (d1/d2)[1])
                    self.assertEqual(d3[2], (d1/d2)[2])
                    self.assertEqual(NumDict.__truediv__(
                        d1, d2), NumDict.__rtruediv__(d2, d1))

    def test_truediv_and_rtruediv_differentiation(self):
        for i, j in self.sequenceGenerator(-10, 10):
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for truediv and default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                if(d2.default != 0):
                    d3 = NumDict.__truediv__(d1, d2)
            if(d2.default != 0):
                d3, grads = tape.gradients(d3, (d1, d2))
                self.assertEqual(grads[0].default, 1/d2.default)
                self.assertEqual(grads[1].default,
                                 (-d1.default)/(d2.default**2))
            with tape:
                # testing differentiation for rtruediv and default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                if(d1.default != 0):
                    d4 = NumDict.__rtruediv__(d1, d2)
            if(d1.default != 0):
                d4, grads = tape.gradients(d4, (d1, d2))
                self.assertEqual(grads[0].default,
                                 (-d2.default)/(d1.default**2))
                self.assertEqual(grads[1].default, 1/d1.default)
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for truediv and elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                if(d2[1]*d2[2] != 0):
                    d3 = NumDict.__truediv__(d1, d2)
            if(d2[1]*d2[2] != 0):
                d3, grads = tape.gradients(d3, (d1, d2))
                self.assertEqual(grads[0][1], 1/d2[1])
                self.assertEqual(grads[0][2], 1/d2[2])
                self.assertEqual(grads[1][1], (-d1[1])/(d2[1]**2))
                self.assertEqual(grads[1][2], (-d1[2])/(d2[2]**2))
            tape = nd.GradientTape()
            with tape:
                # testing differentiation for rtruediv and elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                if(d1[1]*d1[2] != 0):
                    d3 = NumDict.__rtruediv__(d1, d2)
            if(d1[1]*d1[2] != 0):
                d3, grads = tape.gradients(d3, (d1, d2))
                self.assertEqual(grads[0][1], (-d2[1])/(d1[1]**2))
                self.assertEqual(grads[0][2], (-d2[2])/(d1[2]**2))
                self.assertEqual(grads[1][1], 1/d1[1])
                self.assertEqual(grads[1][2], 1/d1[2])

    def test_pow_and_rpow(self):  # TODO FIX/DEBUG
        """for i in range(-40, 40):
            for j in range(-40, 40):
                tape = nd.GradientTape()
                with tape:
                    # testing basic functionality
                    d1 = nd.NumDict(default=i)
                    d2 = nd.NumDict(default=j)
                    self.assertEqual(
                        (d1 ** d2).default, d1.default ** d2.default)
                    self.assertEqual(NumDict.__pow__(
                        d1, d2).default, (d1**d2).default)
                    self.assertEqual(NumDict.__pow__(
                        d1, d2).default, NumDict.__rpow__(d2, d1).default)
                    d3 = NumDict.__pow__(d1, d2)
                    d4 = NumDict.__rpow__(d2, d1)
                # testing pow and rpow differentiation
                d3, grads1 = tape.gradients(d3, (d1, d2))
                print(i)
                print(j)
                print((math.log(abs(d1.default)))*d1.default ** d2.default)
                self.assertEqual(
                    grads1[0].default, (d2.default)*(d1.default) ** (d2.default-1))
                self.assertEqual(grads1[1].default,
                                 (math.log(abs(d1.default)))*d1.default ** d2.default)
                d4, grads2 = tape.gradients(d4, (d1, d2))
                self.assertEqual(
                    grads2[0].default, (d2.default)*(d1.default) ** (d2.default-1))
                self.assertEqual(grads1[1].default, (math.log(
                    abs(d1.default)))*d1.default ** d2.default)
                self.assertEqual(grads1[0], grads2[0])
                self.assertEqual(grads1[1], grads2[1])
                 tape = nd.GradientTape()
                with tape:
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
                        self.assertEqual(d3[1], d1[1]/d2[1])
                        self.assertEqual(d3[2], d1[2]/d2[2])
                        d3 = NumDict.__truediv__(d1, d2)
                        self.assertEqual(d3[1], (d1/d2)[1])
                        self.assertEqual(d3[2], (d1/d2)[2])
                        self.assertEqual(NumDict.__truediv__(
                            d1, d2), NumDict.__rtruediv__(d2, d1))
                    if(d2[1]*d2[2] != 0):
                        d3 = NumDict.__truediv__(d1, d2)
                if(d2[1]*d2[2] != 0):
                    d3, grads = tape.gradients(d3, (d1, d2))
                    self.assertEqual(grads[0][1], 1/d2[1])
                    self.assertEqual(grads[0][2], 1/d2[2])
                    self.assertEqual(grads[1][1], (-d1[1])/(d2[1]**2))
                    self.assertEqual(grads[1][2], (-d1[2])/(d2[2]**2))
                tape = nd.GradientTape()
                with tape:
                    d1 = nd.NumDict({1: i, 2: (i+j)})
                    d2 = nd.NumDict({1: j, 2: (i-j)})
                    if(d1[1]*d1[2] != 0):
                        d3 = NumDict.__rtruediv__(d1, d2)
                if(d1[1]*d1[2] != 0):
                    d3, grads = tape.gradients(d3, (d1, d2))
                    self.assertEqual(grads[0][1], (-d2[1])/(d1[1]**2))
                    self.assertEqual(grads[0][2], (-d2[2])/(d1[2]**2))
                    self.assertEqual(grads[1][1], 1/d1[1])
                    self.assertEqual(grads[1][2], 1/d1[2]) """

    def test_multiplication_basic_functionality(self):
        for i, j in self.sequenceGenerator(-10, 10):
            tape = nd.GradientTape()
            with tape:
                # testing basic functionality for default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                self.assertEqual((d1*d2).default, d1.default*d2.default)
                # testing multiplying by 0
                d0 = nd.NumDict(default=0.0)
                self.assertEqual((d2*d0).default, 0.0)
                self.assertEqual((d1*d0).default, 0.0)
                # testing communative
                d3 = nd.NumDict(default=3.0)
                self.assertEqual((d2*d1*d3).default, (d3*d1*d2).default)
                self.assertEqual((d1*d2*d3).default, (d1*d2*d3).default)
            tape = nd.GradientTape()
            with tape:
                # testing basic functionality for elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                d3 = d1*d2
                self.assertEqual(d3[1], d1[1]*d2[1])
                self.assertEqual(d3[2], d1[2]*d2[2])

    def test_multiplication_differentiation(self):
        for i, j in self.sequenceGenerator(-10, 10):
            tape = nd.GradientTape()
            with tape:
                #testing differentiation for default
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d3 = d1*d2
            d3, grads = tape.gradients(d3, (d1, d2))
            self.assertEqual(grads[0].default, d2.default)
            self.assertEqual(grads[1].default, d1.default)
            tape = nd.GradientTape()
            with tape:
                #testing differentiation for multiple NumDicts
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d3 = d1*d1*d2
            d3, grads = tape.gradients(d3, (d1, d2))
            self.assertEqual(grads[0].default, d2.default*d1.default*2)
            self.assertEqual(grads[1].default, d1.default*d1.default)
            tape = nd.GradientTape()
            with tape:
                #testing differentiation with non NumDict elements
                d1 = nd.NumDict(default=i)
                d2 = nd.NumDict(default=j)
                d3 = d1*d2
                d4 = i*d3
            d3, grads = tape.gradients(d4, (d1, d2))
            self.assertEqual(grads[0].default, i*d2.default)
            self.assertEqual(grads[1].default, i*d1.default)
            tape = nd.GradientTape()
            with tape:
                #testing differentiation for elements
                d1 = nd.NumDict({1: i, 2: (i+j)})
                d2 = nd.NumDict({1: j, 2: (i-j)})
                d3 = d1*d2
            d3, grads = tape.gradients(d3, (d1, d2))
            self.assertEqual(grads[0][1], d2[1])
            self.assertEqual(grads[0][2], d2[2])
            self.assertEqual(grads[1][1], d1[1])
            self.assertEqual(grads[1][2], d1[2])
