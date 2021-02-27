import pyClarion.numdicts as nd

import unittest
import math

from pyClarion.numdicts.numdicts import NumDict


class TestNumdicts(unittest.TestCase):
    def test_multiplication(self):
        # testing basic functionality
        for i in range(-40, 40):
            for j in range(-40, 40):
                tape = nd.GradientTape()
                with tape:
                    d1 = nd.NumDict(default=i/4)
                    d2 = nd.NumDict(default=j/4)
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
                    d1 = nd.NumDict({1: i/4, 2: (i+j)/4})
                    d2 = nd.NumDict({1: j/4, 2: (i-j)/4})
                    d3 = d1*d2
                    self.assertEqual(d3[1], d1[1]*d2[1])
                    self.assertEqual(d3[2], d1[2]*d2[2])
        # testing differentiation
        for i in range(-40, 40):
            for j in range(-40, 40):
                tape = nd.GradientTape()
                with tape:
                    d1 = nd.NumDict(default=i/4)
                    d2 = nd.NumDict(default=j/4)
                    d3 = d1*d2
                d3, grads = tape.gradients(d3, (d1, d2))
                self.assertEqual(grads[0].default, d2.default)
                self.assertEqual(grads[1].default, d1.default)
                tape = nd.GradientTape()
                with tape:
                    d1 = nd.NumDict(default=i/4)
                    d2 = nd.NumDict(default=j/4)
                    d3 = d1*d1*d2
                d3, grads = tape.gradients(d3, (d1, d2))
                self.assertEqual(grads[0].default, d2.default*d1.default*2)
                self.assertEqual(grads[1].default, d1.default*d1.default)
                tape = nd.GradientTape()
                with tape:
                    d1 = nd.NumDict(default=i/4)
                    d2 = nd.NumDict(default=j/4)
                    d3 = d1*d2
                    d4 = i*d3
                d3, grads = tape.gradients(d4, (d1, d2))
                self.assertEqual(grads[0].default, i*d2.default)
                self.assertEqual(grads[1].default, i*d1.default)
                tape = nd.GradientTape()
                with tape:
                    d1 = nd.NumDict({1: i/4, 2: (i+j)/4})
                    d2 = nd.NumDict({1: j/4, 2: (i-j)/4})
                    d3 = d1*d2
                d3, grads = tape.gradients(d3, (d1, d2))
                self.assertEqual(grads[0][1], d2[1])
                self.assertEqual(grads[0][2], d2[2])
                self.assertEqual(grads[1][1], d1[1])
                self.assertEqual(grads[1][2], d1[2])

    def test_addition(self):
        for i in range(-40, 40):
            for j in range(-40, 40):
                tape = nd.GradientTape()
                with tape:
                    # testing basic functionality
                    d1 = nd.NumDict(default=i/4)
                    d2 = nd.NumDict(default=j/4)
                    self.assertEqual((d1+d2).default, d1.default+d2.default)
                    d3 = nd.NumDict(default=3.0)
                    self.assertEqual((d2+d1+d3).default, (d3+d1+d2).default)
                    self.assertEqual((d3+d1+d2).default, (d1+d2+d3).default)
                    # testing differentiation
                    d4 = d1 + d2
                d4, grads = tape.gradients(d4, (d1, d2))
                self.assertEqual(grads[0].default, 1.0)
                self.assertEqual(grads[1].default, 1.0)
                tape = nd.GradientTape()
                with tape:
                    d1 = nd.NumDict({1: i/4, 2: (i+j)/4})
                    d2 = nd.NumDict({1: j/4, 2: (i-j)/4})
                    d3 = d1+d2
                    self.assertEqual(d3[1], d1[1]+d2[1])
                    self.assertEqual(d3[2], d1[2]+d2[2])
                d3, grads = tape.gradients(d3, (d1, d2))
                self.assertEqual(grads[0][1], 1.0)
                self.assertEqual(grads[0][2], 1.0)
                self.assertEqual(grads[1][1], 1.0)
                self.assertEqual(grads[1][2], 1.0)

    def test_subtraction(self):
        for i in range(-40, 40):
            for j in range(-40, 40):
                tape = nd.GradientTape()
                with tape:
                    # testing basic functionality
                    d1 = nd.NumDict(default=i/4)
                    d2 = nd.NumDict(default=j/4)
                    self.assertEqual((d1-d2).default, d1.default-d2.default)
                    self.assertEqual((d2-d1).default, d2.default-d1.default)
                    # testing differentiation
                    d4 = d1 - d2
                d4, grads = tape.gradients(d4, (d1, d2))
                self.assertEqual(grads[0].default, 1.0)
                self.assertEqual(grads[1].default, -1.0)
                tape = nd.GradientTape()
                with tape:
                    d1 = nd.NumDict({1: i/4, 2: (i+j)/4})
                    d2 = nd.NumDict({1: j/4, 2: (i-j)/4})
                    d3 = d1-d2
                    self.assertEqual(d3[1], d1[1]-d2[1])
                    self.assertEqual(d3[2], d1[2]-d2[2])
                d3, grads = tape.gradients(d3, (d1, d2))
                self.assertEqual(grads[0][1], 1.0)
                self.assertEqual(grads[0][2], 1.0)
                self.assertEqual(grads[1][1], -1.0)
                self.assertEqual(grads[1][2], -1.0)

    def test_truediv_and_rtruediv(self):
        for i in range(-40, 40):
            for j in range(-40, 40):
                tape = nd.GradientTape()
                with tape:
                    # testing basic functionality
                    d1 = nd.NumDict(default=i/4)
                    d2 = nd.NumDict(default=j/4)
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
                    if(d2.default != 0):
                        d3 = NumDict.__truediv__(d1, d2)
                # testing differentiation for truediv
                if(d2.default != 0):
                    d3, grads = tape.gradients(d3, (d1, d2))
                    self.assertEqual(grads[0].default, 1/d2.default)
                    self.assertEqual(grads[1].default,
                                     (-d1.default)/(d2.default**2))
                with tape:
                    d1 = nd.NumDict(default=i/4)
                    d2 = nd.NumDict(default=j/4)
                    if(d1.default*d2.default != 0):
                        self.assertEqual(
                            (d1/d2).default, d1.default/d2.default)
                        self.assertEqual(NumDict.__truediv__(
                            d1, d2).default, (d1/d2).default)
                        self.assertEqual(NumDict.__truediv__(
                            d1, d2).default, NumDict.__rtruediv__(d2, d1).default)
                    if(d1.default != 0):
                        d4 = NumDict.__rtruediv__(d1, d2)
                # testing differentiation for rtruediv
                if(d1.default != 0):
                    d4, grads = tape.gradients(d4, (d1, d2))
                    self.assertEqual(grads[0].default,
                                     (-d2.default)/(d1.default**2))
                    self.assertEqual(grads[1].default, 1/d1.default)
                tape = nd.GradientTape()
                with tape:
                    d1 = nd.NumDict({1: i/4, 2: (i+j)/4})
                    d2 = nd.NumDict({1: j/4, 2: (i-j)/4})
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
                    d1 = nd.NumDict({1: i/4, 2: (i+j)/4})
                    d2 = nd.NumDict({1: j/4, 2: (i-j)/4})
                    if(d1[1]*d1[2] != 0):
                        d3 = NumDict.__rtruediv__(d1, d2)
                if(d1[1]*d1[2] != 0):
                    d3, grads = tape.gradients(d3, (d1, d2))
                    self.assertEqual(grads[0][1], (-d2[1])/(d1[1]**2))
                    self.assertEqual(grads[0][2], (-d2[2])/(d1[2]**2))
                    self.assertEqual(grads[1][1], 1/d1[1])
                    self.assertEqual(grads[1][2], 1/d1[2])
