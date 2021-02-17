import pyClarion.numdicts as nd

import unittest

class TestNumdicts(unittest.TestCase):
    def test_basic_multiplication(self):
        tape = nd.GradientTape()
        with tape:
            #testing basic functionality
            for i in range(-40,40):
                for j in range(-40,40):
                    d1 = nd.NumDict(default=i/4)
                    d2 = nd.NumDict(default=j/4)
                    assert((d1*d2).default == d1.default*d2.default)
                    #testing multiplying by 0
                    d0 = nd.NumDict(default=0.0)
                    assert((d2*d0).default == 0.0)
                    assert((d1*d0).default == 0.0)
                    #testing communative
                    d3 = nd.NumDict(default=3.0)
                    assert((d2*d1*d3).default == (d3*d1*d2).default == (d1*d2*d3).default)
    def test_addition(self):
        tape = nd.GradientTape()
        with tape:
            #testing basic functionality
            for i in range(-40,40):
                for j in range(-40,40):
                    d1 = nd.NumDict(default=i/4)
                    d2 = nd.NumDict(default=j/4)
                    assert((d1+d2).default == d1.default+d2.default)
                    d3 = nd.NumDict(default=3.0)
                    assert((d2+d1+d3).default == (d3+d1+d2).default == (d1+d2+d3).default)