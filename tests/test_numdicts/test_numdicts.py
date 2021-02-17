import pyClarion.numdicts as nd

import unittest
import math

class TestNumdicts(unittest.TestCase):
    def test_multiplication(self):
        #testing basic functionality
        for i in range(-40,40):
            for j in range(-40,40):
                tape = nd.GradientTape()
                with tape:
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
        #testing differentiation
        for i in range(-40,40):
                for j in range(-40,40):
                    tape = nd.GradientTape()
                    with tape:
                        d1 = nd.NumDict(default=i/4)
                        d2 = nd.NumDict(default=j/4)
                        d3 = d1*d2
                    d3, grads = tape.gradients(d3,(d1,d2))
                    assert(grads[0].default == d2.default)
                    assert(grads[1].default == d1.default)
                    tape = nd.GradientTape()
                    with tape:
                        d1 = nd.NumDict(default=i/4)
                        d2 = nd.NumDict(default=j/4)
                        d3 = d1*d1*d2
                    d3, grads = tape.gradients(d3,(d1,d2))
                    assert(grads[0].default == d2.default*d1.default*2)
                    assert(grads[1].default == d1.default*d1.default)    
                    tape = nd.GradientTape()
                    with tape:
                        d1 = nd.NumDict(default=i/4)
                        d2 = nd.NumDict(default=j/4)
                        d3 = d1*d2
                        d4 = i*d3
                    d3, grads = tape.gradients(d4,(d1,d2))
                    assert(grads[0].default == i*d2.default)    
                    assert(grads[1].default == i*d1.default)

    def test_addition(self):
        for i in range(-40,40):
            for j in range(-40,40):
                tape = nd.GradientTape()
                with tape:
                    #testing basic functionality
                    d1 = nd.NumDict(default=i/4)
                    d2 = nd.NumDict(default=j/4)
                    assert((d1+d2).default == d1.default+d2.default)
                    d3 = nd.NumDict(default=3.0)
                    assert((d2+d1+d3).default == (d3+d1+d2).default == (d1+d2+d3).default)
                    #testing differentiation
                    d4 = d1 + d2
                d4, grads = tape.gradients(d4,(d1,d2))
                assert(grads[0].default == 1.0)
                assert(grads[1].default == 1.0)