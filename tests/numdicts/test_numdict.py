import unittest
from uuid import uuid4
from itertools import product

from pyClarion.numdicts import Key, KeyForm, numdict, Index
from pyClarion.numdicts.keyspaces import KSRoot, KSNode

@unittest.skip("very broken")
class NumDictTestCase(unittest.TestCase):
    
    def setUp(self):
        ksp = KeySpace()
        f, c = ksp.f, ksp.c
        f.a.b; f.a.c; f.a.d; f.e.g; f.e.h
        c.c1; c.c2; c.c3
        self.i_f = Index(ksp, KeyForm(Key("f"), (2,)))
        self.i_c = Index(ksp, KeyForm(Key("c"), (1,)))
        self.i_w = Index(ksp, KeyForm(Key("(c,f)"), (1,2)))

    def test_numdict(self):
        ...
        d1 = numdict(self.i_f, {"f:a:b": 1.0, "f:a:c": -3.0}, 0.0)
        d2 = numdict(self.i_f, {}, 1.0)
        d3 = d1.sum(d2)
        # ...

if __name__ == "__main__":
    unittest.main()
