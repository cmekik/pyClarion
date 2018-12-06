import unittest
from pyClarion.base.symbols import *


class ConstructSymbolTest(unittest.TestCase):

    def test_unhashable_args(self):
        """ConstructSymbols with unhashable values should be unhashable."""

        with self.subTest(i="unhashable ctype"):
            with self.assertRaises(TypeError):
                hash(ConstructSymbol([1, 2, 3], 1))
        
        with self.subTest(i="unhashable cid"):
            with self.assertRaises(TypeError):
                hash(ConstructSymbol(1, [1, 2, 3]))

        with self.subTest(i="unhashable ctype, cid"):
            with self.assertRaises(TypeError):
                hash(ConstructSymbol([1, 2, 3], [1, 2, 3]))

    def test_str(self):
        """Str should be str(ctype)(repr(cid)), for prettiness."""

        # Str is used for ctype, but not cid, for pretty printing 
        csym = ConstructSymbol("Chunk", "APPLE")
        self.assertEqual(str(csym), "Chunk('APPLE')")


class ConstructTypeTest(unittest.TestCase):

    def test_str(self):
        """Str should pretty print for named flag states, otherwise use repr."""

        with self.subTest(i="named state"):
            ctype = ConstructType.Node
            s = str(ctype)
            self.assertEqual(s, ctype.name)

        with self.subTest(i="unnamed state"):
            ctype = ConstructType.Chunk | ConstructType.Subsystem
            s = str(ctype)
            self.assertEqual(s, repr(ctype))
