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

    def test_repr(self):
        """Str should be str(ctype)(repr(cid)), for prettiness."""

        csym = ConstructSymbol(ConstructType.Chunk, "APPLE")
        self.assertEqual(repr(csym), "<ConstructSymbol: Chunk('APPLE')>")

    def test_str(self):
        """Str should be str(ctype)(repr(cid)), for prettiness."""

        csym = ConstructSymbol(ConstructType.Chunk, "APPLE")
        self.assertEqual(str(csym), "Chunk('APPLE')")


class ConstructFactoryTest(unittest.TestCase):
    """Sanity checks for construct symbol factories."""

    def assertCtypeIs(self, csym, ctype):
        """Assert csym.ctype is ctype."""

        self.assertIs(csym.ctype, ctype)

    def assertCidTypeIs(self, csym, cidtype):
        """Assert type(csym.cid) is cidtype"""

        self.assertIs(type(csym.cid), cidtype)

    def setUp(self):

        self.data = [
            (
                Feature, 
                ("dim", "val"), 
                ConstructType.Feature, 
                DVPair
            ),
            (
                Chunk,
                ("name",),
                ConstructType.Chunk,
                None
            ),
            (
                Flow, 
                ("name", FlowType.TT), 
                ConstructType.Flow, 
                FlowID
            ),
            (
                Response, 
                ("name", ConstructType.Chunk), 
                ConstructType.Response, 
                ResponseID
            ),
            (
                Behavior,
                ("name", Response("name", ConstructType.Chunk)),
                ConstructType.Behavior,
                BehaviorID
            ),
            (
                Buffer,
                ("name", (Subsystem("name1"), Subsystem("name2"))),
                ConstructType.Buffer,
                BufferID
            ),
            (
                Subsystem,
                ("name",),
                ConstructType.Subsystem,
                None
            ),
            (
                Agent,
                ("name",),
                ConstructType.Agent,
                None
            )
        ]

    def test_factories(self):

        for factory, params, ctype, cid_type in self.data:
            with self.subTest(i = "{} factory".format(factory.__name__)):
                csym = factory(*params)
                self.assertCtypeIs(csym, ctype)
                if cid_type is not None:
                    self.assertCidTypeIs(csym, cid_type)
