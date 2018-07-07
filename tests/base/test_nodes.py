import unittest
from pyClarion.base.nodes import Feature, Chunk
from enum import auto

class ChunkTestCase(unittest.TestCase):

    class Color(Feature):
        WHITE = auto()
        BLACK = auto()

    class Shape(Feature):
        SQUARE = auto()
        CIRCLE = auto()

    def setUp(self):

        self.white_square = Chunk(
            microfeatures = {self.Color.WHITE, self.Shape.SQUARE},
            label="white square" 
        )

        self.black_or_white_square = Chunk(
            microfeatures = {
                self.Color.BLACK, 
                self.Color.WHITE, 
                self.Shape.SQUARE
            }, 
        )

    def test_repr(self):

        enclosing_str = ["<Chunk: 'white square' ", ">"]
        self.assertEqual(
            self.white_square.__repr__(),
            str(self.white_square.microfeatures).join(enclosing_str)
        )

    def test_initialize_weights(self):
        
        self.assertEqual(
            self.white_square.dim2weight,
            {self.Color : 1., self.Shape : 1.}
        )