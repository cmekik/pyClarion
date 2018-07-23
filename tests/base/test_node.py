import unittest
from pyClarion.base.node import Microfeature, Chunk
from enum import auto

class ChunkTestCase(unittest.TestCase):

    def setUp(self):

        self.white = Microfeature("Color", "White")
        self.black = Microfeature("Color", "Black")
        self.square = Microfeature("Shape", "Square")
        self.circle = Microfeature("Shape", "Circle")

        self.white_square = Chunk(
            microfeatures = {self.white, self.square},
            dim2weight = {"Color" : 1, "Shape" : 1},
            label="white square" 
        )

        self.black_or_white_square = Chunk(
            microfeatures = {
                self.black, 
                self.white, 
                self.square
            }, 
            dim2weight = {"Color" : 1, "Shape" : 1}
        )

    def test_repr(self):

        enclosing_str = ["<Chunk: 'white square' ", ">"]
        self.assertEqual(
            self.white_square.__repr__(),
            str(self.white_square.microfeatures).join(enclosing_str)
        )