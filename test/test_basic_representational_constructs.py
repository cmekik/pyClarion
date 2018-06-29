import basic_representational_constructs as brc
from enum import auto
import unittest

class ChunkTestCase(unittest.TestCase):

    class Color(brc.Feature):
        WHITE = auto()
        BLACK = auto()

    class Shape(brc.Feature):
        SQUARE = auto()
        CIRCLE = auto()

    def setUp(self):

        self.white_square = brc.Chunk(
            microfeatures = {self.Color.WHITE, self.Shape.SQUARE},
            top_down_weights = {self.Color: 1., self.Shape: 1.},
            label="white square" 
        )

        self.black_or_white_square = brc.Chunk(
            microfeatures = {
                self.Color.BLACK, 
                self.Color.WHITE, 
                self.Shape.SQUARE
            },
            top_down_weights = {self.Color: 1., self.Shape: 1.} 
        )

        self.sensory_input_1 = {
            self.Color.BLACK : 1.,
            self.Color.WHITE : 0.,
            self.Shape.SQUARE : 1.,
            self.Shape.CIRCLE : 0.
        }

        self.sensory_input_2 = {
            self.Color.BLACK : 0.,
            self.Color.WHITE : 1.,
            self.Shape.SQUARE : 1.,
            self.Shape.CIRCLE : 0.
        }

        self.sensory_input_3 = {
            self.Color.BLACK : 1.,
            self.Color.WHITE : 0.,
            self.Shape.SQUARE : 0.,
            self.Shape.CIRCLE : 1.
        }

    def test_top_down(self):
        
        with self.subTest(msg="white-square"):
            result = self.white_square.top_down(1.)
            expected = {self.Color.WHITE: 1., self.Shape.SQUARE: 1.}
            self.assertEqual(result, expected)

        with self.subTest(msg="white-square-partial"):
            result = self.white_square.top_down(.5)
            expected = {self.Color.WHITE: .5, self.Shape.SQUARE: .5}
            self.assertEqual(result, expected)

        with self.subTest(msg="black-or-white-square"):
            result = self.black_or_white_square.top_down(1.)
            expected = {
                self.Color.WHITE: .5, 
                self.Color.BLACK: .5, 
                self.Shape.SQUARE: 1.
            }
            self.assertEqual(result, expected)

    def test_bottom_up(self):

        with self.subTest(msg="si_1-white-square"):
            result = self.white_square.bottom_up(self.sensory_input_1)
            expected = 1./(2.**1.1)
            self.assertAlmostEqual(result, expected)

        with self.subTest(msg="si_2-white-square"):
            result = self.white_square.bottom_up(self.sensory_input_2)
            expected = 2./(2.**1.1)
            self.assertAlmostEqual(result, expected)

        with self.subTest(msg="si_1-black-or-white-square"):
            result = self.black_or_white_square.bottom_up(self.sensory_input_1)
            expected = 2./(2.**1.1)
            self.assertAlmostEqual(result, expected)

        with self.subTest(msg="si_1-black-or-white-square"):
            result = self.black_or_white_square.bottom_up(self.sensory_input_3)
            expected = 1./(2.**1.1)
            self.assertAlmostEqual(result, expected)