import unittest
import unittest.mock
from enum import auto
from pyClarion.base.node import Feature, Chunk
from pyClarion.default.common import TopDown, BottomUp, Rule, BLA

class TestTopDown(unittest.TestCase):

    def test_call(self):
        
        class Color(Feature):
            WHITE = auto()
            BLACK = auto()

        class Shape(Feature):
            SQUARE = auto()
            CIRCLE = auto()

        chunk1 = Chunk(
            microfeatures = {Color.WHITE, Shape.SQUARE},
        )

        chunk2 = Chunk(
            microfeatures = {Color.BLACK, Color.WHITE, Shape.SQUARE}, 
        )

        td1 = TopDown(chunk1)
        td2 = TopDown(chunk2)

        with self.subTest(msg="white-square"):
            result = td1({chunk1 : 1.})
            expected = {Color.WHITE: 1., Shape.SQUARE: 1.}
            self.assertEqual(result, expected)

        with self.subTest(msg="white-square-partial"):
            result = td1({chunk1 : .5})
            expected = {Color.WHITE: .5, Shape.SQUARE: .5}
            self.assertEqual(result, expected)

        with self.subTest(msg="black-or-white-square"):
            result = td2({chunk2 : 1.})
            expected = {
                Color.WHITE: .5, 
                Color.BLACK: .5, 
                Shape.SQUARE: 1.
            }
            self.assertEqual(result, expected)

class TestBottomUp(unittest.TestCase):

    def test_bottom_up(self):

        class Color(Feature):
            WHITE = auto()
            BLACK = auto()

        class Shape(Feature):
            SQUARE = auto()
            CIRCLE = auto()

        chunk1 = Chunk(
            microfeatures = {Color.WHITE, Shape.SQUARE},
        )

        chunk2 = Chunk(
            microfeatures = {Color.BLACK, Color.WHITE, Shape.SQUARE}, 
        )

        bu1 = BottomUp(chunk1)
        bu2 = BottomUp(chunk2)

        sensory_input_1 = {
            Color.BLACK : 1.,
            Color.WHITE : 0.,
            Shape.SQUARE : 1.,
            Shape.CIRCLE : 0.
        }

        sensory_input_2 = {
            Color.BLACK : 0.,
            Color.WHITE : 1.,
            Shape.SQUARE : 1.,
            Shape.CIRCLE : 0.
        }

        sensory_input_3 = {
            Color.BLACK : 1.,
            Color.WHITE : 0.,
            Shape.SQUARE : 0.,
            Shape.CIRCLE : 1.
        }

        with self.subTest(msg="si_1-white-square"):
            result = bu1(sensory_input_1)
            expected = 1./(2.**1.1)
            self.assertAlmostEqual(result[chunk1], expected)

        with self.subTest(msg="si_2-white-square"):
            result = bu1(sensory_input_2)
            expected = 2./(2.**1.1)
            self.assertAlmostEqual(result[chunk1], expected)

        with self.subTest(msg="si_1-black-or-white-square"):
            result = bu2(sensory_input_1)
            expected = 2./(2.**1.1)
            self.assertAlmostEqual(result[chunk2], expected)

        with self.subTest(msg="si_1-black-or-white-square"):
            result = bu2(sensory_input_3)
            expected = 1./(2.**1.1)
            self.assertAlmostEqual(result[chunk2], expected)    

class TestRule(unittest.TestCase):

    def setUp(self):
        
        self.mock_chunk_1 = unittest.mock.Mock(spec=Chunk)
        self.mock_chunk_2 = unittest.mock.Mock(spec=Chunk)
        self.mock_chunk_3 = unittest.mock.Mock(spec=Chunk)

        self.chunk2weight = {
            self.mock_chunk_1 : .5,
            self.mock_chunk_2 : .5
        }        

        self.rule = Rule(self.chunk2weight, self.mock_chunk_3)

    def test_apply(self):

        chunk2strengths = [
            {
                self.mock_chunk_1 : 0.,
                self.mock_chunk_2 : 0.
            },            
            {
                self.mock_chunk_1 : 0.,
                self.mock_chunk_2 : 1.
            },            
            {
                self.mock_chunk_1 : 0.,
                self.mock_chunk_2 : 1.,
                self.mock_chunk_3 : 1.
            },            
            {
                self.mock_chunk_1 : .5,
                self.mock_chunk_2 : 1.
            },
            {
                self.mock_chunk_1 : 1.,
                self.mock_chunk_2 : 1.
            }                        
        ]
        expected = [0, .5, .5, .75, 1.]

        for i, (c2s, e) in enumerate(zip(chunk2strengths, expected)):
            with self.subTest(i=i):
                conc2strength = self.rule(c2s)
                self.assertAlmostEqual(conc2strength[self.mock_chunk_3], e)

class TestBLA(unittest.TestCase):
    
    def setUp(self):

        self.bla = BLA()

        # Initialize BLA at time = 0
        self.bla.update(0.)

    def test_update(self):

        self.assertEqual(self.bla.timestamps[0], 0.)

    def test_compute_bla(self):

        test_times = [1.5, 2., 15.]
        expected = [ 
            2. * (1.5 ** -.5),
            2. * ((2. ** -.5) + ((2. - 1.5) ** -.5)),
            2. * ((15. ** -.5) + ((15. - 1.5) ** -.5) + ((15. - 2.) ** -.5))
        ]

        for i, (t, e) in enumerate(zip(test_times, expected)):
            with self.subTest(i=i):
                self.assertAlmostEqual(self.bla.compute_bla(t), e)
            self.bla.update(t)

    def test_below_density(self):

        self.bla.density = .9

        test_times = [1., 4., 9., 25.]
        expected = [False, False, True, True]

        for i, (t, e) in enumerate(zip(test_times, expected)):
            with self.subTest(i=i):
                self.assertEqual(self.bla.below_density(t), e)