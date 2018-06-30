import unittest
import unittest.mock
from chunk import Chunk
from rule import Rule

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
                self.assertAlmostEqual(self.rule.apply(c2s), e)