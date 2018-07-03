import unittest
from stat import BLA

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
