import unittest
import unittest.mock
from clock import Clock
from bla import BLA

class TestBLA(unittest.TestCase):
    
    def setUp(self):

        self.mock_clock = unittest.mock.Mock(spec=Clock)
        self.bla = BLA(clock=self.mock_clock)

        # Initialize BLA at time = 0
        self.mock_clock.get_time.return_value = 0.
        self.bla.update()

    def test_update(self):

        self.assertEqual(self.bla.timestamps[0], self.mock_clock.get_time())

    def test_compute_bla(self):

        test_times = [1.5, 2., 15.]
        expected = [ 
            2. * (1.5 ** -.5),
            2. * ((2. ** -.5) + ((2. - 1.5) ** -.5)),
            2. * ((15. ** -.5) + ((15. - 1.5) ** -.5) + ((15. - 2.) ** -.5))
        ]

        for i, (t, e) in enumerate(zip(test_times, expected)):
            self.mock_clock.get_time.return_value = t
            with self.subTest(i=i):
                self.assertAlmostEqual(self.bla.compute_bla(), e)
            self.bla.update()

    def test_below_density(self):

        self.bla.density = .9

        test_times = [1., 4., 9., 25.]
        expected = [False, False, True, True]

        for i, (t, e) in enumerate(zip(test_times, expected)):
            self.mock_clock.get_time.return_value = t
            with self.subTest(i=i):
                self.assertEqual(self.bla.below_density(), e)
