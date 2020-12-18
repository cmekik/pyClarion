
from pyClarion import Symbol, Propagator
from pyClarion.base.realizers import Realizer

import unittest
from unittest import mock

class TestRealizerMethods(unittest.TestCase):

    def test_init_requires_name_of_type_Symbol(self):

        propagator = mock.Mock()
        
        with self.subTest(name=Symbol("chunks", 1)):
            Realizer(name=Symbol("chunks", 1), emitter=propagator) # Ok
        
        with self.subTest(name="My Realizer"):
            self.assertRaises(
                TypeError, 
                Realizer,
                name="My Realizer", 
                emitter=propagator
            )
    
        with self.subTest(name=1234):
            self.assertRaises(
                TypeError, 
                Realizer,
                name=1234, 
                emitter=propagator
            )
    
    