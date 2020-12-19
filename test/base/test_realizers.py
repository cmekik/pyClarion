from pyClarion import Symbol
from pyClarion.base.realizers import Realizer

import unittest
from unittest import mock


class TestRealizerMethods(unittest.TestCase):

    def test_init_accepts_name_of_type_Symbol(self):

        emitter = mock.Mock()
        Realizer(name=Symbol("chunks", 1), emitter=emitter) # Ok

    def test_init_rejects_name_of_type_other_than_Symbol(self):        

        emitter = mock.Mock()
        self.assertRaises(
            TypeError, 
            Realizer,
            name="My Realizer", 
            emitter=emitter
        )
 