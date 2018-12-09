import unittest
from pyClarion.base.symbols import *
from pyClarion.base.packets import *


class TestMakePacket(unittest.TestCase):

    def test_activation_packet(self):

        d = {Chunk('name'): 1., Microfeature('dim', 'val'): .7}
        source = Flow('name', FlowType.TT | FlowType.TB)
        packet = make_packet(source, d)
        with self.subTest(i = 'packet well-formed'):
            self.assertEqual(packet, ActivationPacket(d, source))
        with self.subTest(i = 'packet.strength is a mappingproxy'):
            with self.assertRaises(TypeError):
                packet.strengths[Chunk('name2')] = 1.

    def test_decision_packet(self):

        d = {Chunk('eat'): 4., Chunk('play'): .7, Chunk('do homework'): .3}
        chosen = Chunk('play')
        source = Response('name', ConstructType.Chunk)
        packet = make_packet(source, (d, chosen))
        with self.subTest(i = 'packet well-formed'):
            self.assertEqual(packet, DecisionPacket(d, chosen, source))
        with self.subTest(i = 'packet.strength is a mappingproxy'):
            with self.assertRaises(TypeError):
                packet.strengths[Chunk('name2')] = 1.
