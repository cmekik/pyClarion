import unittest
from pyClarion.base.symbols import *
from pyClarion.base.realizers import *


class TestMayConnect(unittest.TestCase):

    def assertMayConnect(self, realizer, data):

        for source, target, expectation in data:
            with self.subTest(i=' -> '.join([str(source), str(target)])):
                self.assertEqual(
                    realizer.may_connect(source, target), expectation
                )


class TestSubsystemRealizerMayConnect(TestMayConnect):

    def assertMayConnect(self, data):
        # data is a list of tuples of the form 
        #     [(source, target, expected output), ...]

        subsystem = SubsystemRealizer(csym = Subsystem(1))

        super().assertMayConnect(subsystem, data)

    def test_basic_nodes_and_flows(self):

        data = [
            (Chunk(1), Flow(1, FlowType.TT), True),
            (Chunk(1), Flow(1, FlowType.TB), True),
            (Chunk(1), Flow(1, FlowType.BT), False),
            (Chunk(1), Flow(1, FlowType.BB), False),
            (Flow(1, FlowType.TT), Chunk(1), True),
            (Flow(1, FlowType.TB), Chunk(1), False),
            (Flow(1, FlowType.BT), Chunk(1), True),
            (Flow(1, FlowType.BB), Chunk(1), False),
            (Microfeature('d', 'v'), Flow(1, FlowType.TT), False),
            (Microfeature('d', 'v'), Flow(1, FlowType.TB), False),
            (Microfeature('d', 'v'), Flow(1, FlowType.BT), True),
            (Microfeature('d', 'v'), Flow(1, FlowType.BB), True),
            (Flow(1, FlowType.TT), Microfeature('d', 'v'), False),
            (Flow(1, FlowType.TB), Microfeature('d', 'v'), True),
            (Flow(1, FlowType.BT), Microfeature('d', 'v'), False),
            (Flow(1, FlowType.BB), Microfeature('d', 'v'), True),
        ]

        self.assertMayConnect(data)

    def test_response_and_behavior(self):

        data = [
            (
                Response(1, ConstructType.Chunk), 
                Behavior(1, Response(1, ConstructType.Chunk)), 
                True
            ),
            (
                Response(1, ConstructType.Chunk), 
                Behavior(2, Response(2, ConstructType.Chunk)), 
                False
            ),
            (
                Chunk(1),
                Response(1, ConstructType.Chunk),
                True
            ),
            (
                Flow(1, FlowType.TT),
                Response(1, ConstructType.Chunk),
                False
            )
        ]

        self.assertMayConnect(data)


class TestAgentRealizerMayConnect(TestMayConnect):

    def assertMayConnect(self, data):
        # data is a list of tuples of the form 
        #     [(source, target, expected output), ...]

        agent = AgentRealizer(Agent(1))

        super().assertMayConnect(agent, data)

    def test_may_connect(self):

        data = [
            (Subsystem(1), Buffer(1, (Subsystem(1), Subsystem(2))), False),
            (Buffer(1, (Subsystem(1), Subsystem(2))), Subsystem(1),  True),
            (Buffer(1, (Subsystem(1), Subsystem(2))), Subsystem(2),  True),
            (Buffer(1, (Subsystem(1), Subsystem(2))), Subsystem(3),  False),
        ]
