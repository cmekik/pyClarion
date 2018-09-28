"""
Tools for linking up a network of construct realizers.

Example
=======

>>> from pyClarion.base.packets import Level
>>> from pyClarion.base.processors import Channel, MaxJunction, UpdateJunction
>>> from pyClarion.base.symbols import Microfeature, Chunk, Flow, FlowType
>>> class FineGrainedTopDownChannel(Channel[float]):
...     '''Represents a top-down link between two individual nodes.'''
... 
...     def __init__(self, chunk, microfeature, weight):
...         self.chunk = chunk
...         self.microfeature = microfeature
...         self.weight = weight
...
...     def __call__(self, input_map):
...         output = ActivationPacket(
...             {
...                 self.microfeature : input_map.get(self.chunk, 0.0) * self.weight
...             },
...             origin = Level.TopLevel,
...             default_factory = lambda key: 0.0
...         )
...         return output
... 
>>> ch = Chunk("APPLE")
>>> mf1 = Microfeature("color", "red")
>>> mf2 = Microfeature("tasty", True)
>>> fl1 = Flow(id=(ch, mf1), flow_type=FlowType.TopDown)
>>> fl2 = Flow(id=(ch, mf2), flow_type=FlowType.TopDown)
>>> channel1 = FineGrainedTopDownChannel(ch, mf1, 1.0)
>>> channel2 = FineGrainedTopDownChannel(ch, mf2, 1.0)
>>> ch_realizer = NodeRealizer(ch, MaxJunction())
>>> mf1_realizer = NodeRealizer(mf1, MaxJunction())
>>> mf2_realizer = NodeRealizer(mf2, MaxJunction())
>>> flow_realizer_1 = FlowRealizer(fl1, UpdateJunction(), channel1)
>>> flow_realizer_2 = FlowRealizer(fl2, UpdateJunction(), channel2)
>>> nodes = {
...     ch : NodePropagator(ch_realizer),
...     mf1 : NodePropagator(mf1_realizer),
...     mf2 : NodePropagator(mf2_realizer),
... }
...
>>> flows = {
...     fl1 : FlowPropagator(flow_realizer_1),
...     fl2 : FlowPropagator(flow_realizer_2)
... } 
>>> # We need to connect everything up.
>>> nodes[mf1].watch(fl1, flows[fl1].get_pull_method())
>>> nodes[mf2].watch(fl2, flows[fl2].get_pull_method())
>>> flows[fl1].watch(ch, nodes[ch].get_pull_method())
>>> flows[fl2].watch(ch, nodes[ch].get_pull_method())
>>> # Check that buffers are empty prior to test
>>> nodes[mf1].output_buffer == ActivationPacket()
True
>>> nodes[mf2].output_buffer == ActivationPacket()
True
>>> # Set initial chunk activation.
>>> nodes[ch].watch('External Input', lambda x: ActivationPacket({ch : 1.0}))
>>> for connector in nodes.values():
...     connector()
... 
>>> # Propagate activations
>>> for connector in flows.values():
...     connector()
... 
>>> for connector in nodes.values():
...     connector()
... 
>>> # Check that propagation worked
>>> nodes[mf1].output_buffer == ActivationPacket({mf1 : 1.0})
True
>>> nodes[mf2].output_buffer == ActivationPacket({mf2 : 1.0})
True

"""


import abc
import copy
from typing import (
    TypeVar, Generic, Hashable, Callable, Any, Dict, Set, List, Iterable, Union
)
from pyClarion.base.symbols import Node
from pyClarion.base.packets import Packet, ActivationPacket, DecisionPacket
from pyClarion.base.realizers.basic import (
    BasicConstructRealizer, NodeRealizer, FlowRealizer, AppraisalRealizer, 
    ActivityRealizer
)

##################
# TYPE VARIABLES #
##################

St = TypeVar('St', bound=BasicConstructRealizer)
It = TypeVar('It', bound=Packet)
Ot = TypeVar('Ot', bound=Packet)


################
# ABSTRACTIONS #
################

class Observer(Generic[It], abc.ABC):
    """
    Connects client to relevant constructs as a listener.

    This is an abstract class, it cannot be directly instantiated.

    For details, see module documentation.
    """

    def __init__(self) -> None:

        self.input_links: Dict[Hashable, Callable[..., It]] = dict()

    @abc.abstractmethod
    def __call__(self) -> None:
        pass

    def pull(self) -> List[It]:
        
        return [
            callback() for callback in self.input_links.values()
        ] 

    def watch(self, identifier: Hashable, callback: Callable[..., It]) -> None:

        self.input_links[identifier] = callback

    def drop(self, identifier: Hashable):

        del self.input_links[identifier]


class Observable(Generic[Ot], abc.ABC):

    output_buffer : Ot

    @abc.abstractmethod
    def __call__(self):
        """Update ``self.output_buffer``."""
        pass

    @abc.abstractmethod
    def get_pull_method(self) -> Callable:
        pass


class Propagator(Observable[Ot], Observer[It], Generic[It, Ot]):
    """
    Allows listeners to pull output of client.

    This is an abstract class, it cannot be directly instantiated.

    For details, see module documentation.
    """
    
    def __init__(self) -> None:
        
        Observer.__init__(self)
        Observable.__init__(self)


class BasicConstructPropagator(Propagator[It, Ot], Generic[St, It, Ot]):

    def __init__(self, realizer: St) -> None:

        super().__init__()
        self.realizer = realizer
        self.output_buffer = self.propagate([])

    def __call__(self) -> None:
        """Update ``self.output_buffer``."""

        self.output_buffer = self.propagate(self.pull())

    @abc.abstractmethod
    def propagate(self, inputs: List[It]) -> Ot:
        """
        Compute and return current output of ``self.client``.
        
        This is an abstract method.
        """

        pass


class ActivationPropagator(BasicConstructPropagator[St, ActivationPacket, ActivationPacket]):

    def __init__(self, realizer: St) -> None:

        super().__init__(realizer)

    def get_pull_method(self) -> Callable:

        return self.get_output

    def get_output(self, nodes: Iterable[Node] = None) -> ActivationPacket:
        """
        Return current output of ``self``.

        Returns a subpacket of ``self.output_buffer``. Intended to be used as a 
        pull method for listeners of this propagator.

        :param nodes: Nodes to be included in returned subpacket. If ``None``, 
            all nodes in ``self.output_buffer`` will be included. 
        """

        if not nodes:
            nodes = self.output_buffer.keys()
        return self.output_buffer.subpacket(nodes)


####################
# CONCRETE CLASSES #
####################


class NodePropagator(ActivationPropagator[NodeRealizer]):
    """
    Embeds a node in a network.

    For details, see module documentation.
    """

    def pull(self) -> List[ActivationPacket]:
        
        return [
            callback([self.realizer.construct]) 
            for callback in self.input_links.values()
        ]

    def propagate(self, inputs: List[ActivationPacket]) -> ActivationPacket:
        """Compute and return current output of client node."""
        
        return self.realizer.junction(*inputs)


class FlowPropagator(ActivationPropagator[FlowRealizer]):
    """
    Embeds an activation flow in a Clarion agent.

    For details, see module documentation.
    """

    def propagate(self, inputs: List[ActivationPacket]) -> ActivationPacket:
        """Compute and return output of client activation flow."""

        return self.realizer.channel(
            self.realizer.junction(*inputs)
        )


class AppraisalPropagator(
    BasicConstructPropagator[AppraisalRealizer, ActivationPacket, DecisionPacket]
):
    """
    Embeds an action selector in a Clarion agent.

    For details, see module documentation.
    """

    def __init__(self, realizer: AppraisalRealizer) -> None:

        super().__init__(realizer)

    def propagate(self, inputs: List[ActivationPacket]) -> DecisionPacket:
        """Compute and return output of client selector."""

        return self.realizer.selector(
            self.realizer.junction(*inputs)
        )

    def get_pull_method(self) -> Callable:

        return self.get_output

    def get_output(self, nodes: Iterable[Node] = None) -> DecisionPacket:
        """
        Return current output of ``self``.

        Returns a deepcopy of ``self.output_buffer``. Intended to be used as a 
        pull method for listeners of this propagator. 
        """

        return copy.deepcopy(self.output_buffer)


class ActivityDispatcher(Observer[DecisionPacket]):

    def __init__(self, realizer: ActivityRealizer) -> None:
        
        super().__init__()
        self.realizer = realizer

    def __call__(self):

        inputs = self.pull()
        for decision_packet in inputs:
            self.realizer.effector(decision_packet)