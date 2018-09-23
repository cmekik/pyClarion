"""
Tools for linking up a network of construct realizers.

Example
=======

>>> from pyClarion.base.processor import Channel, MaxJunction
>>> from pyClarion.base.symbols import Microfeature, Chunk, Flow, FlowType
>>> class MyPacket(ActivationPacket[float]):
...     def default_activation(self, key):
...         return 0.0
... 
>>> class MyMaxJunction(MaxJunction[MyPacket]):
... 
...     def __call__(self, *input_maps : MyPacket) -> MyPacket:
...         output = MyPacket()
...         if input_maps:
...             output.update(super().__call__(*input_maps))
...         return output
... 
>>> class MyTopDownPacket(MyPacket):
...     pass
... 
>>> class FineGrainedTopDownChannel(Channel[MyPacket, MyTopDownPacket]):
...     '''Represents a top-down link between two individual nodes.'''
... 
...     def __init__(self, chunk, microfeature, weight):
...         self.chunk = chunk
...         self.microfeature = microfeature
...         self.weight = weight
...
...     def __call__(self, input_map : MyPacket) -> MyTopDownPacket:
...         output = MyTopDownPacket({
...             self.microfeature : input_map[self.chunk] * self.weight
...         })
...         return output
... 
>>> ch = Chunk("APPLE")
>>> mf1 = Microfeature("color", "red")
>>> mf2 = Microfeature("tasty", True)
>>> fl1 = Flow(id=(ch, mf1), flow_type=FlowType.TopDown)
>>> fl2 = Flow(id=(ch, mf2), flow_type=FlowType.TopDown)
>>> channel1 = FineGrainedTopDownChannel(ch, mf1, 1.0)
>>> channel2 = FineGrainedTopDownChannel(ch, mf2, 1.0)
>>> ch_struct = NodeRealizer(ch, MyMaxJunction())
>>> mf1_struct = NodeRealizer(mf1, MyMaxJunction())
>>> mf2_struct = NodeRealizer(mf2, MyMaxJunction())
>>> flow_struct_1 = FlowRealizer(fl1, MyMaxJunction(), channel1)
>>> flow_struct_2 = FlowRealizer(fl2, MyMaxJunction(), channel2)
>>> nodes = {
...     ch : NodePropagator(ch_struct),
...     mf1 : NodePropagator(mf1_struct),
...     mf2 : NodePropagator(mf2_struct),
... }
...
>>> flows = {
...     fl1 : FlowPropagator(flow_struct_1),
...     fl2 : FlowPropagator(flow_struct_2)
... } 
>>> # We need to connect everything up.
>>> nodes[mf1].watch(fl1, flows[fl1].get_pull_method())
>>> nodes[mf2].watch(fl2, flows[fl2].get_pull_method())
>>> flows[fl1].watch(ch, nodes[ch].get_pull_method())
>>> flows[fl2].watch(ch, nodes[ch].get_pull_method())
>>> # Check that buffers are empty prior to test
>>> nodes[mf1].output_buffer == MyPacket()
True
>>> nodes[mf2].output_buffer == MyPacket()
True
>>> # Set initial chunk activation.
>>> nodes[ch].add_link('External Input', lambda x: MyPacket({ch : 1.0}))
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
>>> nodes[mf1].output_buffer == MyPacket({mf1 : 1.0})
True
>>> nodes[mf2].output_buffer == MyPacket({mf2 : 1.0})
True

"""


import abc
import copy
from typing import (
    TypeVar, Generic, Hashable, Callable, Any, Dict, Set, List, Iterable, Union
)
from pyClarion.base.symbols import Node
from pyClarion.base.packet import Packet, ActivationPacket, DecisionPacket
from pyClarion.base.realizer import (
    BasicConstructRealizer, NodeRealizer, FlowRealizer, 
    AppraisalRealizer, ActivityRealizer
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
        self.input_buffer: List[It] = list()

    @abc.abstractmethod
    def __call__(self) -> None:
        pass

    def pull(self) -> None:
        
        self.input_buffer = [
            callback() for callback in self.input_links.values()
        ] 

    def watch(self, identifier: Hashable, callback: Callable[..., It]) -> None:

        self.input_links[identifier] = callback

    def drop(self, identifier: Hashable):

        del self.input_links[identifier]


class Observable(Generic[St, Ot], abc.ABC):

    def __init__(self, realizer: St) -> None:
        
        self.realizer = realizer
        self.output_buffer : Ot = self.propagate()

    @abc.abstractmethod
    def __call__(self):
        """Update ``self.output_buffer``."""
        pass

    @abc.abstractmethod
    def propagate(self) -> Ot:
        """
        Compute and return current output of ``self.client``.
        
        This is an abstract method.
        """

        pass

    @abc.abstractmethod
    def get_pull_method(self) -> Callable:
        pass


class Propagator(Observable[St, Ot], Observer[It], Generic[St, It, Ot]):
    """
    Allows listeners to pull output of client.

    This is an abstract class, it cannot be directly instantiated.

    For details, see module documentation.
    """
    
    def __init__(self, realizer: St) -> None:
        
        Observer.__init__(self)
        Observable.__init__(self, realizer)

    def __call__(self) -> None:
        """Update ``self.output_buffer``."""

        self.pull()        
        self.output_buffer = self.propagate()


class ActivationPropagator(Propagator[St, ActivationPacket, ActivationPacket]):

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

    def pull(self) -> None:
        
        self.input_buffer = [
            callback([self.realizer.construct]) 
            for callback in self.input_links.values()
        ]

    def propagate(self) -> ActivationPacket:
        """Compute and return current output of client node."""
        
        return self.realizer.junction(*self.input_buffer)


class FlowPropagator(ActivationPropagator[FlowRealizer]):
    """
    Embeds an activation flow in a Clarion agent.

    For details, see module documentation.
    """

    def propagate(self) -> ActivationPacket:
        """Compute and return output of client activation flow."""

        return self.realizer.channel(
            self.realizer.junction(*self.input_buffer)
        )


class AppraisalPropagator(
    Propagator[AppraisalRealizer, ActivationPacket, DecisionPacket]
):
    """
    Embeds an action selector in a Clarion agent.

    For details, see module documentation.
    """

    def __call__(self):

        super().__call__()

    def propagate(self) -> DecisionPacket:
        """Compute and return output of client selector."""

        return self.realizer.selector(
            self.realizer.junction(*self.input_buffer)
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
        
        self.realizer = realizer
        super().__init__()

    def __call__(self):

        self.pull()
        for decision_packet in self.input_buffer:
            self.realizer.effector(decision_packet)