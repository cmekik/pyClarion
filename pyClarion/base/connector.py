'''
Tools for propagating information within a Clarion agent.

Usage
=====

The abstract ``Connector`` class enables client objects to listen and react to 
information flows within a Clarion agent. The abstract ``Propagator`` class 
extends the functionality of the ``Connector`` class to enable propagation of 
client outputs to downstream listeners.

This module defines four concrete classes based on these abstractions: 
``NodeConnector``, ``ChannelConnector``, ``SelectorConnector``, and 
``EffectorConnector``. ``NodeConnector`` instances are charged with determining 
and relaying the output of their client nodes to listeners. 
``ChannelConnectors`` are charged with feeding inputs to client ``Channel`` 
instances and relaying their outputs to listeners. 

Listeners to ``NodeConnector`` instances are expected to be either 
``ChannelConnector`` or ``SelectorConnector`` instances. ``NodeConnector`` 
instances are expected to only listen to ``ChannelConnector`` instances, and 
``EffectorConnector`` instances are expected to only listen to 
``SelectorConnector`` instances.

Instantiation
-------------

Since ``Connector`` is an abstract class, it cannot be directly instantiated:

>>> from pyClarion.base.junction import MaxJunction
>>> class MyPacket(ActivationPacket):
...     def default_activation(self, key):
...         return 0.0
... 
>>> class MyMaxJunction(MaxJunction[MyPacket]):
... 
...     def __call__(self, *input_maps : MyPacket) -> MyPacket:
... 
...         return MyPacket(super().__call__(*input_maps))
... 
>>> Connector(Node(), MyMaxJunction())
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class Connector with abstract methods __call__

The same is true of the ``Propagator`` class.

>>> Propagator(Node(), MyMaxJunction())
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class Propagator with abstract methods propagate

Users are not expected to directly implement the abstract ``__call__`` and 
``propagate`` methods of these classes. Instead, they should work with the 
concrete classes provided by this module.

Example
-------

The example below is adapted from the documentation of 
``pyClarion.base.channel`` to use ``Connector`` objects.

>>> from pyClarion.base.node import Microfeature, Chunk
>>> class MyTopDownPacket(MyPacket):
...     pass
... 
>>> class FineGrainedTopDownChannel(Channel[MyPacket]):
...     """Represents a top-down link between two individual nodes."""
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
>>> edge1 = FineGrainedTopDownChannel(ch, mf1, 1.0)
>>> edge2 = FineGrainedTopDownChannel(ch, mf2, 1.0)
>>> connectors = {
...     ch : NodeConnector(ch, MyMaxJunction()),
...     mf1 : NodeConnector(mf1, MyMaxJunction()),
...     mf2 : NodeConnector(mf2, MyMaxJunction()),
...     edge1 : ChannelConnector(edge1, MyMaxJunction()),
...     edge2 : ChannelConnector(edge2, MyMaxJunction())
... }
... 
>>> # We need to connect everything up.
>>> connectors[edge1].register(connectors[mf1].get_reporter())
>>> connectors[edge2].register(connectors[mf2].get_reporter())
>>> connectors[ch].register(
...     connectors[edge1].get_reporter(), connectors[edge2].get_reporter()
... )
>>> # Check that buffers are empty prior to test
>>> connectors[mf1].buffer == dict()
True
>>> connectors[mf2].buffer == dict()
True
>>> # Set initial chunk activation.
>>> connectors[ch].update('Initial Activation', MyPacket({ch : 1.0}))
>>> # Propagate activations
>>> for connector in connectors.values():
...     connector()
... 
>>> # Check that propagation worked
>>> connectors[mf1].buffer == {edge1 : MyPacket({mf1 : 1.0})}
True
>>> connectors[mf2].buffer == {edge2 : MyPacket({mf2 : 1.0})}
True
>>> # Note: ``ch`` still remembers its activation after firing.
>>> connectors[ch].buffer == {'Initial Activation' : MyPacket({ch : 1.0})}
True
>>> # In other words, buffers must be manually cleared when necessary.
>>> connectors[ch].clear()
>>> connectors[ch].buffer == dict()
True


Why Separate Dedicated ``Connector`` classes for Nodes and Channels?
--------------------------------------------------------------------

The choice to have different activation connectors for nodes and channels 
derives from a compromise between fidelity to Clarion theory and a parsimonious, 
flexible implementation.

Since the fundamental representational construct in Clarion theory is the node, 
it is desirable to provide dedicated ``Connector`` instances for individual 
nodes: such a design encapsulates relevant information about a given node in a 
single object. 

Implementation of distinct ``Connector`` types for activation channels and nodes 
provides a uniform interface for the use of channels at varying levels of 
granularity while simultaneously enabling encapsulation of relevant information 
about individual nodes.

An advantage of such an architecture is that it allows for a simple mechanism 
for activation cycle synchronization. This can be done by, for example, 
sequentially propagating activations through all nodes then all channels.

The example below adapts another example from 
``pyClarion.base.channel`` in order to demonstrate how the dual 
activation Connector architecture may handle different levels of granularity.

>>> class MyTopLevelPacket(MyPacket):
...     pass
... 
>>> class MyAssociativeNetwork(Channel[MyPacket]):
... 
...     def __init__(
...         self, assoc : Dict[Node, Dict[Node, float]]
...     ) -> None:
...         """Initialize an associative network.
... 
...         :param assoc: A dict that maps conclusion chunks to 
...         dicts mapping their condition chunks to their respective weights.
...         """
...         self.assoc = assoc
... 
...     def __call__(self, input_map : MyPacket) -> MyTopLevelPacket:
...         output = MyTopLevelPacket()
...         for conclusion_chunk in self.assoc:
...             output[conclusion_chunk] = 0.0
...             for condition_chunk in self.assoc[conclusion_chunk]:
...                 output[conclusion_chunk] += (
...                         input_map[condition_chunk] *
...                         self.assoc[conclusion_chunk][condition_chunk]
...                 )
...         return output
>>> ch1, ch2, ch3 = Chunk('BLOOD ORANGE'), Chunk('RED'), Chunk('YELLOW')
>>> assoc = {
...     ch2 : {ch1 : 0.4},
...     ch3 : {ch1 : 0.2}    
... }
>>> channel = MyAssociativeNetwork(assoc)
>>> connectors = {
...     ch1 : NodeConnector(ch1, MyMaxJunction()),
...     ch2 : NodeConnector(ch2, MyMaxJunction()),
...     ch3 : NodeConnector(ch3, MyMaxJunction()),
...     channel : ChannelConnector(channel, MyMaxJunction())
... }
>>> # Connect everything
>>> connectors[ch1].register(connectors[channel])
>>> connectors[ch2].register(connectors[channel])
>>> connectors[ch3].register(connectors[channel])
>>> connectors[channel].register(connectors[ch1], connectors[ch2], connectors[ch3])
>>> # Set initial chunk activation.
>>> connectors[ch1].update('Initial Activation', MyPacket({ch1 : 1.0}))
>>> # Propagate activations
>>> for Connector in connectors.values():
...     Connector()
... 
>>> # Check that propagation worked
>>> # Note: Initial activation not yet cleared, so ``ch1`` activation persists
>>> connectors[ch1].propagate() == MyPacket({ch1 : 1.0})
True
>>> connectors[ch2].propagate() == MyPacket({ch2 : 0.4})
True
>>> connectors[ch3].propagate() == MyPacket({ch3 : 0.2})
True
>>> connectors[channel].propagate() == MyPacket({ch2 : 0.4, ch3 : 0.2})
True
'''


import abc
import copy
from typing import (
    TypeVar, Generic, Hashable, Callable, Any, Dict, Set, List, Iterable
)
from pyClarion.base.node import Node
from pyClarion.base.packet import Packet, ActivationPacket, SelectorPacket
from pyClarion.base.channel import Channel
from pyClarion.base.junction import Junction
from pyClarion.base.selector import Selector
from pyClarion.base.effector import Effector

##################
# TYPE VARIABLES #
##################

Ct = TypeVar('Ct')
It = TypeVar('It', bound=Packet)
Ot = TypeVar('Ot', bound=Packet)


################
# ABSTRACTIONS #
################

class Connector(Generic[Ct, It], abc.ABC):
    '''
    Connects client to relevant constructs as a listener.

    This is an abstract class, it cannot be directly instantiated.

    For details, see module documentation.
    '''

    def __init__(self, client: Ct, junction: Junction) -> None:

        self.client: Ct = client
        self.junction = junction
        self.input_links: List[Callable[..., It]] = list()
        self.input_buffer: List[It] = list()

    @abc.abstractmethod
    def __call__(self) -> None:
        pass

    def pull(self) -> None:
        
        self.input_buffer = [callback() for callback in self.input_links] 

    def add_link(self, callback: Callable[..., It]) -> None:

        self.input_links.append(callback)


class Propagator(Connector[Ct, It], Generic[Ct, It, Ot]):
    '''
    Allows client to push activation packets to listeners.

    This is an abstract class, it cannot be directly instantiated.

    For details, see module documentation.
    '''
    
    def __init__(self, client: Ct, junction: Junction) -> None:
        
        super().__init__(client, junction)
        self.output_buffer : Ot = self.propagate()

    def __call__(self) -> None:
        '''Update ``self.output_buffer``.'''

        self.pull()        
        self.output_buffer = self.propagate()

    @abc.abstractmethod
    def propagate(self) -> Ot:
        '''
        Compute and return current output of ``self.client``.
        
        This is an abstract method.
        '''
        pass

    @abc.abstractmethod
    def get_pull_method(self) -> Callable:
        pass


class ActivationPropagator(Propagator[Ct, ActivationPacket, ActivationPacket]):

    def get_pull_method(self) -> Callable:

        return self.get_output

    def get_output(self, nodes: Iterable[Node] = None) -> ActivationPacket:
        '''
        Return current output of ``self``.

        Returns a subpacket of ``self.output_buffer``. Intended to be used as a 
        pull method for listeners of this propagator.

        :param nodes: Nodes to be included in returned subpacket. If ``None``, 
            all nodes in ``self.output_buffer`` will be included. 
        '''

        if not nodes:
            nodes = self.output_buffer.keys()
        return self.output_buffer.subpacket(nodes)


####################
# CONCRETE CLASSES #
####################


class NodeConnector(ActivationPropagator[Node]):
    '''Embeds a node in a network.

    For details, see module documentation.
    '''

    def pull(self) -> None:
        
        self.input_buffer = [
            callback([self.client]) for callback in self.input_links
        ]

    def propagate(self) -> ActivationPacket:
        '''Compute and return current output of client node.

        Passes activation packets stored in ``self.buffer`` through 
        ``self.junction`` and returns the result.
        '''
        
        return self.junction(*self.input_buffer)


class ChannelConnector(ActivationPropagator[Channel]):
    '''Embeds an activation channel in a Clarion agent.

    For details, see module documentation.
    '''

    def propagate(self) -> ActivationPacket:
        '''Update listeners with new output of client activation channel.

        Passes activation packets stored in ``self.buffer`` through 
        ``self.junction`` and feeds result to ``self.client``. Returns the 
        output of ``self.clent``.
        '''

        return self.client(self.junction(*self.input_buffer))


class SelectorConnector(Propagator[Selector, ActivationPacket, SelectorPacket]):
    '''
    Embeds an action selector in a Clarion agent.

    For details, see module documentation.
    '''

    def propagate(self) -> SelectorPacket:
        '''Update listeners with new output of client activation channel.

        Passes activation packets stored in ``self.buffer`` through 
        ``self.junction`` and feeds result to ``self.client``. Returns the 
        output of ``self.clent``.
        '''

        return self.client(self.junction(*self.input_buffer))

    def get_pull_method(self) -> Callable:

        return self.get_output

    def get_output(self, nodes: Iterable[Node] = None) -> SelectorPacket:
        '''
        Return current output of ``self``.

        Returns a deepcopy of ``self.output_buffer``. Intended to be used as a 
        pull method for listeners of this propagator. 
        '''

        return copy.deepcopy(self.output_buffer)


class EffectorConnector(Connector[Effector, SelectorPacket]):
    '''
    Embeds an action effector in a Clarion agent.

    For details, see module documentation.
    '''

    def __call__(self) -> None:

        self.pull()
        self.client(self.junction(*self.input_buffer))