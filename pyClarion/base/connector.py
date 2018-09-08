'''
Tools for propagating information within a Clarion agent.

Usage
=====

The abstract ``Connector`` class enables clients to listen and react to 
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


Why Dedicated ``Connector`` classes for Nodes and Channels?
-----------------------------------------------------------

The choice to have two types of activation connectors derives from a compromise 
between fidelity to Clarion theory and a parsimonious, flexible implementation.

Since the fundamental representational construct in Clarion theory is the node, 
it is desirable to provide dedicated ``Connector`` instances for individual 
nodes: such a design encapsulates relevant information about a given node in a 
single object. 

Implementation of distinct ``Connector`` types for activation channels and nodes 
provides a uniform interface for the use of channels at varying levels of 
granularity while simultaneously enabling encapsulation of relevant information 
about individual nodes.

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
from typing import TypeVar, Generic, Hashable, Callable, Any, Dict, Set
from pyClarion.base.node import Node
from pyClarion.base.packet import ActivationPacket, SelectorPacket
from pyClarion.base.channel import Channel
from pyClarion.base.junction import Junction
from pyClarion.base.selector import Selector
from pyClarion.base.effector import Effector

##################
# TYPE VARIABLES #
##################

Ct = TypeVar('Ct', bound=Hashable)
Pt = TypeVar('Pt', bound=ActivationPacket)


################
# ABSTRACTIONS #
################


class Reporter(object):
    '''
    A helper class for updating ``Connector`` instances.
    
    Exposes the ``update`` method of a client ``Connector`` instance. Reports 
    updates from an upstream sources to client.
    '''

    def __init__(self, connector_id : Hashable, updater : Callable) -> None:
        '''
        Initialize ``Reporter`` instance.

        :param connector_id: A unique id for client connector.
        :param updater: Method for updating client connector.
        '''

        self.connector_id = connector_id
        self.updater = updater

    def update(self, construct : Hashable, payload : Pt):
        '''
        Pass on new activations to activation Connector assigned to client.

        For param details see ``Connector.update``.
        '''

        self.updater(construct, payload)


class Connector(Generic[Ct, Pt], abc.ABC):
    '''
    Connects client to relevant constructs as a listener.

    This is an abstract class, it cannot be directly instantiated.

    For details, see module documentation.
    '''

    def __init__(self, client : Ct, junction : Junction) -> None:
        '''
        Initialize an activation Connector serving construct ``client``.

        :param client: The construct that is served by self.
        :param junction: Method for combining activations used by client.
        '''

        self.client : Ct = client
        self.junction : Junction = junction
        self.buffer : Dict[Hashable, Pt] = dict()

    @abc.abstractmethod
    def __call__(self) -> None:
        pass

    def update(self, construct : Hashable, payload : Pt) -> None:
        '''Update output of ``construct`` recorded in ``self.buffer``.'''

        self.buffer[construct] = payload

    def clear(self) -> None:
        '''Empty ``self.buffer``.'''

        self.buffer.clear()

    def get_reporter(self) -> Reporter:
        '''Create and return an Reporter with an update link to self.'''

        reporter : Reporter = Reporter(
            id(self), self.update
        )
        return reporter


class Propagator(Connector[Ct, Pt]):
    '''
    Allows client to push activation packets to listeners.

    This is an abstract class, it cannot be directly instantiated.

    For details, see module documentation.
    '''
    
    def __init__(self, client : Ct, junction : Junction) -> None:
        
        super().__init__(client, junction)
        self.listeners : Set[Reporter] = set()

    def __call__(self) -> None:
        '''
        Update listeners with new output of ``self.client``.

        .. warning::
           It is expected that ``Connector`` instances will only update 
           listeners with the activations of their clients. However, this 
           expectation is currently not enforced.
        '''
        
        output_packet = self.propagate()
        self.notify_listeners(output_packet)

    def register(self, *reporters : Reporter) -> None:
        '''
        Register reporters as listeners of ``self``.

        :param reporters: ``Reporter`` that listen to self.
        '''

        for reporter in reporters:
            self.listeners.add(reporter)

    def clear_listeners(self) -> None:
        '''Empty ``self.listeners``.'''

        self.listeners.clear()

    @abc.abstractmethod
    def propagate(self) -> ActivationPacket:
        '''
        Compute and return current output of ``self.client``.
        
        This is an abstract method.
        '''
        pass

    def notify_listeners(self, payload : ActivationPacket) -> None:
        '''Report new output of ``self.client`` to listeners.'''

        for listener in self.listeners:
            listener.update(self.client, payload)


####################
# CONCRETE CLASSES #
####################


class NodeConnector(Propagator[Node, Pt]):
    '''Embeds a node in a Clarion agent.

    For details, see module documentation.
    '''

    def propagate(self) -> ActivationPacket:
        '''Compute and return current output of client node.

        Passes activation packets stored in ``self.buffer`` through 
        ``self.junction`` and returns the result.
        '''

        junction_output = self.junction(*self.buffer.values())
        activation_dict = {self.client : junction_output[self.client]}
        output_packet = type(junction_output)(activation_dict)
        return output_packet


class ChannelConnector(Propagator[Channel, Pt]):
    '''Embeds an activation channel in a Clarion agent.

    For details, see module documentation.
    '''

    def propagate(self) -> ActivationPacket:
        '''Update listeners with new output of client activation channel.

        Passes activation packets stored in ``self.buffer`` through 
        ``self.junction`` and feeds result to ``self.client``. Returns the 
        output of ``self.clent``.
        '''

        input_packet = self.junction(*self.buffer.values())
        output_packet = self.client(input_packet)
        return output_packet


class SelectorConnector(Propagator[Selector, Pt]):
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

        input_packet = self.junction(*self.buffer.values())
        output_packet = self.client(input_packet)
        return output_packet


class EffectorConnector(Connector[Effector, Pt]):
    '''
    Embeds an action effector in a Clarion agent.

    For details, see module documentation.
    '''

    def __call__(self) -> None:

        input_packet = self.junction(*self.buffer.values())
        self.client(input_packet)


if __name__ == '__main__':
    import doctest
    doctest.testmod()