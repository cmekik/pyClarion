'''
This module provides tools for propagating activations in a Clarion agent.

Usage
=====

This module provides the abstract ``ActivationHandler`` class and, more 
importantly, its two concrete subclasses, ``NodeHandler`` and 
``ChannelHandler``. ``ActivationHandler`` instances define how their clients are 
embedded in a Clarion agent's neural network.

``NodeHandler`` instances are charged with finalizing and relaying the output of 
their client nodes to listeners. ``ChannelHandlers``, on the other hand, are 
charged with feeding client ``Channel`` instances their input and relaying their 
output to listeners. It is expected that ``ChannelHandler`` instances will 
listen to ``NodeHandler`` instances and vice versa, though this is not enforced.

Since ``ActivationHandler`` is an abstract class, it cannot be directly 
instantiated:

>>> from pyClarion.base.node import Node
>>> from pyClarion.base.activation.packet import ActivationPacket
>>> from pyClarion.base.activation.junction import GenericMaxJunction
>>> class MyPacket(ActivationPacket):
...     def default_activation(self, key):
...         return 0.0
... 
>>> class MyMaxJunction(GenericMaxJunction):
... 
...     @property
...     def output_type(self):
...         return MyPacket
...
>>> ActivationHandler(Node(), MyMaxJunction())
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class ActivationHandler with abstract methods propagate

Users are not expected to implement the abstract ``propagate`` method of the 
``ActivationHandler`` class. Instead, they should work with the concrete classes 
``NodeHandler`` and ``ChannelHandler``, each of which implement their own 
version of ``propagate``.

Example
-------

The example below is adapted from the documentation of 
``pyClarion.base.activation.channel`` to use ``ActivationHandler`` objects.

>>> from pyClarion.base.node import Microfeature, Chunk
>>> from pyClarion.base.activation.packet import TopDownPacket
>>> from pyClarion.base.activation.channel import TopDown
>>> class MyTopDownPacket(TopDownPacket, MyPacket):
...     pass
... 
>>> class FineGrainedTopDown(TopDown):
...     """Represents a top-down link between two individual nodes."""
... 
...     def __init__(self, chunk, microfeature, weight):
...         self.chunk = chunk
...         self.microfeature = microfeature
...         self.weight = weight
...
...     def __call__(self, input_map : ActivationPacket) -> MyTopDownPacket:
...         output = MyTopDownPacket({
...             self.microfeature : input_map[self.chunk] * self.weight
...         })
...         return output
... 
>>> ch = Chunk("APPLE")
>>> mf1 = Microfeature("color", "red")
>>> mf2 = Microfeature("tasty", True)
>>> edge1 = FineGrainedTopDown(ch, mf1, 1.0)
>>> edge2 = FineGrainedTopDown(ch, mf2, 1.0)
>>> handlers = {
...     ch : NodeHandler(ch, MyMaxJunction()),
...     mf1 : NodeHandler(mf1, MyMaxJunction()),
...     mf2 : NodeHandler(mf2, MyMaxJunction()),
...     edge1 : ChannelHandler(edge1, MyMaxJunction()),
...     edge2 : ChannelHandler(edge2, MyMaxJunction())
... }
... 
>>> # We need to connect everything up.
>>> handlers[edge1].register(handlers[mf1])
>>> handlers[edge2].register(handlers[mf2])
>>> handlers[ch].register(handlers[edge1], handlers[edge2])
>>> # Check that buffers are empty prior to test
>>> handlers[mf1].buffer == dict()
True
>>> handlers[mf2].buffer == dict()
True
>>> # Set initial chunk activation.
>>> handlers[ch].update('Initial Activation', MyPacket({ch : 1.0}))
>>> # Propagate activations
>>> for handler in handlers.values():
...     handler()
... 
>>> # Check that propagation worked
>>> handlers[mf1].buffer == {edge1 : MyPacket({mf1 : 1.0})}
True
>>> handlers[mf2].buffer == {edge2 : MyPacket({mf2 : 1.0})}
True
>>> # Note: ``ch`` still remembers its activation after firing.
>>> handlers[ch].buffer == {'Initial Activation' : MyPacket({ch : 1.0})}
True
>>> # In other words, buffers must be manually cleared when necessary.
>>> handlers[ch].clear()
>>> handlers[ch].buffer == dict()
True


Why Separate Activation Handlers for Nodes and Channels?
--------------------------------------------------------

The choice to have two types of activation handlers derives from a compromise 
between fidelity to Clarion theory and a parsimonious, flexible implementation.

Since the fundamental representational construct in Clarion theory is the node, 
it is desirable to provide dedicated ``ActivativationHandler`` instances for 
individual nodes: such a design encapsulates relevant information about a given 
node in a single object. 

Implementation of distinct ``ActivationHandler`` types for activation channels 
and nodes provides a uniform interface for the use of channels at varying levels 
of granularity while simultaneously enabling encapsulation of relevant 
information about individual nodes.

The example below adapts another example from 
``pyClarion.base.activation.channel`` in order to demonstrate how the dual 
activation handler architecture may handle different levels of granularity.

>>> from pyClarion.base.activation.packet import TopLevelPacket
>>> from pyClarion.base.activation.channel import TopLevel
>>> class MyTopLevelPacket(TopLevelPacket, MyPacket):
...     pass
... 
>>> class MyAssociativeNetwork(TopLevel):
... 
...     def __init__(
...         self, association_matrix : T.Dict[Node, T.Dict[Node, float]]
...     ) -> None:
...         """Initialize an associative network.
... 
...         :param association_matrix: A dict that maps conclusion chunks to 
...         dicts mapping their condition chunks to their respective weights.
...         """
...         self.assoc = association_matrix
... 
...     def __call__(self, input_map : ActivationPacket) -> MyTopLevelPacket:
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
>>> association_matrix = {
...     ch1 : {ch1 : 0.0, ch2 : 0.4, ch3 : 0.2},
...     ch2 : {ch1 : 0.4, ch2 : 0.0, ch3 : 0.1},
...     ch3 : {ch1 : 0.2, ch2 : 0.1, ch3 : 0.0}    
... }
>>> channel = MyAssociativeNetwork(association_matrix)
>>> handlers = {
...     ch1 : NodeHandler(ch1, MyMaxJunction()),
...     ch2 : NodeHandler(ch2, MyMaxJunction()),
...     ch3 : NodeHandler(ch3, MyMaxJunction()),
...     channel : ChannelHandler(channel, MyMaxJunction())
... }
>>> # Connect everything
>>> handlers[ch1].register(handlers[channel])
>>> handlers[ch2].register(handlers[channel])
>>> handlers[ch3].register(handlers[channel])
>>> handlers[channel].register(handlers[ch1], handlers[ch2], handlers[ch3])
>>> # Set initial chunk activation.
>>> handlers[ch1].update('Initial Activation', MyPacket({ch1 : 1.0}))
>>> # Propagate activations
>>> for handler in handlers.values():
...     handler()
... 
>>> # Check that propagation worked
>>> # Note: Initial activation not yet cleared, so ``ch1`` activation persists
>>> handlers[ch1].propagate() == MyPacket({ch1 : 1.0})
True
>>> handlers[ch2].propagate() == MyPacket({ch2 : 0.4})
True
>>> handlers[ch3].propagate() == MyPacket({ch3 : 0.2})
True
>>> handlers[channel].propagate() == MyPacket({ch1 : 0.0, ch2 : 0.4, ch3 : 0.2})
True
'''


import abc
import typing as T
from pyClarion.base.node import Node
from pyClarion.base.activation.channel import Channel
from pyClarion.base.activation.packet import ActivationPacket
from pyClarion.base.activation.junction import Junction


##################
# TYPE VARIABLES #
##################


Tv = T.TypeVar('Tv', bound=T.Hashable)


###############
# ABSTRACTION #
###############


class ActivationHandler(T.Generic[Tv], abc.ABC):
    '''Embeds a client in a Clarion network.

    This is an abstract class, it cannot be directly instantiated.

    For details, see module documentation.
    '''

    def __init__(self, client : Tv, junction : Junction) -> None:
        '''
        Initialize an activation handler serving construct ``client``.

        :param client:
        :param junction:
        '''

        self._client : Tv = client
        self._junction = junction
        self._buffer : T.Dict[T.Hashable, ActivationPacket] = dict()
        self._listeners : T.Set['ActivationHandler'] = set()
        # String literal in type hint above represents forward reference to 
        # type(self). See PEP 484 section on Forward References for details.

    def __call__(self) -> None:
        '''Update listeners with new output of ``self.client``.

        .. warning::
           It is expected that ``NodeHandler`` instances will only update 
           listeners with the activations of their clients. However, this 
           expectation is currently not enforced.
        '''
        
        output_packet = self.propagate()
        self.notify_listeners(output_packet)


    def register(
        self, *handlers : 'ActivationHandler'
        # String literal in type hint above represents forward reference to 
        # type(self). See PEP 484 section on Forward References for details.
    ) -> None:
        '''Register ``handler`` as a listener of ``self``.

        :param handlers: Activation handler that listen to self.
        '''

        for handler in handlers:
            self.listeners.add(handler)

    def update(
        self, construct : T.Hashable, packet : ActivationPacket
    ) -> None:
        '''Update output of ``construct`` recorded in ``self.buffer``.
        '''

        self.buffer[construct] = packet

    def clear(self) -> None:
        '''Empty ``self.buffer``.'''

        self.buffer.clear()

    @abc.abstractmethod
    def propagate(self) -> ActivationPacket:
        '''Compute and return current output of ``self.client``.
        
        This is an abstract method.
        '''
        pass

    def notify_listeners(self, packet : ActivationPacket) -> None:
        '''Report new output of ``self.client`` to listeners.
        '''

        for listener in self.listeners:
            listener.update(self.client, packet)

    @property
    def client(self) -> Tv:
        '''The client construct whose outputs are handled by ``self``.
        '''
        return self._client

    @property
    def buffer(self) -> T.Dict[T.Hashable, ActivationPacket]:
        '''Buffers inputs to ``self.client``.
        
        Activations recorded here persist until changed or manually cleared.
        '''
        return self._buffer

    @property
    def junction(self) -> Junction:
        '''Preprocessor for combining input activations.'''
        return self._junction

    @property
    def listeners(self) -> T.Set['ActivationHandler']:
        '''Set of constructs interested in output of ``self.client``.'''
        # String literal in type hint above represents forward reference to 
        # type(self). See PEP 484 section on Forward References for details.
        return self._listeners 


####################
# CONCRETE CLASSES #
####################


class NodeHandler(ActivationHandler[Node]):
    '''Embeds a node in a Clarion network.

    For details, see module documentation.
    '''

    def propagate(self) -> ActivationPacket:
        '''Compute and return current output of client node.

        Passes activation packets stored in ``self.buffer`` through 
        ``self.junction`` and returns the result.
        '''

        junction_output = self.junction(*self.buffer.values())
        output_packet = type(junction_output)(
            {self.client : junction_output[self.client]}
        )
        return output_packet



class ChannelHandler(ActivationHandler[Channel]):
    '''Embeds an activation channel in a Clarion network.

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


if __name__ == '__main__':
    import doctest
    doctest.testmod()