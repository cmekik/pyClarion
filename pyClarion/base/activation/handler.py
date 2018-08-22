'''
This module provides tools for propagating activations in a Clarion agent.

Usage
=====

The exports of this module is the abstract ``ActivationHandler`` class and, 
more importantly its two concrete subclasses, ``NodeHandler`` and 
``ChannelHandler``. ``ActivationHandlers`` define how their clients are 
embedded in a Clarion agent's neural network.

``NodeHandler`` instances are charged with finalizing and relaying the 
activation values of their client nodes to interested channels. 
``ChannelHandlers``, on the other hand, are charged with feeding ``Channel`` 
instances their input and relaying their output back to interested nodes.

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

Why Separate Activation Handlers for Nodes and Channels?
--------------------------------------------------------

The choice to have two types of activation handlers derives from a compromise 
between fidelity to Clarion theory and a parsimonious, flexible implementation.

``ActivationHandlers`` define how their client constructs are embedded in a 
Clarion agent's neural network. Since the fundamental representational construct 
in Clarion theory is the node, it is desirable to provide dedicated 
``ActivativationHandler`` instances for individual nodes: such a design 
encapsulates relevant information about a given node in a single object. 

It would be possible to include routines for managing activations of ``Channel`` 
instances associated with a particular node within the node's dedicated
``ActivationHandler``. However, such an architecture does not handle ``Channel`` 
types at a coarse level of granularity well. 

Implementation of distinct ``ActivationHandler`` types for activation channels 
and nodes enables the use of channels at varying levels of granularity with no 
overhead, while simultaneoutsly enabling encapsulation of relevant information 
about individual nodes.
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
        '''Update listeners with new activation of ``self``.

        .. warning::
           It is expected that ``NodeHandler`` instances will only update 
           listeners with the activations of their clients. 
        '''
        
        output_packet = self.propagate()
        self.notify_listeners(self.client, output_packet)


    def register(
        self, handler : 'ActivationHandler'
        # String literal in type hint above represents forward reference to 
        # type(self). See PEP 484 section on Forward References for details.
    ) -> None:
        '''Register ``handler`` as a listener of ``self``.

        :param handler: An activation handler that listens to self.
        '''
        self.listeners.add(handler)

    def update(
        self, construct : T.Hashable, packet : ActivationPacket
    ) -> None:
        '''Update buffer with a new activation packet.
        '''

        self.buffer[construct] = packet

    @abc.abstractmethod
    def propagate(self) -> ActivationPacket:
        '''Compute and return current activation of self.
        '''
        pass

    def notify_listeners(
        self, construct : T.Hashable, packet : ActivationPacket
    ) -> None:

        for listener in self.listeners:
            listener.update(construct, packet)

    @property
    def client(self) -> Tv:
        '''The client construct whose activations are handled by ``self``.
        '''
        return self._client

    @property
    def buffer(self) -> T.Dict[T.Hashable, ActivationPacket]:
        return self._buffer

    @property
    def junction(self) -> Junction:
        return self._junction

    @property
    def listeners(self) -> T.Set['ActivationHandler']:
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
        '''Compute and return current activation of self.

        Passes activation packets stored in ``self.buffer`` through 
        ``self.junction`` and returns the result.
        '''

        output_packet = self.junction(*self.buffer.values())
        return output_packet



class ChannelHandler(ActivationHandler[Channel]):
    '''Embeds an activation channel in a Clarion network.

    For details, see module documentation.
    '''

    def propagate(self) -> ActivationPacket:
        '''Update listeners with new activation of ``self``.
        '''

        input_packet = self.junction(*self.buffer.values())
        output_packet = self.client(input_packet)
        return output_packet


if __name__ == '__main__':
    import doctest
    doctest.testmod()