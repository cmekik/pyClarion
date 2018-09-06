'''
This module provides tools for joining activation flows in pyClarion.

Usage
=====

This module exports the ``Junction`` class and related constructs. ``Junction`` 
objects are similar to pyClarion activation channels (see 
``pyClarion.base.channel``) in their design, but have a distinct call 
signature. ``Junction`` objects expect a sequence of activation packets as 
input and output a single activation packet.

Instantiation
-------------

The Junction class is abstract.

>>> Junction()
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class Junction with abstract methods __call__

It may be instanced when its ``__call__`` method is implemented.

Use Cases
---------

Like activation channels, junctions may capture static transformations of 
their inputs. Below, ``MyJunction`` returns the first activation packet it 
receieves, and discards the others. 

>>> class MyJunction(Junction[ActivationPacket]):
...     def __call__(
...     self, *input_maps : ActivationPacket) -> ActivationPacket:
...         return input_maps[0]
... 
>>> junc = MyJunction()
>>> from pyClarion.base.node import Node
>>> class MyPacket(ActivationPacket):
...     def default_activation(self, key):
...         return 0.0
...
>>> n1, n2, n3 = Node(), Node(), Node()
>>> p1 = MyPacket({n1 : 1.0, n2 : 0.1})
>>> p2 = MyPacket({n3 : 6.543})
>>> p3 = MyPacket()
>>> junc(p1) == p1
True
>>> junc(p2, p1) == p2
True
>>> junc(p3, p2, p1) == p3
True

Stateful Junctions
~~~~~~~~~~~~~~~~~~

Again like channels, the more interesting use of a junction is for capturing 
stateful processes. A stateful junction enables tunable prioritization of 
input streams.

Here is a junction where bottom-up activations are suppressed to half-strength 
but top-level activations flow unhindered.

>>> class MyTopLevelPacket(MyPacket):
...     pass
...
>>> class MyBottomUpPacket(MyPacket):
...     pass
... 
>>> class MyStatefulJunction(Junction[MyPacket]):
... 
...     def __init__(self, top_level_weight : float, bottom_up_weight : float):
...         self.weights = {
...             MyTopLevelPacket : top_level_weight,
...             MyBottomUpPacket : bottom_up_weight
...         }
...
...     def __call__(self, *input_maps : MyPacket) -> MyPacket:
... 
...         output = MyPacket()
...         nodes = get_nodes(*input_maps)
...         for n in nodes:
...             for input_map in input_maps:
...                 weighted = self.weight(n, input_map)
...                 if output[n] < weighted: 
...                     output[n] = weighted
...         return output
...     
...     def weight(self, n, packet):
...         return packet[n] * self.weights[type(packet)]
...
>>> junc = MyStatefulJunction(1.0, 0.5)
>>> p1 = MyTopLevelPacket({n1 : .2})
>>> p2 = MyBottomUpPacket({n1 : .3})
>>> p3 = MyBottomUpPacket({n1 : .6})
>>> junc(p1, p2) == MyPacket({n1 : .2})
True
>>> junc(p1, p3) == MyPacket({n1 : .3})
True

Generic Junctions
-----------------

There are several kinds of junction that occur in Clarion theory. For 
convenience, this module provides generic implementations of several of these 
junctions. In order to use these generic implementations, one must specify 
the desired output packet type.

>>> class MyMaxJunction(GenericMaxJunction[MyPacket]):
... 
...     @property
...     def output_type(self):
...         return MyPacket
...
>>> junc = MyMaxJunction()
>>> p1 = MyPacket({n1 : .2})
>>> p2 = MyPacket({n1 : .7})
>>> junc(p1, p2) == MyPacket({n1 : .7})
True
'''

from typing import Generic, TypeVar, Type
import abc
from pyClarion.base.node import get_nodes
from pyClarion.base.packet import ActivationPacket


###############
# ABSTRACTION #
###############


T = TypeVar('T', bound=ActivationPacket)


class Junction(Generic[T], abc.ABC):
    """An abstract class for handling the combination of chunk and/or 
    (micro)feature activations from multiple sources.
    """

    @abc.abstractmethod
    def __call__(self, *input_maps: T) -> T:
        """Return a combined mapping from chunks and/or microfeatures to 
        activations.

        kwargs:
            input_maps : Dicts mapping from chunks and/or microfeatures 
            to input activations.
        """

        pass


#####################
# GENERIC JUNCTIONS #
#####################


class GenericJunction(Junction[T]):
    """Base class for generic implementations of junctions. Expects an 
    output_type property.
    """

    @property
    @abc.abstractmethod
    def output_type(self) -> Type[T]:
        '''The output type of this junction'''
        pass


class GenericMaxJunction(GenericJunction[T]):
    """An activation junction returning max activations for all input nodes.
    """

    def __call__(self, *input_maps : T) -> T:
        """Return the maximum activation value for each input node.

        kwargs:
            input_maps : A set of mappings from chunks and/or (micro)features 
            to activations.
        """

        node_set = get_nodes(*input_maps)
        output = self.output_type()
        for n in node_set:
            for input_map in input_maps:
                if output[n] < input_map[n]:
                    output[n] = input_map[n]
        return output


if __name__ == '__main__':
    import doctest
    doctest.testmod()