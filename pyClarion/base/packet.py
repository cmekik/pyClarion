"""
Tools for representing information about Clarion nodes.

Packets
=======

The ``Packet`` class represents represents mappings from nodes to data. 
``Packet`` instances behave like ``dict`` objects.

>>> n1, n2 = Node(), Node()
>>> p = ActivationPacket({n1 : 0.3}) 
>>> p[n1]
0.3
>>> p[n1] = 0.6
>>> p[n1]
0.6
>>> p[n2] = 0.2
>>> p[n2]
0.2

Value Types
-----------

``Packet`` is implemented as a generic class taking one type variable. This type 
variable specifies the expected packet value type. Its use is optional.

Activation Packets
==================

Activation packets represent patterns of node activations. 

The ``ActivationPacket`` class provides a ``default_activation`` method, which 
may be overridden to capture assumptions about default activation values. 

>>> class MyPacket(ActivationPacket):
...     def default_activation(self, key):
...         return 0.0
...
>>> MyPacket()
MyPacket({})
>>> MyPacket()[Node()]
0.0

When a default value is provided by ``default_activation``, ``ActivationPacket`` 
objects handle unknown keys like ``collections.defaultdict`` objects: they 
output a default value and record the new ``(key, value)`` pair.

>>> p = MyPacket()
>>> n3 = Node()
>>> n3 in p
False
>>> p[n3]
0.0
>>> n3 in p
True

The ``default_activation`` method can be set to return different default values 
for different nodes.

>>> from pyClarion.base.symbols import Microfeature, Chunk
>>> class MySubtlePacket(ActivationPacket[float]):
...     def default_activation(self, key):
...         if isinstance(key, Microfeature):
...             return 0.5
...         elif isinstance(key, Chunk):
...             return 0.0
... 
>>> mf = Microfeature("color", "red")
>>> ch = Chunk(1234)
>>> p = MySubtlePacket()
>>> mf in p
False
>>> ch in p
False
>>> p[mf]
0.5
>>> p[ch]
0.0

Activation Packet Types
-----------------------

The type of an ``ActivationPacket`` may be used to drive conditional processing. 

Different activation sources may output packets of different types. For 
instance, a top-down activation cycle may output an instance of 
``TopDownPacket``, as illustrated in the example below.

>>> class MyTopDownPacket(MyPacket):
...     '''Represents the output of a top-down activation cycle.'''
...
...     pass
... 
>>> def my_top_down_activation_cycle(packet):
...     '''A dummy top-down activation cycle for demonstration purposes''' 
...     val = max(packet.values())
...     return MyTopDownPacket({n3 : val})
... 
>>> packet = MyPacket({n1 : .2, n2 : .6})
>>> output = my_top_down_activation_cycle(packet)
>>> output == MyPacket({n3 : .6})
True
>>> isinstance(output, MyPacket)
True
>>> isinstance(output, MyTopDownPacket)
True

Decision Packets
================

The ``DecisionPacket`` class represents the results of an action selection 
cycle.

``DecisionPacket`` have an attribute called ``chosen``, whose contents represent 
chosen action chunks.

>>> ch1, ch2 = Chunk(1), Chunk(2)
>>> DecisionPacket({ch1 : .78, ch2 : .24}, chosen={ch1})
DecisionPacket({Chunk(id=1): 0.78, Chunk(id=2): 0.24}, chosen={Chunk(id=1)})

"""

from abc import abstractmethod
from typing import MutableMapping, TypeVar, Hashable, Mapping, Set, Any, Iterable
from collections import UserDict
from pyClarion.base.symbols import Node, Chunk


At = TypeVar("At")


class Packet(UserDict, MutableMapping[Node, At]):
    """
    Base class for encapsulating information about nodes.

    Takes one type variable, ``At``, which is an unrestricted type variable 
    denoting the expected type for data values.
    """

    def __repr__(self) -> str:
        
        repr_ = ''.join(
            [
                type(self).__name__,
                '(',
                super().__repr__(),
                ')'
            ]
        )
        return repr_


class ActivationPacket(Packet[At]):
    """
    A class for representing node activations.

    Default activation values may be implemented by overriding the 
    ``default_activation`` method. When defined, default activations are handled 
    similarly to ``collections.defaultdict``.

    The precise type of an ``ActivationPacket`` instance may encode important 
    metadata, such as information about the source of the packet. 

    See module documentation for further details and examples.
    """

    def __missing__(self, key: Node) -> At:

        value : At = self.default_activation(key)
        self[key] = value
        return value

    def default_activation(self, key: Node) -> At:
        """Return designated default value for the given input."""
        
        raise KeyError

    def subpacket(self, nodes: Iterable[Node]):
        """Return a subpacket containing activations for ``nodes``."""
        
        return type(self)({node: self[node] for node in nodes})


class DecisionPacket(Packet[At]):
    """
    Represents the output of an action selection routine.

    Contains information about the selected actions and strengths of actionable
    chunks.
    """

    def __init__(
        self, 
        kvpairs: Mapping[Node, At] = None,
        chosen: Set[Chunk] = None
    ) -> None:
        '''
        Initialize a ``DecisionPacket`` instance.

        :param kvpairs: Strengths of actionable chunks.
        :param chosen: The set of actions to be fired.
        '''

        super().__init__(kvpairs)
        self.chosen = chosen

    def __eq__(self, other: Any) -> bool:

        if (
            super().__eq__(other) and
            self.chosen == other.chosen
        ):
            return True
        else:
            return False

    def __repr__(self) -> str:
        
        repr_ = ''.join(
            [
                type(self).__name__, 
                '(',
                super(Packet, self).__repr__(),
                ', ',
                'chosen=' + repr(self.chosen),
                ')'
            ]
        )
        return repr_
