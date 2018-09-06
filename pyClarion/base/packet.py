'''
This module provides tools for representing activation patterns in pyClarion.

Usage
=====

This module exports the ``BaseActivationPacket`` class, which is a base class 
for representing collections of node activations.

Instantiation
-------------

A ``ActivationPacket`` base activation packet behaves like a ``dict``. 

>>> ActivationPacket()
ActivationPacket({})

The ``ActivationPacket`` class provides a ``default_activation`` method, which 
may be overridden to capture assumptions about default activation values. 

>>> class MyPacket(ActivationPacket[float]):
...     def default_activation(self, key):
...         return 0.0
...
>>> MyPacket()
MyPacket({})
>>> MyPacket()[Node()]
0.0

In the example above, a type parameter is passed to the base class 
``ActivationPacket``. This is an optional step. In standard python, it should 
have no noticeable effect on the program, but may serve to clarify the intended 
usage of the defined subclass. If type-checking is enabled, it may be used as a 
regular type parameter.

Basic Behavior
--------------

An ``ActivationPacket`` instance behaves mostly like a ``dict`` object.

>>> n1, n2 = Node(), Node()
>>> p = MyPacket({n1 : 0.3})
>>> p[n1]
0.3
>>> p[n1] = 0.6
>>> p[n1]
0.6
>>> p[n2] = 0.2
>>> p[n2]
0.2

In fact, almost all methods available to ``dict`` are also available to 
``ActivationPacket``.

Default Behavior
----------------

When a default value is provided by ``default_activation``, ``ActivationPacket`` 
objects handle unknown keys like ``collections.defaultdict`` objects: they 
output a default value and record the new ``(key, value)`` pair.

>>> n3 = Node()
>>> n3 in p
False
>>> p[n3]
0.0
>>> n3 in p
True

The ``default_activation`` method can be set to return different default values 
for different nodes.

>>> from pyClarion.base.node import Microfeature, Chunk
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

Packet Types
------------

The precise type of an ```BaseActivationPacket``` is meaningful. Different 
activation sources may output packets of different types. For instance, a 
top-down activation cycle may output an instance of ``TopDownPacket``, as 
illustrated in the example below.

>>> class MyTopDownPacket(MyPacket):
...     """Represents the output of a top-down activation cycle.
...     """
...     pass
... 
>>> def my_top_down_activation_cycle(packet):
...     """A dummy top-down activation cycle for demonstration purposes""" 
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
'''

from abc import abstractmethod
from typing import MutableMapping, TypeVar
from collections import UserDict
from pyClarion.base.node import Node


T = TypeVar("T")


class ActivationPacket(UserDict, MutableMapping[Node, T]):
    """A class for representing node activations.

    Has type ``MutableMapping[pyClarion.base.node.Node, T]``, where ``T`` is an 
    unrestricted type variable. It is generally expected that ``T`` will be some 
    numerical type such as ``float``, however this expectation is not enforced. 
    Violate this at your own risk.

    By default ``ActivationPacket`` objects raise an exception when given an 
    unknown key. However, if there is a theoretically driven default activation 
    level, it can be implemented by overriding the ``default_activation`` 
    method. Default activations are handled similarly to 
    ``collections.defaultdict``.

    The precise type of an ``ActivationPacket`` instance may encode important 
    metadata. 

    See module documentation for further details and examples.
    """

    def __repr__(self) -> str:
        
        return super().__repr__().join([type(self).__name__ + '(', ')'])

    def __missing__(self, key : Node) -> T:

        self[key] = value = self.default_activation(key)
        return value

    def default_activation(self, key : Node) -> T:
        '''Return designated default value for the given input.
        '''
        
        raise KeyError


if __name__ == '__main__':
    import doctest
    doctest.testmod()