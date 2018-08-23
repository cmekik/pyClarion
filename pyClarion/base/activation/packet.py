'''
This module provides tools for representing activation patterns in pyClarion.

Usage
=====

This module exports the ``BaseActivationPacket`` class, which is a base class 
for representing collections of node activations.

Instantiation
-------------

``BaseActivationPacket`` is an abstract class with one abstract method called 
``default_activation``; it cannot be directly instanced.

>>> BaseActivationPacket()
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class BaseActivationPacket with abstract methods default_activation

In other words, ``BaseActivationPacket`` must be subclassed before use. 

>>> class MyPacket(BaseActivationPacket):
...     def default_activation(self, key):
...         return 0.0
...
>>> MyPacket()
MyPacket({})

Basic Behavior
--------------

An ``BaseActivationPacket`` instance behaves mostly like a ``dict`` object.

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
``BaseActivationPacket``.

Default Behavior
----------------

Unlike regular dicts, a ``KeyError`` is not raised when an ``BaseActivationPacket`` 
receives an unknown key. ``BaseActivationPacket`` objects handle unknown keys like 
``collections.defaultdict`` objects: they output a default value and record 
the new ``(key, value)`` pair. However, unlike ``collections.defaultdict`` 
objects, default values for ``BaseActivationPacket`` objects are provided by the 
``default_activation`` method. 

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
>>> class MySubtlePacket(BaseActivationPacket):
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

Strictly speaking, a top-down activation flow should drive ``Microfeature`` 
activations based on ``Chunk`` activations. For simplicity, the example above 
omits this detail.
'''

from abc import abstractmethod
from typing import MutableMapping, TypeVar
from collections import UserDict
from pyClarion.base.node import Node


T = TypeVar("T")


class BaseActivationPacket(UserDict, MutableMapping[Node, T]):
    """An abstract class for representing node activations.

    Has type ``MutableMapping[pyClarion.base.node.Node, T]``, where ``T`` is an 
    unrestricted type variable. It is generally expected that ``T`` will be some 
    numerical type such as ``float``, however this expectation is not enforced. 
    Violate this at your own risk.

    Nodes not contained in an BaseActivationPacket object are assumed to be at a 
    default activation level. Default activation levels are defined by the 
    BaseActivationPacket.default_activation method and are handled similarly to 
    ``collections.defaultdict``.

    The precise type of an ``BaseActivationPacket`` instance may encode important 
    metadata. 

    See module documentation for further details and examples.
    """

    def __repr__(self) -> str:
        return super().__repr__().join([type(self).__name__ + '(', ')'])

    def __missing__(self, key : Node) -> T:
        self[key] = value = self.default_activation(key)
        return value

    @abstractmethod
    def default_activation(self, key : Node) -> T:
        '''Return designated default value for the given input.
        '''
        pass


if __name__ == '__main__':
    import doctest
    doctest.testmod()