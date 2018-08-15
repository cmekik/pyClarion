'''
This module provides tools for recording and manipulating node activations.

Usage
=====

The main construct exported by this module is the ``ActivationPacket`` class.

Instantiation
-------------

``ActivationPacket`` is an abstract class with one abstract method called 
``default_activation``; it cannot be directly instanced.

>>> ActivationPacket()
Traceback (most recent call last):
    ...
TypeError: Can't instantiate abstract class ActivationPacket with abstract methods default_activation

In other words, ``ActivationPacket`` must be subclassed before use. 

>>> class MyPacket(ActivationPacket):
...     def default_activation(self, key):
...         return 0.0
...
>>> MyPacket()
MyPacket({})

Basic Behavior
--------------

An ``ActivationPacket`` instance behaves mostly like a like a ``dict`` object.

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

In fact, all methods available to ``dict`` are also available to 
``ActivationPacket``.

Default Behavior
----------------

Unlike regular dicts, a ``KeyError`` is not raised when an ``ActivationPacket`` 
receives an unknown key. In this case, ``ActivationPacket`` objects behave like 
``collections.defaultdict`` objects: they output a default value and record 
the new ``(key, value)`` pair.

>>> n3 = Node()
>>> n3 in p
False
>>> p[n3]
0.0
>>> n3 in p
True

Default values are provided by the ``default_activation`` method. This method 
can be set to return different default values for different nodes.

>>> from pyClarion.base.node import Microfeature, Chunk
>>> class MySubtlePacket(ActivationPacket):
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

The type of an ```ActivationPacket``` is meaningful. Different activation 
sources may output packets of different types. For instance, a top-down 
activation cycle should output an instance of ``TopDownPacket``.

>>> class MyTopDownPacket(TopDownPacket, MyPacket):
...     """Represents the output of a top-down activation cycle.
...     """
...     pass
... 
>>> p = MyTopDownPacket()
>>> isinstance(p, MyPacket)
True
>>> isinstance(p, MyTopDownPacket)
True

This module provides generic types for major activation flow types that can be 
found within Clarion. These basic types are meant to promote more precise 
definition of higher-level constructs (i.e., allow for more specific signature 
declarations), reduce coupling between pyClarion objects, and facilitate 
conditional processing of activations.
'''

from abc import abstractmethod
from typing import MutableMapping, TypeVar
from collections import UserDict
from ..node import Node


###############
# ABSTRACTION #
###############


T = TypeVar("T")

class ActivationPacket(UserDict, MutableMapping[Node, T]):
    """An abstract class for representing node activations.

    Has type ``MutableMapping[pyClarion.base.node.Node, T]``, where ``T`` is an 
    unrestricted type variable.

    Nodes not contained in an ActivationPacket object are assumed to be at a 
    default activation level. Default activation levels are defined by the 
    ActivationPacket.default_activation method and are handled similarly to 
    ``collections.defaultdict``.
    
    The precise type of an ActivationPacket may encode important metadata. 

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


###############################
# BASE ActivationPacket TYPES #
###############################


class TopDownPacket(ActivationPacket):
    """An activation packet resulting from a top-down activation flow.
    """
    pass


class BottomUpPacket(ActivationPacket):
    """An activation packet resulting from a bottom-up activation flow.
    """
    pass


class TopLevelPacket(ActivationPacket):
    """An activation packet resulting from a top-level activation flow.
    """
    pass


class BottomLevelPacket(ActivationPacket):
    """An activation packet resulting from a bottom-level activation flow.
    """
    pass


if __name__ == '__main__':
    import doctest
    doctest.testmod()