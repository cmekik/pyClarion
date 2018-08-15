'''This module captures the fundamental representational constructs of Clarion.

The most basic representational construct in Clarion is the connectionist node: 
an individual unit in a network that may receive activation from and propagate 
activation to other units. In Clarion, there are two essential kinds of node: 
microfeatures and chunks. 

This module provides the Microfeature and Chunk classes, implemented as frozen 
dataclasses, for representing microfeatures and chunks respectively.

Usage
=====

Create Microfeature and Chunk instances to represent nodes known to a 
Clarion agent. Below, a microfeature node representing the dimension-value pair
``('color', 'red')`` and a chunk node with id ``1234`` are defined:

>>> mf = Microfeature(dim='color', val='red')
>>> ch = Chunk(id=1234)

The Microfeature and Chunk classes are subclasses of the Node class. 

>>> isinstance(mf, Node)
True
>>> isinstance(ch, Node)
True

Equality Checking
-----------------

Microfeature and Chunk instances compare equal iff the contents of their data 
fields are equal. Below, ``mf`` compares equal to a new Microfeature instance 
that happens to be initialized with the same dimension and value. Likewise, 
``ch`` compares equal to a new Chunk that happens to be initialized with the 
same id.

>>> mf == Microfeature('color', 'red')
True
>>> ch == Chunk(1234)
True

Here are a few false equality comparisons:

>>> mf == Microfeature('color', 'blue')
False
>>> mf == Microfeature('code', 'red')
False
>>> ch == Chunk('COLOR')
False

Attributes
----------

Since Microfeature and Chunk instances are frozen, attempting to set new 
or existing attributes will cause an error:

>>> mf.dim = 'code'
Traceback (most recent call last):
    ...
dataclasses.FrozenInstanceError: cannot assign to field 'dim'
>>> ch.my_attribute = 'some data'
Traceback (most recent call last):
    ...
dataclasses.FrozenInstanceError: cannot assign to field 'my_attribute'

However, it is possible to subclass Microfeature or Chunk in order to add 
additional fields, as shown below.

>>> from typing import Hashable
>>> @dataclasses.dataclass(frozen=True)
... class MyMicrofeature(Microfeature):
...     meta : Hashable
>>> MyMicrofeature('color', 'red', 'some metadata')
MyMicrofeature(dim='color', val='red', meta='some metadata')

Node Objects
------------

The ``Node`` class is included as a base class for ``Microfeature`` and 
``Chunk`` and as a convenience for possible future extensions to Clarion. It is
not intended to be directly used. Nevertheless, ``Node`` objects can be created
just like ``Microfeature`` or ``Chunk`` objects, but behave rather differently.

Most importantly ``Node`` objects have no data fields.

>>> n = Node()

Consequently, ``Node`` equality is instance-based. Below, the node ``n`` 
compares equal to itself but not to another newly created node.

>>> n == n
True
>>> n == Node()
False
'''


import typing as T
import dataclasses


###########
# CLASSES #
###########


@dataclasses.dataclass(init=True, repr=True, eq=False, frozen=True)
class Node(object):
    '''A generic connectionist node.

    Although this class may be used on its own, it is intended for use as a base 
    class for more specific node types.
    '''
    
    pass


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class Microfeature(Node):
    '''A microfeature node.

    Microfeatures are implicit, connectionist representations. They represent 
    dimension-value pairs.

    Microfeature objects are frozen dataclasses that compare equal iff the 
    contents of their data fields are equal.

    See module documentation for details and examples.
    '''

    dim : T.Hashable
    val : T.Hashable


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class Chunk(Node):
    '''A chunk node. 

    Chunks are explicit, localist representations. They represent individual 
    concepts.

    Chunk objects are frozen dataclasses that compare equal iff the contents of 
    their data fields are equal.

    See module documentation for details and examples.
    '''

    id : T.Hashable


################
# TYPE ALIASES #
################


Dim2Num = T.Dict[T.Hashable, float]
FeatureSet = T.Set[Microfeature]
NodeSet = T.Set[Node]
ChunkSet = T.Set[Chunk]


#############
# FUNCTIONS #
#############


def get_nodes(*node_iterables : T.Iterable[Node]) -> T.Set[Node]:
    """Construct the set of all nodes in a set of node containers.

    Usage example:

    >>> l = [Chunk(1234), Microfeature('color', 'red')]
    >>> s = {Chunk('COLOR'), Chunk(1234)}
    >>> d = {
    ...     Microfeature('color', 'red'): 1., 
    ...     Microfeature('color', 'blue'): .5
    ... }
    >>> get_nodes(l, s, d) == {
    ...     Chunk(id=1234), 
    ...     Microfeature(dim='color', val='red'), 
    ...     Chunk(id='COLOR'),
    ...     Microfeature('color', 'blue')
    ... }    
    True

    :param node_iterables: A sequence of iterables containing nodes.
    """

    node_set = set()
    for node_iterable in node_iterables:
        for node in node_iterable:
            node_set.add(node)
    return node_set


if __name__ == "__main__":
    import doctest
    doctest.testmod()