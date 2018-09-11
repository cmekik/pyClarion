"""
Tools for capturing fundamental representational constructs.

The most basic representational construct in Clarion is the connectionist node: 
an individual unit in a network that may receive activation from and propagate 
activation to other units.

Nodes
===== 

In Clarion, there are two essential kinds of node: microfeatures and chunks, 
which are represented by the ``Microfeature`` and ``Chunk`` classes 
respectively.

Both of these classes are implemeted as frozen dataclasses. 

Examples
--------

Below, a microfeature node representing the dimension-value pair ``('color', 
'red')`` and a chunk node with id ``1234`` are defined:

>>> mf = Microfeature(dim='color', val='red')
>>> ch = Chunk(id=1234)

The ``Microfeature`` and ``Chunk`` classes are subclasses of the Node class. 

>>> isinstance(mf, Node)
True
>>> isinstance(ch, Node)
True

``Node`` objects should be used as ``Mapping`` keys. Their intended role 
is to allow easy and uniform retrieval of information related to the nodes they 
represent within the theoretical context of a particular model.

Equality Checking
~~~~~~~~~~~~~~~~~

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
~~~~~~~~~~

``Microfeature`` and ``Chunk`` instances are frozen. Attempting to set new 
or existing attributes will cause an error:

>>> mf.dim = 'code'
Traceback (most recent call last):
    ...
dataclasses.FrozenInstanceError: cannot assign to field 'dim'
>>> ch.my_attribute = 'some data'
Traceback (most recent call last):
    ...
dataclasses.FrozenInstanceError: cannot assign to field 'my_attribute'

However, it is possible to subclass ``Microfeature`` or ``Chunk`` in order to 
add additional fields, as shown below.

>>> from typing import Hashable
>>> @dataclasses.dataclass(frozen=True)
... class MyMicrofeature(Microfeature):
...     meta : Hashable
>>> MyMicrofeature('color', 'red', 'some metadata')
MyMicrofeature(dim='color', val='red', meta='some metadata')

Node Objects
~~~~~~~~~~~~

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

Flows
=====

Explanation required...

"""


import typing as t
import dataclasses
import enum


###########
# CLASSES #
###########

@dataclasses.dataclass(init=True, repr=True, eq=False, frozen=True)
class Attribution(object):
    """
    A generic connectionist node.

    Although this class may be used on its own, it is intended for use as a base
    class for more specific node types.
    """
    
    pass


@dataclasses.dataclass(init=True, repr=True, eq=False, frozen=True)
class Node(Attribution):
    """
    A generic connectionist node.

    Although this class may be used on its own, it is intended for use as a base
    class for more specific node types.
    """
    
    pass


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class Microfeature(Node):
    """
    A microfeature node.

    Microfeatures are implicit, connectionist representations. They represent
    dimension-value pairs.

    Microfeature objects are frozen dataclasses that compare equal iff the
    contents of their data fields are equal.

    See module documentation for details and examples.
    """

    dim: t.Hashable
    val: t.Hashable


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class Chunk(Node):
    """
    A chunk node.

    Chunks are explicit, localist representations. They represent individual
    concepts.

    Chunk objects are frozen dataclasses that compare equal iff the contents of
    their data fields are equal.

    See module documentation for details and examples.
    """

    id: t.Hashable


class Plicity(enum.Enum):
    """
    Enumerates of possible knowledge orientations.

    Explicit - Folded-out
    Implicit - Folded-in
    Deplicit - Un-folded
    Abplicit - Folded-away
    """

    Explicit = enum.auto()
    Implicit = enum.auto()
    Deplicit = enum.auto()
    Abplicit = enum.auto()


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class Flow(Attribution):
    """
    A body of knowledge represented as an activation flow.
    """

    id: t.Hashable
    plicity: Plicity


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class Appraisal(Attribution):
    """
    A body of knowledge represented as an activation flow.
    """

    id: t.Hashable


################
# TYPE ALIASES #
################


Dim2Num = t.Dict[t.Hashable, float]
FeatureSet = t.Set[Microfeature]
NodeSet = t.Set[Node]
ChunkSet = t.Set[Chunk]


#############
# FUNCTIONS #
#############


def get_nodes(*node_iterables: t.Iterable[Node]) -> t.Set[Node]:
    """
    Construct the set of all nodes in a set of node containers.

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
