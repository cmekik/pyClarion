"""
Tools for naming, identifying, and indexing therotically relevant constructs.
"""

from typing import Hashable, Iterable, Set
from dataclasses import dataclass
from enum import Enum, auto


######################
### SYMBOL CLASSES ###
######################


@dataclass(init=True, repr=True, eq=False, frozen=True)
class ConstructSymbol(object):
    """Represents some theoretical construct."""

    pass


class BasicConstructSymbol(ConstructSymbol):
    """Represents some basic theoretical construct."""

    pass


class CompositeConstructSymbol(ConstructSymbol):
    """Represents some theoretical construct owning other constructs."""

    pass


### NODE SYMBOLS ###


class Node(BasicConstructSymbol):
    """
    A generic connectionist node.

    Represents a distinct piece of knowledge such as a chunk or a microfeature. 
    Intended for use as a common base class for Microfeature and Chunk classes.
    """
    
    pass


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Microfeature(Node):
    """
    A microfeature node.

    Microfeatures are implicit, connectionist representations. They represent
    dimension-value pairs.
    """

    dim: Hashable
    val: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Chunk(Node):
    """
    A chunk node.

    Chunks are explicit, localist representations. They represent individual
    concepts.
    """

    id: Hashable


### FLOW SYMBOLS ###


class FlowType(Enum):
    """An enumeration of level types."""

    TopLevel = auto()
    BottomLevel = auto()
    TopDown = auto()
    BottomUp = auto()


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Flow(BasicConstructSymbol):
    """
    A body of agent knowledge in the form of an activation flow.
    """

    id: Hashable
    flow_type: FlowType


### APPRAISAL SYMBOLS ###


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Appraisal(BasicConstructSymbol):
    """
    A class of judgments and/or decisions an agent can make.
    """

    id: Hashable


### ACTIVITY SYMBOLS ###


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Activity(BasicConstructSymbol):
    """
    A class of things that an agent can do.
    """

    id: Hashable


### BUFFER SYMBOLS ###


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Memory(BasicConstructSymbol):
    """
    A memory buffer.
    """

    id: Hashable


### SUBSYSTEM SYMBOLS ###


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Subsystem(CompositeConstructSymbol):
    """
    A functionally distinct piece of an agent's cognitive apparatus.
    """

    id: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Agent(CompositeConstructSymbol):
    """
    A Clarion agent.
    """

    id: Hashable


#############
# FUNCTIONS #
#############


def get_nodes(*node_iterables: Iterable[Node]) -> Set[Node]:
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
