import typing as t
import dataclasses
import enum


######################
### SYMBOL CLASSES ###
######################


@dataclasses.dataclass(init=True, repr=True, eq=False, frozen=True)
class ConstructSymbol(object):
    """Represents some theoretical construct."""

    pass


@dataclasses.dataclass(init=True, repr=True, eq=False, frozen=True)
class BasicConstructSymbol(ConstructSymbol):
    """Represents some basic theoretical construct."""

    pass


@dataclasses.dataclass(init=True, repr=True, eq=False, frozen=True)
class CompositeConstructSymbol(ConstructSymbol):
    """Represents some theoretical construct owning other constructs."""

    pass


### NODE SYMBOLS ###


@dataclasses.dataclass(init=True, repr=True, eq=False, frozen=True)
class Node(BasicConstructSymbol):
    """
    A generic unit of knowledge.

    Symbol for a distinct piece of knowledge such as a chunk or a microfeature. 
    Intended for use as a common base class for Microfeature and Chunk classes.
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


### FLOW SYMBOLS ###


class FlowType(enum.Enum):
    """An enumeration of level types."""

    TopLevel = enum.auto()
    BottomLevel = enum.auto()
    TopDown = enum.auto()
    BottomUp = enum.auto()


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class Flow(BasicConstructSymbol):
    """
    A body of agent knowledge in the form of an activation flow.
    """

    id: t.Hashable
    flow_type: FlowType


### APPRAISAL SYMBOLS ###


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class Appraisal(BasicConstructSymbol):
    """
    A class of judgments and/or decisions an agent can make.
    """

    id: t.Hashable


### ACTIVITY SYMBOLS ###


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class Activity(BasicConstructSymbol):
    """
    A class of things that an agent can do.
    """

    id: t.Hashable


### MEMORY SYMBOLS ###


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class Memory(BasicConstructSymbol):
    """
    A memory buffer.
    """

    id: t.Hashable


### SUBSYSTEM SYMBOLS ###


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class Subsystem(CompositeConstructSymbol):
    """
    A functionally distinct piece of an agent's cognitive apparatus.
    """

    id: t.Hashable


### AGENT SYMBOLS ###


@dataclasses.dataclass(init=True, repr=True, eq=True, frozen=True)
class Agent(CompositeConstructSymbol):
    """
    A Clarion agent.
    """

    id: t.Hashable


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
