"""Tools for naming and indexing therotically relevant constructs."""

from typing import Hashable, Iterable, Set
from dataclasses import dataclass
from pyClarion.base.enums import FlowType


@dataclass(init=True, repr=True, eq=False, frozen=True)
class ConstructSymbol(object):
    """Generic symbol for a theoretical construct."""

    pass


@dataclass(init=True, repr=True, eq=False, frozen=True)
class BasicConstructSymbol(ConstructSymbol):
    """Symbol for a basic theoretical construct."""

    pass


@dataclass(init=True, repr=True, eq=False, frozen=True)
class ContainerConstructSymbol(ConstructSymbol):
    """Symbol for a theoretical construct owning other constructs."""

    pass


@dataclass(init=True, repr=True, eq=False, frozen=True)
class Node(BasicConstructSymbol):
    """Symbol for a generic connectionist node."""
    
    pass


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Microfeature(Node):
    """
    Symbol for a microfeature node.

    Microfeature nodes represent implicit knowledge. In Clarion, they are 
    characterized by a dimension-value pair (dv-pair). Microfeatures that share 
    the same dimension entry are treated as alternatives. 
    """

    dim: Hashable
    val: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Chunk(Node):
    """Symbol for a chunk node."""

    id: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Flow(BasicConstructSymbol):
    """Symbol for an activation flow."""

    id: Hashable
    flow_type: FlowType


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Appraisal(BasicConstructSymbol):
    """Symbol for a class of judgments and/or decisions."""

    id: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Behavior(BasicConstructSymbol):
    """Symbol for actions available to an agent."""

    id: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Buffer(BasicConstructSymbol):
    """Symbol for an activation buffer."""

    id: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Subsystem(ContainerConstructSymbol):
    """Symbol for a functionally distinct section of a cognitive apparatus."""

    id: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Agent(ContainerConstructSymbol):
    """Symbol for a Clarion agent."""

    id: Hashable
