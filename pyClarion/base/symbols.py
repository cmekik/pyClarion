"""Tools for naming and indexing therotically relevant constructs."""

from typing import Hashable, Iterable, Set
from dataclasses import dataclass
from pyClarion.base.enums import FlowType


@dataclass(init=True, repr=True, eq=False, frozen=True)
class ConstructSymbol(object):
    """Represents a theoretical construct."""

    pass


class BasicConstructSymbol(ConstructSymbol):
    """Represents a basic theoretical construct."""

    pass


class ContainerConstructSymbol(ConstructSymbol):
    """Represents a theoretical construct owning other constructs."""

    pass


class Node(BasicConstructSymbol):
    """Represents a generic node."""
    
    pass


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Microfeature(Node):
    """Represents a microfeature node."""

    dim: Hashable
    val: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Chunk(Node):
    """Represents a chunk node."""

    id: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Flow(BasicConstructSymbol):
    """Represents a body of knowledge."""

    id: Hashable
    flow_type: FlowType


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Appraisal(BasicConstructSymbol):
    """Represents a class of judgments and/or decisions."""

    id: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Buffer(BasicConstructSymbol):
    """Represents an activation buffer."""

    id: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Actions(BasicConstructSymbol):
    """Represents actions available to an agent."""

    id: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Subsystem(ContainerConstructSymbol):
    """Represents a functionally distinct section of a cognitive apparatus."""

    id: Hashable


@dataclass(init=True, repr=True, eq=True, frozen=True)
class Agent(ContainerConstructSymbol):
    """Represents a Clarion agent."""

    id: Hashable
