"""Tools for representing information about Clarion nodes."""

from abc import abstractmethod
from enum import Enum, auto
from typing import MutableMapping, TypeVar, Hashable, Mapping, Set, Any, Iterable, Callable, cast
from collections import UserDict
from pyClarion.base.symbols import Node, Chunk, FlowType


At = TypeVar("At")


class Packet(MutableMapping[Node, At]):
    """
    Base class for encapsulating information about nodes.

    Takes one type variable, ``At``, which is an unrestricted type variable 
    denoting the expected type for data values.
    """

    pass


class Level(Enum):

    TopLevel = auto()
    BottomLevel = auto()


class ActivationPacket(dict, Packet[At]):
    """
    A class for representing node activations.

    Default activation values may be implemented by overriding the 
    ``default_activation`` method. When defined, default activations are handled 
    similarly to ``collections.defaultdict``.

    The precise type of an ``ActivationPacket`` instance may encode important 
    metadata, such as information about the source of the packet. 

    See module documentation for further details and examples.
    """

    def __init__(
        self, 
        kvpairs: Mapping[Node, At] = None,
        default_factory: Callable[[Node], At] = None,
        origin: Level = None
    ) -> None:
        '''
        Initialize a ``DecisionPacket`` instance.

        :param kvpairs: Strengths of actionable chunks.
        :param chosen: The set of actions to be fired.
        '''

        super().__init__()
        if kvpairs:
            self.update(kvpairs)
        self.default_factory = default_factory
        self.origin = origin

    def __repr__(self) -> str:
        
        repr_ = ''.join(
            [
                type(self).__name__,
                '(',
                super().__repr__(),
                ", ",
                "default_factory=",
                repr(self.default_factory),
                ", ",
                "origin=",
                repr(self.origin),
                ')'
            ]
        )
        return repr_

    def __missing__(self, key: Node) -> At:

        if self.default_factory:
            value : At = self.default_factory(key)
            self[key] = value
            return value
        else:
            raise KeyError(key)

    def subpacket(self, nodes: Iterable[Node]):
        """Return a subpacket containing activations for ``nodes``."""
        
        return type(self)(
            {node: self[node] for node in nodes}, 
            default_factory=self.default_factory, 
            origin=self.origin
        )


class DecisionPacket(dict, Packet[At]):
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

        super().__init__()
        if kvpairs:
            self.update(kvpairs)
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
                super().__repr__(),
                ', ',
                'chosen=' + repr(self.chosen),
                ')'
            ]
        )
        return repr_
