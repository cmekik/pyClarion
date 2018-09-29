"""Tools for representing information about Clarion nodes."""

from abc import abstractmethod
from enum import Enum, auto
from typing import (
    MutableMapping, TypeVar, Hashable, Mapping, Set, Any, Iterable, Callable, 
    List, Optional, cast
)
from collections import UserDict
from pyClarion.base.enums import Level
from pyClarion.base.symbols import Node, Chunk, FlowType


At = TypeVar("At")


class ActivationPacket(dict, MutableMapping[Node, At]):
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
        default_factory: Callable[[Optional[Node]], At] = None,
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
        
        return ''.join(self._repr())

    def __missing__(self, key: Node) -> At:

        if self.default_factory:
            value : At = self.default_factory(key)
            self[key] = value
            return value
        else:
            raise KeyError(key)

    def copy(self):

        return self.subpacket(self.keys())

    def subpacket(self, nodes: Iterable[Node]):
        """Return a subpacket containing activations for ``nodes``."""
        
        return type(self)(
            {node: self[node] for node in nodes}, 
            default_factory=self.default_factory, 
            origin=self.origin
        )

    def _repr(self) -> List[str]:

        repr_ = [
            type(self).__name__,
            '(',
            super().__repr__(),
            ", ",
            "origin=",
            repr(self.origin),
            ", ",
            "default_factory=",
            repr(self.default_factory),
            ")"
        ]
        return repr_


class DecisionPacket(ActivationPacket[At]):
    """
    Represents the output of an action selection routine.

    Contains information about the selected actions and strengths of actionable
    chunks.
    """

    def __init__(
        self, 
        kvpairs: Mapping[Node, At] = None,
        default_factory: Callable[[Optional[Node]], At] = None,
        origin: Level = None,
        chosen: Set[Chunk] = None
    ) -> None:
        '''
        Initialize a ``DecisionPacket`` instance.

        :param kvpairs: Strengths of actionable chunks.
        :param chosen: The set of actions to be fired.
        '''

        super().__init__(kvpairs, default_factory, origin)
        self.chosen = chosen

    def __eq__(self, other: Any) -> bool:

        return (
            super().__eq__(other) and
            self.chosen == other.chosen
        )

    def _repr(self) -> List[str]:

        repr_ = super()._repr()
        supplement = [
            "chosen=",
            repr(self.chosen),
            ", "
        ]
        return repr_[:-3] + supplement + repr_[-3:]
        