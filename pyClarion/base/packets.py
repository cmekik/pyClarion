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
DefaultActivation = Callable[[Optional[Node]], At]

class ActivationPacket(dict, MutableMapping[Node, At]):
    """A class for representing node activations."""

    def __init__(
        self, 
        kvpairs: Mapping[Node, At] = None,
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
        self.origin = origin

    def __repr__(self) -> str:
        
        return ''.join(self._repr())

    def copy(self):

        return self.subpacket(self.keys())

    def subpacket(
        self, 
        nodes: Iterable[Node], 
        default_activation: DefaultActivation = None
    ) -> 'ActivationPacket':
        """Return a subpacket containing activations for ``nodes``."""
        
        mapping: dict = self._subpacket(nodes, default_activation)
        origin = self.origin
        output: 'ActivationPacket[At]' = ActivationPacket(mapping, origin)
        return output

    def _repr(self) -> List[str]:

        repr_ = [
            type(self).__name__,
            '(',
            super().__repr__(),
            ", ",
            "origin=",
            repr(self.origin),
            ")"
        ]
        return repr_

    def _subpacket(
        self, 
        nodes: Iterable[Node], 
        default_activation: DefaultActivation = None
    ) -> dict:

        output: dict = {}
        for node in nodes:
            if node in self:
                activation = self[node]
            elif default_activation:
                activation = default_activation(node)
            else:
                raise KeyError(
                    "Node {} not in self".format(str(node))
                )
            output[node] = activation
        return output


class DecisionPacket(ActivationPacket[At]):
    """
    Represents the output of an action selection routine.

    Contains information about the selected actions and strengths of actionable
    chunks.
    """

    def __init__(
        self, 
        kvpairs: Mapping[Node, At] = None,
        origin: Level = None,
        chosen: Set[Chunk] = None
    ) -> None:
        '''
        Initialize a ``DecisionPacket`` instance.

        :param kvpairs: Strengths of actionable chunks.
        :param chosen: The set of actions to be fired.
        '''

        super().__init__(kvpairs, origin)
        self.chosen = chosen

    def __eq__(self, other: Any) -> bool:

        return (
            super().__eq__(other) and
            self.chosen == other.chosen
        )

    def _repr(self) -> List[str]:

        repr_ = super()._repr()
        supplement = [
            ", ",
            "chosen=",
            repr(self.chosen),
        ]
        return repr_[:-1] + supplement + repr_[-1:]

    def subpacket(
        self, 
        nodes: Iterable[Node], 
        default_activation: DefaultActivation = None
    ) -> 'DecisionPacket[At]':
        """Return a subpacket containing activations for ``nodes``."""
        
        mapping = self._subpacket(nodes, default_activation)
        origin = self.origin
        chosen = self.chosen
        output: 'DecisionPacket[At]' = DecisionPacket(
            mapping, origin, chosen
        )
        return output
        