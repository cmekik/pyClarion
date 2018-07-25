"""This module provides abstractions for constructing Clarion subjects.

Subsystems house and manage functionally distinct collections of activation 
channels. They control activation cycles and learning.
"""


import abc
import typing as T

from . import node
from . import activation


class Statistic(abc.ABC):
    """Tracks a statistic.

    Contains and updates some relevant statistic(s). Does not store additional 
    information about related constructs, such as references to the objects of 
    stored statistics.
    """
    pass


class Component(abc.ABC):
    pass


class Subsystem(abc.ABC):
    """A Clarion subsystem.
    """

    @property
    @abc.abstractmethod
    def channels(self) -> activation.ChannelSet:
        pass

    @property
    @abc.abstractmethod
    def components(self) -> T.Set[Component]:
        pass

    @property
    @abc.abstractmethod
    def actions(self) -> node.Chunk2Callable:
        pass

    @abc.abstractmethod
    def __call__(
        self, 
        input_map : node.Node2Float
    ) -> None:
        pass


class Subject(object):

    def __call__(self, input_map : node.Node2Float) -> None:
        pass


def execute_actions(
    actionable_chunks : node.ChunkSet, action_map : node.Chunk2Callable
): 
    """Execute actionable chunks.
    """

    for chunk in actionable_chunks:
        # Try executing action attached to chunk. If it fails move on.
        try:
            action_map[chunk]()
        except KeyError:
            continue