"""This module provides abstractions for constructing Clarion subjects.

WARNING:
    At this point in time, most classes defined here are very experimental. The 
    abstractions may change drastically. 
"""


import abc
import typing as T

from . import node
from . import activation


####### ABSTRACTIONS #######

class Statistic(abc.ABC):
    """Tracks a statistic.

    Contains and updates some relevant statistic(s). Does not store additional 
    information about related constructs, such as references to the objects of 
    stored statistics.
    """
    pass

class ActionHandler(abc.ABC):
    """An abstraction for linking actionable chunks to appropriate behavior.
    """

    @abc.abstractmethod
    def __call__(self, actionable_chunks : node.ChunkSet) -> None:
        """Execute actions associated with given actionable chunks.

        kwargs:
            actionable_chunks: A set of chunks selected for action execution.
        """
        pass

class Component(abc.ABC):
    """An abstraction for managing some class of activation channels associated 
    with a subsystem.

    Components are abstractions meant to capture learning and forgetting 
    routines. They monitor the activity of the subsystem to which they belong 
    and modify its members (channels and/or parameters).
    """
    pass

class Subsystem(abc.ABC):
    """A Clarion subsystem.
    """

    @property
    @abc.abstractmethod
    def channels(self) -> activation.ChannelSet:
        """A set of activation channels representing top- and bottom-level 
        knowledge stored in self.
        """
        pass

    @property
    @abc.abstractmethod
    def components(self) -> T.Set[Component]:
        """A set of subsystem components for handling learning, forgetting and 
        other changes to the knowledge-base of the subject.
        """
        pass

    @property
    @abc.abstractmethod
    def actions(self) -> node.Chunk2Callable:
        """An object binding chunks to actions executable by this subsystem.
        """
        pass

    @abc.abstractmethod
    def __call__(
        self, 
        input_map : node.Node2Float
    ) -> None:
        """Run through one processing cycle of self.
        """
        pass

class Subject(object):
    """An abstraction for representing Clarion agents.

    Subject objects facilitate the interface between subsystems and the 
    environment. The main responsibility of these objects is to distribute 
    sensory input to subsystems. They also serve to bind together all 
    subsystems associated with a given subject.

    It may also be useful to define action callbacks affecting the environment 
    as methods of this class. Action-centered subsystems would be passed sets 
    of these methods as the callbacks to execute following an action decision. 
    Internal actions may be defined within the relevant subsystem class 
    definition and passed to relevant subsystems in the same way.
    """

    @abc.abstractmethod
    def __call__(self, input_map : node.Node2Float) -> None:
        """Receive and process a new set of sensory/world information.
        """
        pass


####### FUNCTIONS #######

def max_strength(
    selected : node.ChunkSet, activations : node.Node2Float
) -> float:
    """Returns maximum strength among selected chunks.

    If no chunks are selected, returns 0.

    kwargs:
        selected : Selected chunks.
        activations : Chunk activations.
    """
    try:
        return max([activations[chunk] for chunk in selected])
    except ValueError:
        return 0. 

def execute_actions(
    actionable_chunks : node.ChunkSet, action_map : node.Chunk2Callable
): 
    """Execute actionable chunks.

    Action execution used to be handled by a dedicated class. In hindsight, 
    this was probably a better design. It will be readopted in a later version.
    """

    for chunk in actionable_chunks:
        # Try executing action attached to chunk. If it fails move on.
        try:
            action_map[chunk]()
        except KeyError:
            continue