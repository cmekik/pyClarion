import abc
import typing as T
from pyClarion.base.packet import BaseActivationPacket
from pyClarion.base.channel import Channel
from pyClarion.base.selector import Selector
from pyClarion.base.effector import Effector


class Statistic(abc.ABC):
    """Tracks a statistic.

    Contains and updates some relevant statistic(s). Does not store additional 
    information about related constructs, such as references to the objects of 
    stored statistics.
    """
    pass


class Component(abc.ABC):
    """Manages some class of activation channels associated with a subsystem.

    Components are abstractions meant to capture learning and forgetting 
    routines. They monitor the activity of the subsystem to which they belong 
    and modify its members (channels and/or parameters).
    """
    pass


class Subsystem(abc.ABC):
    """A Clarion subsystem.
    """

    @abc.abstractmethod
    def __call__(self) -> None:
        """Run through one processing cycle.
        """
        pass

    def select_actions(self, input_map : BaseActivationPacket) -> None:
        """Select a 
        """

        actionable_chunks = self.effector.get_actionable_chunks(input_map)
        self.effector.buffer = self.selector(input_map, actionable_chunks)

    def execute_actions(self) -> None:
        
        self.effector.fire_buffered()

    @property
    @abc.abstractmethod
    def channels(self) -> T.Set[Channel]:
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
    def selector(self) -> Selector:
        """Binds actionable chunks to action routines.
        """
        pass

    @property
    @abc.abstractmethod
    def effector(self) -> Effector:
        """Binds actionable chunks to action routines.
        """
        pass


class Agent(object):
    """Represents a Clarion agent.

    ``Agent`` objects facilitate the interface between subsystems and the 
    environment. The main responsibility of these objects is to distribute 
    sensory input to subsystems. They also serve to bind together all 
    subsystems associated with a given subject.

    It may also be useful to define action callbacks affecting the environment 
    as methods of this class. Action-centered subsystems would be passed sets 
    of these methods as the callbacks to execute following an action decision. 
    Internal actions may be defined within the relevant subsystem class 
    definition and passed to relevant subsystems in the same way.
    """

    def __call__(self) -> None:
        """Receive and process a new set of sensory/world information.
        """
        pass

    @property
    def subsystems(self) -> T.Set[Subsystem]:
        pass