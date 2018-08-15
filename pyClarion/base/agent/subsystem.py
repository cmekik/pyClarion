import abc

from ..activation.packet import ActivationPacket
from ..activation.channel import ChannelSet
from .component import ComponentSet
from .selector import Selector
from .effector import Effector

class Subsystem(abc.ABC):
    """A Clarion subsystem.
    """

    @abc.abstractmethod
    def __call__(
        self, 
        input_map : ActivationPacket
    ) -> None:
        """Run through one processing.
        """
        pass

    def select_actions(self, input_map : ActivationPacket) -> None:
        """Select a 
        """

        actionable_chunks = self.effector.get_actionable_chunks(input_map)
        self.effector.buffer = self.selector(input_map, actionable_chunks)

    def execute_actions(self) -> None:
        
        self.effector.fire_buffered()

    @property
    @abc.abstractmethod
    def channels(self) -> ChannelSet:
        """A set of activation channels representing top- and bottom-level 
        knowledge stored in self.
        """
        pass

    @property
    @abc.abstractmethod
    def components(self) -> ComponentSet:
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