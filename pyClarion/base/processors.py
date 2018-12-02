"""Abstractions for processing activation packets."""


from abc import ABC, abstractmethod
from typing import Generic
from pyClarion.base.packets import ActivationPacket, DecisionPacket, At


class ActivationProcessor(Generic[At], ABC):
    """Abstract base class for routines that manipulate activation packets."""

    pass


class Channel(ActivationProcessor[At]):
    """Abstract base class for routines that transform activation patterns."""
    
    @abstractmethod
    def __call__(self, packet: ActivationPacket[At]) -> ActivationPacket[At]:
        """Compute and return activations resulting from an input to this 
        channel. 

        :param packet: An activation packet representing the input to self.
        """

        pass


class Junction(ActivationProcessor[At]):
    """Abstract base class for routines that combine activation packets."""

    @abstractmethod
    def __call__(self, *packets: ActivationPacket[At]) -> ActivationPacket:
        """
        Construct a combined activation packet from inputs.

        :param packets: A sequence of activation packets representing inputs to 
            self.
        """

        pass


class Selector(ActivationProcessor[At]):
    """Abstract base class for routines that construct decision packets."""

    @abstractmethod
    def __call__(self, packet: ActivationPacket[At]) -> DecisionPacket[At]:
        """
        Construct a decision packet based on input activations.

        :param packet: An activation packet representing the input to self.
        """

        pass


class Effector(ActivationProcessor[At]):
    """Abstract base class for routines that execute decision packet commands."""
    
    @abstractmethod
    def __call__(self, packet : DecisionPacket[At]) -> None:
        """
        Execute actions recommended by input decision packet.

        :param packet: A decision packet specifying action recommendations.
        """

        pass


class Source(ActivationProcessor[At]):
    """Abstract base class for routines that output activations."""
    
    @abstractmethod
    def __call__(self) -> ActivationPacket[At]:
        """Return activation pattern stored in self."""

        pass
