"""Abstractions for processing activation packets."""


###############
### IMPORTS ###
###############


import abc
import typing as typ
import pyClarion.base.packets as pkt


###################
### DEFINITIONS ###
###################


class ActivationProcessor(typ.Generic[pkt.At], abc.ABC):
    """Abstract base class for routines that manipulate activation packets."""

    pass


class Channel(ActivationProcessor[pkt.At]):
    """Abstract base class for routines that transform activation patterns."""
    
    @abc.abstractmethod
    def __call__(
        self, packet: pkt.ActivationPacket[pkt.At]
    ) -> pkt.ActivationPacket[pkt.At]:
        """Compute and return activations resulting from an input to this 
        channel. 

        :param packet: An activation packet representing the input to self.
        """

        pass


class Junction(ActivationProcessor[pkt.At]):
    """Abstract base class for routines that combine activation packets."""

    @abc.abstractmethod
    def __call__(
        self, *packets: pkt.ActivationPacket[pkt.At]
    ) -> pkt.ActivationPacket[pkt.At]:
        """
        Construct a combined activation packet from inputs.

        :param packets: A sequence of activation packets representing inputs to 
            self.
        """

        pass


class Selector(ActivationProcessor[pkt.At]):
    """Abstract base class for routines that construct decision packets."""

    @abc.abstractmethod
    def __call__(
        self, packet: pkt.ActivationPacket[pkt.At]
    ) -> pkt.DecisionPacket[pkt.At]:
        """
        Construct a decision packet based on input activations.

        :param packet: An activation packet representing the input to self.
        """

        pass


class Effector(ActivationProcessor[pkt.At]):
    """Abstract base class for routines that execute decision packet commands."""
    
    @abc.abstractmethod
    def __call__(self, packet : pkt.DecisionPacket[pkt.At]) -> None:
        """
        Execute actions recommended by input decision packet.

        :param packet: A decision packet specifying action recommendations.
        """

        pass


class Source(ActivationProcessor[pkt.At]):
    """Abstract base class for routines that output activations."""
    
    @abc.abstractmethod
    def __call__(self) -> pkt.ActivationPacket[pkt.At]:
        """Return activation pattern stored in self."""

        pass
