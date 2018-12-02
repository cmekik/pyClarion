"""
Abstractions for processing activation packets.

This module provides abstract base classes for each kind of activation 
processor that may be used in a construct realizer. Activation processors 
encapsulate details of processing within construct realizers and they provide 
the main interface by which construct realizer behavior may be customized. 
"""


###############
### IMPORTS ###
###############


import abc
import typing as typ
import pyClarion.base.packets as pkt


##############
### PUBLIC ###
##############


__all__ = [
    "ActivationProcessor",
    "Channel",
    "Junction",
    "Selector",
    "Effector",
    "Source"
]


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
    """
    Abstract base class for routines that output steady activation patterns.

    Output activations may be modified by external processes (e.g., internal 
    cognitive actions, environmental stimulus etc.), but do not directly depend 
    on activation flows within an agent's cognitive apparatus. 
    """
    
    @abc.abstractmethod
    def __call__(self) -> pkt.ActivationPacket[pkt.At]:
        """Return activation pattern stored in self."""

        pass
