"""
Tools for programmatically managing construct realizers.

.. warning::
   Module experimental.

"""

import abc
from typing import Mapping
from pyClarion.base.symbols import Agent, Subsystem, Memory
from pyClarion.base.realizer import (
    BasicConstructRealizer, SubsystemRealizer, MemoryRealizer
)


class ConstructReviewer(abc.ABC):
    """
    Manages some class of realizers associated with one or more subsystems.

    Administrators implement learning and forgetting routines. They monitor 
    subsystem and buffer activity and modify client construct realizers.
    """

    def __init__(
        self, 
        agent: Agent, 
        subsystems: Mapping[Subsystem, SubsystemRealizer], 
        buffers: Mapping[Memory, MemoryRealizer]
    ) -> None:
        
        self.agent = agent
        self.subsystems = subsystems
        self.buffers = buffers

    @abc.abstractmethod
    def update_knowledge(self) -> None:
        """
        Updates knowledge given result of current activation cycle.

        The API for this is under development. A more specific signature is 
        forthcoming.
        """
        pass
    
    @abc.abstractmethod
    def initialize_knowledge(self) -> None:
        '''Create and return channel(s) managed by ``self``.'''

        pass
