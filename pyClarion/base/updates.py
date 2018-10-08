"""
Tools for programmatically managing construct realizers.

.. warning::
   Module experimental.

"""

import abc
from typing import Mapping, TypeVar, Dict
from pyClarion.base.symbols import ConstructSymbol, Agent, Subsystem, Buffer
from pyClarion.base.realizers.abstract import ConstructRealizer
from pyClarion.base.realizers.basic import BufferRealizer 


Ct = TypeVar("Ct",bound=ConstructSymbol)


class UpdateManager(abc.ABC):
    """
    Manages some class of realizers associated with one or more subsystems.

    Administrators implement learning and forgetting routines. They monitor 
    subsystem and buffer activity and modify client construct realizers.
    """

    def __init__(
        self, 
        agent_dict: Dict[Ct, ConstructRealizer[Ct]], 
    ) -> None:
        
        self.agent_dict = agent_dict

    @abc.abstractmethod
    def update(self) -> None:
        """
        Updates knowledge given result of current activation cycle.

        The API for this is under development. A more specific signature is 
        forthcoming.
        """
        pass
