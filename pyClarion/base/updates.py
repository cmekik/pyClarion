"""
Tools for programmatically managing construct realizers.

.. warning::
   Module experimental.

"""

import abc
from typing import Mapping, TypeVar, Dict, Optional, KeysView, Callable
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

    def __init__(self) -> None:
        
        self._constructs: Optional[KeysView] = None

    @abc.abstractmethod
    def update(self) -> None:
        """
        Updates knowledge given result of current activation cycle.

        The API for this is under development. A more specific signature is 
        forthcoming.
        """
        pass

    def get_realizer(self, construct):

        return self._getter(construct)

    def assign(self, constructs: KeysView, getter: Callable) -> None:

        self._constructs = constructs
        self._getter = getter

    @property
    def constructs(self) -> KeysView:

        if self._constructs:
            return self._constructs
        else:
            raise AttributeError("No constructs assigned to self.")
