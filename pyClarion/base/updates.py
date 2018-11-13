"""Tools for automatically managing construct realizers during simulations."""

import abc
from typing import Mapping, TypeVar, Dict, Optional, KeysView, Callable
from pyClarion.base.symbols import ConstructSymbol, Agent, Subsystem, Buffer
from pyClarion.base.realizers.abstract import ConstructRealizer
from pyClarion.base.realizers.basic import BufferRealizer 


Ct = TypeVar("Ct",bound=ConstructSymbol)


class UpdateManager(abc.ABC):
    """
    Manages some set of construct realizers.

    Manages learning and forgetting routines, monitors subsystem and buffer 
    activity, and adds, removes or modifies construct realizers.
    """

    def __init__(self) -> None:
        
        self._constructs: Optional[KeysView] = None

    @abc.abstractmethod
    def update(self) -> None:
        """
        Update client constructs.
        
        This method should trigger processes such as weight updates in neural 
        networks, creation/deletion of chunk nodes, adjustment of parameters, 
        and other routines associated with the maintenance and management of 
        simulated constructs.
        """

        pass

    def get_realizer(self, construct: ConstructSymbol) -> ConstructRealizer:
        """
        Access chosen construct.
        
        :param construct: Chosen construct.
        """

        return self._getter(construct)

    def assign(self, constructs: KeysView, getter: Callable) -> None:
        """
        Assign constructs to self for management.

        .. note:
           Not intuitive. Multiple calls to this method will reset the set of 
           managed constructs. Multiple calls to assign should simply expand the 
           set of managed constructs, not reset it.

        :param constructs: Client constructs.
        :param getter: Getter method for accessing client constructs.
        """

        self._constructs = constructs
        self._getter = getter

    @property
    def constructs(self) -> KeysView:
        """Constructs managed by self."""

        if self._constructs:
            return self._constructs
        else:
            raise AttributeError("No constructs assigned to self.")
