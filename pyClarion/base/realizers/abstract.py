"""
Tools for defining the behavior of theoretcally relevant constructs.
"""


from abc import ABC, abstractmethod
from typing import TypeVar, Generic, MutableMapping
from pyClarion.base.symbols import (
    ConstructSymbol, BasicConstructSymbol, CompositeConstructSymbol, 
)
from pyClarion.base.links import BasicInputMonitor, BasicOutputView

Ct = TypeVar('Ct', bound=ConstructSymbol)
Bt = TypeVar('Bt', bound=BasicConstructSymbol)
Xt = TypeVar('Xt', bound=CompositeConstructSymbol)


####################
### ABSTRATCIONS ###
####################


class Realizer(Generic[Ct], ABC):
    
    def __init__(self, construct: Ct) -> None:

        self.construct = construct

    @abstractmethod
    def do(self) -> None:
        pass


class BasicConstructRealizer(Realizer[Bt]):

    def __init__(self, construct: Bt) -> None:

        super().__init__(construct)

    def _init_io(self) -> None:

        self.input = BasicInputMonitor()
        self.output = BasicOutputView()
        self.do()


class CompositeConstructRealizer(
    MutableMapping[ConstructSymbol, Realizer], Realizer[Xt]
):
    pass
