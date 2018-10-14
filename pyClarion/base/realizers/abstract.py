"""
Tools for defining the behavior of theoretcally relevant constructs.
"""


from abc import ABC, abstractmethod
from typing import TypeVar, Generic, MutableMapping, Dict, Any, Iterator
from pyClarion.base.symbols import (
    ConstructSymbol, BasicConstructSymbol, ContainerConstructSymbol, 
)
from pyClarion.base.utils import may_contain
from pyClarion.base.packets import DefaultActivation
from pyClarion.base.links import BasicInputMonitor, BasicOutputView

Ct = TypeVar('Ct', bound=ConstructSymbol)
Bt = TypeVar('Bt', bound=BasicConstructSymbol)
Xt = TypeVar('Xt', bound=ContainerConstructSymbol)


####################
### ABSTRATCIONS ###
####################


class ConstructRealizer(Generic[Ct], ABC):
    
    def __init__(self, construct: Ct) -> None:

        self.construct = construct

    @abstractmethod
    def propagate(self) -> None:
        pass


class BasicConstructRealizer(ConstructRealizer[Bt]):

    def __init__(self, construct: Bt) -> None:

        super().__init__(construct)

    def _init_io(
        self, 
        has_input: bool = True, 
        has_output: bool = True, 
        default_activation: DefaultActivation = None
    ) -> None:

        if has_input:
            self.input = BasicInputMonitor()
        if has_output:
            self.output = BasicOutputView(default_activation)
            self.propagate()


class ContainerConstructRealizer(
    MutableMapping[ConstructSymbol, ConstructRealizer], ConstructRealizer[Xt]
):

    def __init__(self, construct: Xt) -> None:

        super().__init__(construct)
        self.dict: Dict = dict()

    def __len__(self) -> int:

        return len(self.dict)

    def __contains__(self, obj: Any) -> bool:

        return obj in self.dict

    def __iter__(self) -> Iterator:

        return self.dict.__iter__()

    def __getitem__(self, key: Any) -> Any:

        return self.dict[key]

    def __setitem__(self, key: Any, value: Any) -> None:

        if not may_contain(self.construct, key):
            raise TypeError("Unexpected type {}".format(type(key)))

        if key != value.construct:
            raise ValueError("Mismatch between key and realizer construct.")

        self.dict[key] = value

    def __delitem__(self, key: Any) -> None:

        del self.dict[key]
