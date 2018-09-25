"""
Tools for defining the behavior of theoretcally relevant constructs.
"""


import dataclasses
from typing import TypeVar, Generic, MutableMapping
from pyClarion.base.symbols import (
    ConstructSymbol, BasicConstructSymbol, CompositeConstructSymbol, 
)


Ct = TypeVar('Ct', bound=ConstructSymbol)
Bt = TypeVar('Bt', bound=BasicConstructSymbol)
Xt = TypeVar('Xt', bound=CompositeConstructSymbol)


####################
### ABSTRATCIONS ###
####################


class Realizer(Generic[Ct], object):
    
    construct: Ct

    def __init__(self, construct: Ct) -> None:

        self.construct = construct


@dataclasses.dataclass()
class BasicConstructRealizer(Realizer[Bt]):

    construct: Bt


class CompositeConstructRealizer(
    MutableMapping[ConstructSymbol, Realizer], Realizer[Xt]
):
    pass
