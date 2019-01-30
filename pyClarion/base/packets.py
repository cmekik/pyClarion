"""Tools for representing information about node strengths and decisions."""


from typing import Any, Mapping, Collection, Tuple, NamedTuple, Union, cast 
from types import MappingProxyType
from pyClarion.base.symbols import ConstructSymbol, ConstructType


__all__ = ["ActivationPacket", "DecisionPacket"]


####################
### Type Aliases ###
####################


ConstructSymbolMapping = Mapping[ConstructSymbol, Any]
ConstructSymbolCollection = Collection[ConstructSymbol]


###################
### Definitions ###
###################


class ActivationPacket(NamedTuple):
    """
    Represents node strengths.
    
    :param strengths: Mapping of node strengths.
    :param origin: Construct symbol identifying source of activation packet.
    """

    strengths: ConstructSymbolMapping
    origin: ConstructSymbol


class DecisionPacket(NamedTuple):
    """
    Represents the result of an appraisal.

    :param strengths: Mapping of node strengths.
    :param chosen: Collection of selected actionable nodes.
    :param origin: Construct symbol identifying source of activation packet.
    """

    strengths: ConstructSymbolMapping
    chosen: ConstructSymbolCollection
    origin: ConstructSymbol
