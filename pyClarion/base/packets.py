"""Tools for representing information about node strengths and decisions."""


from typing import Any, Mapping, Collection, Tuple, NamedTuple, Union, cast 
from types import MappingProxyType
from pyClarion.base.symbols import ConstructSymbol, ConstructType


__all__ = ["ActivationPacket", "DecisionPacket", "make_packet"]


####################
### Type Aliases ###
####################


ConstructSymbolMapping = Mapping[ConstructSymbol, Any]
ConstructSymbolCollection = Collection[ConstructSymbol]
AppraisalData = Tuple[ConstructSymbolMapping, ConstructSymbolCollection]
PacketData = Union[ConstructSymbolMapping, AppraisalData]


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


# Mypy complains if Packet is defined earlier... Should be with type aliases.
Packet = Union["ActivationPacket", "DecisionPacket"]

def make_packet(csym: ConstructSymbol, data: PacketData) -> Packet:
    """
    Create an activation or decision packet for a client construct.
    
    Assumes csym.ctype in ConstructType.BasicConstruct.

    :param csym: Client construct.
    :param data: Output of an activation processor.
    """

    if csym.ctype in (
        ConstructType.Node | ConstructType.Flow | ConstructType.Buffer
    ):  
        strengths = cast(ConstructSymbolMapping, data)
        smap = MappingProxyType(strengths)
        return ActivationPacket(strengths=smap, origin=csym)
    elif csym.ctype is ConstructType.Appraisal:
        dstrengths, chosen = cast(AppraisalData, data)
        smap = MappingProxyType(dstrengths)
        return DecisionPacket(strengths=smap, chosen=chosen, origin=csym)
    else:
        raise ValueError(
            "Unexpected ctype {} in argument `csym` to make_packet".format(
                str(csym.ctype)
            )
        )
