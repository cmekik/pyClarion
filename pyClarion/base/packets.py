"""Tools for representing information about node strengths and decisions."""


import typing as typ
import types
import pyClarion.base.symbols as sym


__all__ = [
    "ActivationPacket",
    "DecisionPacket",
    "make_packet"
]


####################
### Type Aliases ###
####################


ConstructSymbolMapping = typ.Mapping[sym.ConstructSymbol, typ.Any]
ConstructSymbolCollection = typ.Collection[sym.ConstructSymbol]
AppraisalData = typ.Tuple[
    ConstructSymbolMapping, 
    ConstructSymbolCollection
]


###################
### Definitions ###
###################


class ActivationPacket(typ.NamedTuple):
    """
    Represents node strengths.
    
    :param strengths: Mapping of node strengths.
    :param origin: Construct symbol identifying source of activation packet.
    """

    strengths: ConstructSymbolMapping
    origin: sym.ConstructSymbol


class DecisionPacket(typ.NamedTuple):
    """
    Represents the result of an appraisal.

    :param strengths: Mapping of node strengths.
    :param chosen: Collection of selected actionable nodes.
    :param origin: Construct symbol identifying source of activation packet.
    """

    strengths: ConstructSymbolMapping
    chosen: ConstructSymbolCollection
    origin: sym.ConstructSymbol


def make_packet(
        csym: sym.ConstructSymbol, 
        data: typ.Union[ConstructSymbolMapping, AppraisalData]
    ):
    """
    Create an activation or decision packet for a client construct.
    
    Assumes csym.ctype in Construct.BasicConstruct.

    :param csym: Client construct.
    :param data: Output of an activation processor.
    """

    packet: typ.Union[ActivationPacket, DecisionPacket]

    if csym.ctype in (
        sym.ConstructType.Node | 
        sym.ConstructType.Flow | 
        sym.ConstructType.Buffer
    ):  
        smap = types.MappingProxyType(typ.cast(ConstructSymbolMapping, data))
        packet = ActivationPacket(strengths=smap, origin=csym)
    elif csym.ctype is sym.ConstructType.Appraisal:
        strengths, chosen = typ.cast(AppraisalData, data)
        smap = types.MappingProxyType(strengths)
        packet = DecisionPacket(strengths=smap, chosen=chosen, origin=csym)
    else:
        raise ValueError(
            "Unexpected ctype {} in argument `csym` to make_packet".format(
                str(csym.ctype)
            )
        )
    
    return packet
