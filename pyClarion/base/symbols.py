"""
Tools for naming and indexing simulated constructs.

This module defines construct symbols, which are symbolic tokens that may be 
used to name, index, and reference simulated constructs.
"""


# Notes For Readers

#   - Type hints signal intended usage.
#   - This file consists of two major sections. The first major section contains 
#     class definitions, the second major section contains construct symbol 
#     factory functions.


import typing as typ
import enum 


__all__ = [
    "ConstructSymbol",
    "ConstructType",
    "FlowType",
    "DVPair",
    "FlowID",
    "Microfeature",
    "Chunk",
    "Flow",
    "Appraisal",
    "Behavior",
    "Buffer",
    "Subsystem",
    "Agent"
]


#########################
### Class Definitions ###
#########################


class ConstructSymbol(typ.NamedTuple):
    """
    General base class for symbols representing simulation constructs.
    
    Construct symbols are used to identify and carry essential information 
    about key simulated constructs in a lightweight manner.

    Every construct symbol is expected to present a construct type in its 
    `type` field, and a construct id in its `id` field. Construct types are 
    used in the control of activation flows, construct ids serve to disambiguate 
    and identify construct symbols of a given type. 
    
    In general, each construct symbol is associated with at least one construct 
    realizer, which defines the behavior of the model vis à vis that construct 
    in some particular context. For information on construct realizers see
    ``pyClarion.base.realizers``.

    :param ctype: Construct type.
    :param cid: Construct ID.
    """
    
    ctype: 'ConstructType'
    cid: typ.Hashable

    def __str__(self):
        """
        Pretty print construct symbol.

        Output has form:
            ConstructName(id)
        """

        return "".join([str(self.ctype), "(", repr(self.cid), ")"])


class ConstructType(enum.Flag):
    """Flag for signaling the construct type of a construct symbol."""

    Microfeature = enum.auto()
    Chunk = enum.auto()
    Flow = enum.auto()
    Appraisal = enum.auto()
    Behavior = enum.auto()
    Buffer = enum.auto()
    Subsystem = enum.auto()
    Agent = enum.auto()

    Node = Microfeature | Chunk
    BasicConstruct = Microfeature | Chunk | Flow | Appraisal | Behavior | Buffer
    ContainerConstruct = Subsystem | Agent

    def __str__(self):
        """
        Return the construct name. 
        
        If no construct name is available, falls back on repr.
        """

        if self.name:
            return self.name
        else:
            return repr(self)


class FlowType(enum.Flag):
    """
    Flag for signaling the direction(s) of an activation flow.
    
    May take on four basic values:
        TT: Activation flows within the top-level.
        BB: Activation flows within the bottom-level.
        TB: Top-down activation flows.
        BT: Bottom-up activation flows.
    """

    TT = enum.auto()
    BB = enum.auto()
    TB = enum.auto()
    BT = enum.auto()


class DVPair(typ.NamedTuple):
    """Represents a microfeature dimension-value pair."""
    
    dim: typ.Hashable
    val: typ.Hashable


class FlowID(typ.NamedTuple):
    """Represents the name and type of a flow."""

    name: typ.Hashable
    ftype: FlowType


##################################
### Construct Symbol Factories ###
##################################


# Construct symbol factory names mimic class naming style to free up namespace 
# for simulation variables. For instance, the chunk factory is named `Chunk()` 
# not `chunk()` to allow use of `chunk` as a variable name by the user.


def Microfeature(dim: typ.Hashable, val: typ.Hashable):
    """
    Return a new microfeature symbol.
    
    Assumes microfeature is in dv-pair form.

    :param dim: Dimension of microfeature.
    :param val: Value of microfeature.
    """

    return ConstructSymbol(ConstructType.Microfeature, DVPair(dim, val))


def Chunk(cid: typ.Hashable):
    """
    Return a new chunk symbol.

    :param cid: Chunk identifier.
    """

    return ConstructSymbol(ConstructType.Chunk, cid)


def Flow(name: typ.Hashable, ftype: FlowType):
    """
    Return a new flow symbol.

    Assumes flow is in name-ftype form.

    :param name: Name of flow.
    :param ftype: Flow type.
    """

    return ConstructSymbol(ConstructType.Flow, FlowID(name, ftype))


def Appraisal(cid: typ.Hashable):
    """
    Return a new appraisal symbol.

    :param cid: Appraisal identifier.
    """

    return ConstructSymbol(ConstructType.Appraisal, cid)


def Behavior(cid: typ.Hashable):
    """
    Return a new behavior symbol.

    :param cid: Behavior identifier.
    """

    return ConstructSymbol(ConstructType.Behavior, cid)


def Buffer(cid: typ.Hashable):
    """
    Return a new buffer symbol.

    :param cid: Buffer identifier.
    """

    return ConstructSymbol(ConstructType.Buffer, cid)



def Subsystem(cid: typ.Hashable):
    """
    Return a new subsystem symbol.

    :param cid: Subsystem identifier.
    """

    return ConstructSymbol(ConstructType.Subsystem, cid)


def Agent(cid: typ.Hashable):
    """
    Return a new agent symbol.

    :param cid: Agent identifier.
    """

    return ConstructSymbol(ConstructType.Agent, cid)
