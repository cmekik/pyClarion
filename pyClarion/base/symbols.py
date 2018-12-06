"""Tools for naming, indexing, and referencing simulated constructs."""


# Notes For Readers

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
    Symbolically represents simulation constructs.
    
    Construct symbols identify and carry essential information about simulated 
    constructs.

    :param ctype: Construct type.
    :param cid: Construct ID.
    """
    
    ctype: 'ConstructType'
    cid: typ.Hashable

    def __str__(self):
        """
        Pretty print construct symbol.

        Output has form: 'ConstructName(id)'
        """

        return "".join([str(self.ctype), "(", repr(self.cid), ")"])


class ConstructType(enum.Flag):
    """Represents construct types for processing logic."""

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
        Returns the construct type name. 
        
        If no construct type name is available, falls back on repr.
        """

        if self.name:
            return self.name
        else:
            return repr(self)


class FlowType(enum.Flag):
    """
    Signals the direction(s) of an activation flow.
    
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


def Microfeature(dim: typ.Hashable, val: typ.Hashable) -> ConstructSymbol:
    """
    Return a new microfeature symbol.
    
    Assumes microfeature is in dv-pair form.

    :param dim: Dimension of microfeature.
    :param val: Value of microfeature.
    """

    return ConstructSymbol(ConstructType.Microfeature, DVPair(dim, val))


def Chunk(cid: typ.Hashable) -> ConstructSymbol:
    """
    Return a new chunk symbol.

    :param cid: Chunk identifier.
    """

    return ConstructSymbol(ConstructType.Chunk, cid)


def Flow(name: typ.Hashable, ftype: FlowType) -> ConstructSymbol:
    """
    Return a new flow symbol.

    Assumes flow is in name-ftype form.

    :param name: Name of flow.
    :param ftype: Flow type.
    """

    return ConstructSymbol(ConstructType.Flow, FlowID(name, ftype))


def Appraisal(cid: typ.Hashable) -> ConstructSymbol:
    """
    Return a new appraisal symbol.

    :param cid: Appraisal identifier.
    """

    return ConstructSymbol(ConstructType.Appraisal, cid)


def Behavior(cid: typ.Hashable) -> ConstructSymbol:
    """
    Return a new behavior symbol.

    :param cid: Behavior identifier.
    """

    return ConstructSymbol(ConstructType.Behavior, cid)


def Buffer(cid: typ.Hashable) -> ConstructSymbol:
    """
    Return a new buffer symbol.

    :param cid: Buffer identifier.
    """

    return ConstructSymbol(ConstructType.Buffer, cid)



def Subsystem(cid: typ.Hashable) -> ConstructSymbol:
    """
    Return a new subsystem symbol.

    :param cid: Subsystem identifier.
    """

    return ConstructSymbol(ConstructType.Subsystem, cid)


def Agent(cid: typ.Hashable) -> ConstructSymbol:
    """
    Return a new agent symbol.

    :param cid: Agent identifier.
    """

    return ConstructSymbol(ConstructType.Agent, cid)
