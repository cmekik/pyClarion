"""Tools for naming, indexing, and referencing simulated constructs."""


# Notes For Readers

# - This file consists of two major sections. The first major section 
#   contains class definitions, the second major section contains construct 
#   symbol factory functions.


__all__ = [
    "ConstructSymbol", "ConstructType", "FlowType", "DVPair", "FlowID", 
    "AppraisalID", "BehaviorID", "BufferID", "Microfeature", "Chunk", "Flow",
    "Appraisal", "Behavior", "Buffer", "Subsystem", "Agent"
]


from typing import NamedTuple, Hashable, Sequence
from enum import Flag, auto


ConstructSymbolSequence = Sequence['ConstructSymbol']


#########################
### Class Definitions ###
#########################


class ConstructType(Flag):
    """Represents various types of construct in Clarion theory."""

    Microfeature = auto()
    Chunk = auto()
    Flow = auto()
    Appraisal = auto()
    Behavior = auto()
    Buffer = auto()
    Subsystem = auto()
    Agent = auto()

    Node = Microfeature | Chunk
    BasicConstruct = (
        Microfeature | Chunk | Flow | Appraisal | Behavior | Buffer
    )
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


class ConstructSymbol(NamedTuple):
    """
    Symbolically represents simulation constructs.
    
    Construct symbols identify and carry essential information about simulated 
    constructs.

    :param ctype: Construct type.
    :param cid: Construct ID.
    """
    
    ctype: ConstructType
    cid: Hashable

    def __str__(self):
        """
        Pretty print construct symbol.

        Output has form: 'ConstructName(id)'
        """

        return "".join([str(self.ctype), "(", repr(self.cid), ")"])


class DVPair(NamedTuple):
    """Represents a microfeature dimension-value pair."""
    
    dim: Hashable
    val: Hashable


class FlowType(Flag):
    """
    Signals the direction(s) of an activation flow.
    
    May take on four basic values:
        TT: Activation flows within the top-level.
        BB: Activation flows within the bottom-level.
        TB: Top-down activation flows.
        BT: Bottom-up activation flows.
    """

    TT = auto()
    BB = auto()
    TB = auto()
    BT = auto()


class FlowID(NamedTuple):
    """Represents the name and type of a flow."""

    name: Hashable
    ftype: FlowType


class AppraisalID(NamedTuple):
    """Represents name, input construct type, and outputs of an appraisal."""

    name: Hashable
    itype: ConstructType


class BehaviorID(NamedTuple):
    """Represents the name and client appraisal of a behavior."""

    name: Hashable
    appraisal: ConstructSymbol


class BufferID(NamedTuple):
    """Represents the name and output destinations of a buffer."""

    name: Hashable
    outputs: ConstructSymbolSequence


##################################
### Construct Symbol Factories ###
##################################


# Construct symbol factory names mimic class naming style to free up namespace 
# for simulation variables. For instance, the chunk factory is named `Chunk()` 
# not `chunk()` to allow use of `chunk` as a variable name by the user.


def Microfeature(dim: Hashable, val: Hashable) -> ConstructSymbol:
    """
    Return a new microfeature symbol.
    
    Assumes microfeature is in dv-pair form.

    :param dim: Dimension of microfeature.
    :param val: Value of microfeature.
    """

    return ConstructSymbol(ConstructType.Microfeature, DVPair(dim, val))


def Chunk(cid: Hashable) -> ConstructSymbol:
    """
    Return a new chunk symbol.

    :param cid: Chunk identifier.
    """

    return ConstructSymbol(ConstructType.Chunk, cid)


def Flow(name: Hashable, ftype: FlowType) -> ConstructSymbol:
    """
    Return a new flow symbol.

    Assumes flow is in name-ftype form.

    :param name: Name of flow.
    :param ftype: Flow type.
    """

    return ConstructSymbol(ConstructType.Flow, FlowID(name, ftype))


def Appraisal(name: Hashable, itype: ConstructType) -> ConstructSymbol:
    """
    Return a new appraisal symbol.

    :param cid: Appraisal identifier.
    """

    return ConstructSymbol(ConstructType.Appraisal, AppraisalID(name, itype))


def Behavior(
    name: Hashable, appraisal: ConstructSymbol
) -> ConstructSymbol:
    """
    Return a new behavior symbol.

    :param name: Behavior identifier.
    :param appraisal: Client appraisal.
    """

    return ConstructSymbol(ConstructType.Behavior, BehaviorID(name, appraisal))


def Buffer(
    name: Hashable, outputs: ConstructSymbolSequence
) -> ConstructSymbol:
    """
    Return a new buffer symbol.

    :param cid: Buffer identifier.
    """

    return ConstructSymbol(ConstructType.Buffer, BufferID(name, outputs))



def Subsystem(cid: Hashable) -> ConstructSymbol:
    """
    Return a new subsystem symbol.

    :param cid: Subsystem identifier.
    """

    return ConstructSymbol(ConstructType.Subsystem, cid)


def Agent(cid: Hashable) -> ConstructSymbol:
    """
    Return a new agent symbol.

    :param cid: Agent identifier.
    """

    return ConstructSymbol(ConstructType.Agent, cid)
