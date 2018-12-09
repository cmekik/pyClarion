"""Tools for naming, indexing, and referencing simulated constructs."""


# Notes For Readers

# - This file consists of two major sections. The first major section 
#   contains class definitions, the second major section contains construct 
#   symbol factory functions.


__all__ = [
    "ConstructSymbol", "ConstructType", "FlowType", "DVPair", "FlowID", 
    "ResponseID", "BehaviorID", "BufferID", "Microfeature", "Chunk", "Flow",
    "Response", "Behavior", "Buffer", "Subsystem", "Agent"
]


from typing import NamedTuple, Hashable, Sequence
from enum import Flag, auto


ConstructSymbolSequence = Sequence['ConstructSymbol']


#########################
### Class Definitions ###
#########################


class ConstructType(Flag):
    """
    Represents various types of construct in Clarion theory.
    
    Basic members (and interpretations):
        Microfeature: Microfeature node.
        Chunk: Chunk node.
        Flow: Links among microfeature and/or chunk nodes.
        Response: Selected responses.
        Behavior: Possible actions.
        Buffer: Temporary store of activations.
        Subsystem: A Clarion subsystem.
        Agent: A full Clarion agent.

    Other members:
        Node: A chunk or microfeature.
        BasicConstruct: Microfeature or chunk or flow or response or behavior or 
            buffer. 
        ContainerConstruct: Subsystem or agent.
    """

    Microfeature = auto()
    Chunk = auto()
    Flow = auto()
    Response = auto()
    Behavior = auto()
    Buffer = auto()
    Subsystem = auto()
    Agent = auto()

    Node = Microfeature | Chunk
    BasicConstruct = (
        Microfeature | Chunk | Flow | Response | Behavior | Buffer
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


class ResponseID(NamedTuple):
    """Represents name and input construct type of a response."""

    name: Hashable
    itype: ConstructType


class BehaviorID(NamedTuple):
    """Represents the name and client response construct of a behavior."""

    name: Hashable
    response: ConstructSymbol


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


def Response(name: Hashable, itype: ConstructType) -> ConstructSymbol:
    """
    Return a new response symbol.

    :param name: Name of response.
    :param itype: Input type to response construct.
    """

    return ConstructSymbol(ConstructType.Response, ResponseID(name, itype))


def Behavior(
    name: Hashable, response: ConstructSymbol
) -> ConstructSymbol:
    """
    Return a new behavior symbol.

    :param name: Name of behavior.
    :param response: Client response construct.
    """

    return ConstructSymbol(ConstructType.Behavior, BehaviorID(name, response))


def Buffer(
    name: Hashable, outputs: ConstructSymbolSequence
) -> ConstructSymbol:
    """
    Return a new buffer symbol.

    :param name: Name of buffer.
    :param outputs: Constructs receiving output of buffer.
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
