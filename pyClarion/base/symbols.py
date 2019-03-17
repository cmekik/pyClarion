"""Tools for naming, indexing, and referencing simulated constructs."""


# Notes For Readers

# - This file consists of two major sections. The first major section 
#   contains class definitions, the second major section contains construct 
#   symbol factory functions.


from typing import Hashable, Tuple
from enum import Flag, auto


#########################
### Class Definitions ###
#########################


class ConstructType(Flag):
    """
    Represents construct types within Clarion theory.
    
    Basic members (and interpretations):
        NullConstruct: Empty construct type (corresponds to flag null).
        Feature: Feature node.
        Chunk: Chunk node.
        TBFlow: Activation flow from top to bottom level.
        BTFlow: Activation flow from bottom to top level.
        TTFlow: Activation flow within top level.
        BBFlow: Activation flow within bottom level.
        Response: Selected responses.
        Behavior: Possible actions.
        Buffer: Temporary store of activations.
        Subsystem: A Clarion subsystem.
        Agent: A full Clarion agent.

    Other members: 
        Node: A chunk or microfeature.
        Flow: Links among microfeature and/or chunk nodes.
        BasicConstruct: Feature or chunk or flow or response or behavior or 
            buffer. 
        ContainerConstruct: Subsystem or agent.
    """

    null_construct = 0
    feature = auto()
    chunk = auto()
    flow_tb = auto()
    flow_bt = auto()
    flow_tt = auto()
    flow_bb = auto()
    response = auto()
    buffer = auto()
    updater = auto()
    subsystem = auto()
    agent = auto()
    node = feature | chunk
    flow_bx = flow_bt | flow_bb 
    flow_tx = flow_tb | flow_tt 
    flow_xb = flow_tb | flow_bb 
    flow_xt = flow_bt | flow_tt 
    flow = flow_tb | flow_bt | flow_tt | flow_bb
    basic_construct = node | flow | response | buffer | updater
    container_construct = subsystem | agent


class ConstructSymbol:
    """
    Symbolically represents simulation constructs.
    
    Construct symbols are immutable objects that identify simulated constructs.
    """

    __slots__ = ('_data')
    
    _data: Tuple[ConstructType, Tuple[Hashable, ...]]

    def __init__(self, ctype: ConstructType, *cid: Hashable):
        """
        Initialize a new ConstructSymbol.

        :param ctype: Construct type.
        :param cid: Hashable sequence serving as Construct ID.
        """

        super().__setattr__('_data', (ctype, tuple(cid)))

    def __setattr__(self, name, value):

        raise AttributeError(
            "{} instance is immutable.".format(self.__class__.__name__)
        )
    
    def __hash__(self):

        return hash(self._data)

    def __eq__(self, other):

        if isinstance(other, ConstructSymbol):
            return self._data == other._data
        else:
            return NotImplemented

    def __repr__(self):

        return "<{}: {}>".format(self.__class__.__name__, str(self))

    def __str__(self):

        ctype_str = self.ctype.name or str(self.ctype)
        cid_arg_repr = ', '.join(map(repr, self.cid))
        csym_str = '{}({})'.format(ctype_str, cid_arg_repr)
        return csym_str

    @property
    def ctype(self) -> ConstructType:
        """Construct type associated with self."""

        return self._data[0]

    @property
    def cid(self) -> Tuple[Hashable, ...]:
        """Construct identifier associated with self."""

        return self._data[1]


class FeatureSymbol(ConstructSymbol):
    """
    Symbolically represents a feature node.

    Extends ConstructSymbol to support accessing dim and val attributes.
    """

    __slots__ = ()

    def __init__(self, dim: Hashable, val: Hashable) -> None:

        super().__init__(ConstructType.feature, dim, val)

    @property
    def dim(self) -> Hashable:

        return self.cid[0]

    @property
    def val(self) -> Hashable:

        return self.cid[1]


##################################
### Construct Symbol Factories ###
##################################


def feature(dim: Hashable, val: Hashable) -> ConstructSymbol:
    """
    Return a new feature symbol.

    :param dim: Dimension of feature.
    :param val: Value of feature.
    """

    return FeatureSymbol(dim, val)


def chunk(name: Hashable) -> ConstructSymbol:
    """
    Return a new chunk symbol.

    :param cid: Chunk identifier.
    """

    return ConstructSymbol(ConstructType.chunk, name)


def flow_bt(name: Hashable) -> ConstructSymbol:
    """
    Return a new bottom-up flow symbol.

    :param cid: Name of flow.
    """

    return ConstructSymbol(ConstructType.flow_bt, name)


def flow_tb(name: Hashable) -> ConstructSymbol:
    """
    Return a new top-down flow symbol.

    :param cid: Name of flow.
    """

    return ConstructSymbol(ConstructType.flow_tb, name)


def flow_tt(name: Hashable) -> ConstructSymbol:
    """
    Return a new top-level flow symbol.

    :param cid: Name of flow.
    """

    return ConstructSymbol(ConstructType.flow_tt, name)


def flow_bb(name: Hashable) -> ConstructSymbol:
    """
    Return a new bottom-level flow symbol.

    :param cid: Name of flow.
    """

    return ConstructSymbol(ConstructType.flow_bb, name)


def response(name: Hashable) -> ConstructSymbol:
    """
    Return a new response symbol.

    :param name: Name of response.
    """

    return ConstructSymbol(ConstructType.response, name)


def buffer(name: Hashable) -> ConstructSymbol:
    """
    Return a new buffer symbol.

    :param name: Name of buffer.
    """

    return ConstructSymbol(ConstructType.buffer, name)


def updater(name: Hashable) -> ConstructSymbol:
    """
    Return a new updater symbol.

    :param name: Name of behavior.
    """

    return ConstructSymbol(ConstructType.updater, name)


def subsystem(name: Hashable) -> ConstructSymbol:
    """
    Return a new subsystem symbol.

    :param name: Name of subsystem.
    """

    return ConstructSymbol(ConstructType.subsystem, name)


def agent(name: Hashable) -> ConstructSymbol:
    """
    Return a new agent symbol.

    :param name: Name of agent.
    """

    return ConstructSymbol(ConstructType.agent, name)
