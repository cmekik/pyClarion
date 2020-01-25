"""Tools for naming, indexing, and referencing simulated constructs."""


# Notes For Readers

# This file consists of two major sections. The first section contains class 
# definitions; the second section contains construct symbol factory functions.


from typing import Optional, Hashable, Tuple, Union
from enum import Flag, auto


__all__ = [
    "ConstructType", "ConstructSymbol", "FeatureSymbol",
    "feature", "chunk", "flow_bt", "flow_tb", "flow_tt", "flow_bb",
    "buffer", "response", "subsystem", "agent"
]


#########################
### Class Definitions ###
#########################
 

class ConstructType(Flag):
    """
    Represents construct types within Clarion theory.

    Signals the role of a construct for controlling processing logic.

    See ConstructSymbol for usage patterns.
    
    Basic members (and interpretations):
        null_construct: Empty construct type (corresponds to flag null).
        feature: Feature node.
        chunk: Chunk node.
        flow_tb: Activation flow from top to bottom level.
        flow_bt: Activation flow from bottom to top level.
        flow_tt: Activation flow within top level.
        flow_bb: Activation flow within bottom level.
        response: Selected responses.
        buffer: Temporary store of activations.
        subsystem: A Clarion subsystem.
        agent: A full Clarion agent.

    Other members: 
        node: A chunk or microfeature.
        flow_bx: Flow originating in bottom level.
        flow_tx: Flow originating in top level.
        flow_xb: Flow ending in bottom level.
        flow_xt: Flow ending in top level.
        flow_h: Horizontal (intra-level) flow.
        flow_v: Vertical (inter-level) flow.
        flow: Links among microfeature and/or chunk nodes.
        basic_construct: Feature or chunk or flow or response or behavior or 
            buffer. 
        container_construct: Subsystem or agent.
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
    subsystem = auto()
    agent = auto()
    node = feature | chunk
    flow_bx = flow_bt | flow_bb 
    flow_tx = flow_tb | flow_tt 
    flow_xb = flow_tb | flow_bb 
    flow_xt = flow_bt | flow_tt 
    flow_h = flow_bb | flow_tt
    flow_v = flow_tb | flow_bt
    flow_bv = flow_bb | flow_v
    flow_tv = flow_tt | flow_v
    flow = flow_tb | flow_bt | flow_tt | flow_bb
    basic_construct = node | flow | response | buffer
    container_construct = subsystem | agent

    @classmethod
    def from_str(cls, s):
        """Return a construct type based on a name string."""

        try:
            return cls.__members__[s]
        except KeyError:
            raise ValueError(
            "%r is not a valid %s name".format(value, cls.__name__)
        )


class ConstructSymbol(object):
    """
    Symbolic representation of a Clarion construct.

    Abbreviated ConSymb.
    
    Construct symbols are immutable objects that identify simulated constructs. 
    Each symbol has a construct type, which is used to facilitate filtering and 
    conditional logic acting categorically on core Clarion construct types. 
    
    To disambiguate different constructs of the same type within a given agent, 
    each construct symbol is associated with a construct id. Construct ids may 
    be any hashable object. They may also be used to support finer filtering and 
    logic.
    """

    __slots__ = ('_data')
    
    _data: Tuple[ConstructType, Tuple[Hashable, ...]]

    def __init__(
        self, ctype: Union[ConstructType, str, int], *cid: Optional[Hashable]
    ) -> None:
        """
        Initialize a new ConstructSymbol.

        :param ctype: Construct type.
        :param cid: Hashable sequence serving as Construct ID.
        """

        if isinstance(ctype, str):
            ctype = ConstructType.from_str(ctype)
        elif isinstance(ctype, int):
            ctype = ConstructType(ctype)
        elif isinstance(ctype, ConstructType):
            # do nothing, all good.
            pass
        else:
            raise TypeError(
                "Unexpected type {} for arg ctype.".format(type(ctype).__name__)
            )
        
        super().__setattr__('_data', (ctype, tuple(cid)))

    def __setattr__(self, name, value):

        raise AttributeError(
            "{} instance is immutable.".format(type(self).__name__)
        )
    
    def __hash__(self):

        return hash(self._data)

    def __eq__(self, other):

        if isinstance(other, ConstructSymbol):
            return self._data == other._data
        else:
            return NotImplemented

    def __repr__(self):

        ctr = "ConstructType({})".format(self.ctype.value)
        if self.ctype.name is not None:
            ctr = repr(self.ctype.name)
        args = ctr + ', ' + ', '.join(map(repr, self.cid))
        r = "{}({})".format(type(self).__name__, args)

        return r

    def __str__(self):

        if self.ctype.name is not None:
            ctype_str = self.ctype.name
            cid_repr = ', '.join(map(repr, self.cid))
            s = '{}({})'.format(ctype_str, cid_repr)
        else:
            s = repr(self)

        return s

    @property
    def ctype(self) -> ConstructType:
        """Construct type associated with self."""

        return self._data[0]

    @property
    def cid(self) -> Tuple[Optional[Hashable], ...]:
        """Construct identifier associated with self."""

        return self._data[1]


# Abbreviated handle for ConstructSymbol.
ConSymb = ConstructSymbol


class FeatureSymbol(ConstructSymbol):
    """
    Symbolic representation of a feature node.

    Extends ConstructSymbol to support accessing dim and val attributes.
    """

    __slots__ = ()

    def __init__(
        self, dim: Optional[Hashable], val: Optional[Hashable]
    ) -> None:

        super().__init__(ConstructType.feature, dim, val)

    def __repr__(self):

        args = ', '.join(map(repr, self.cid))
        r = "{}({})".format(type(self).__name__, args)

        return r

    @property
    def dim(self) -> Optional[Hashable]:

        return self.cid[0]

    @property
    def val(self) -> Optional[Hashable]:

        return self.cid[1]


##################################
### Construct Symbol Factories ###
##################################

# These are convenience functions for easily constructing construct symbols.
# They simply wrap the appropriate ConstructSymbol constructor.


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
