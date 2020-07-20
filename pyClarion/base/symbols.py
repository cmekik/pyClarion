"""Provides tools for naming, indexing, and selecting constructs."""


__all__ = [
    "ConstructType", "Symbol", "MatchSet","feature", "chunk", "flow_in", 
    "flow_bt", "flow_tb", "flow_tt", "flow_bb", "terminus", "buffer", 
    "subsystem", "agent"
]


from enum import Flag, auto
from typing import (
    Optional, Hashable, Tuple, Union, Iterable, Callable, MutableSet
)


#########################
### Class Definitions ###
#########################
 

class ConstructType(Flag):
    """
    Represents construct types within Clarion theory.

    Signals the role of a construct for controlling processing logic.

    See Symbol for usage patterns.
    
    Basic members (and interpretations):
        null_construct: Empty construct type (corresponds to flag null).
        feature: Feature node.
        chunk: Chunk node.
        flow_in: Activation input to a subsystem.
        flow_tb: Activation flow from top to bottom level.
        flow_bt: Activation flow from bottom to top level.
        flow_tt: Activation flow within top level.
        flow_bb: Activation flow within bottom level.
        terminus: Selected responses.
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
        basic_construct: Feature or chunk or flow or terminus or behavior or 
            buffer. 
        container_construct: Subsystem or agent.
    """

    null_construct = 0
    feature = auto()
    chunk = auto()
    flow_in = auto()
    flow_tb = auto()
    flow_bt = auto()
    flow_tt = auto()
    flow_bb = auto()
    terminus = auto()
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
    flow = flow_tb | flow_bt | flow_tt | flow_bb | flow_in
    basic_construct = node | flow | terminus | buffer
    container_construct = subsystem | agent


class Symbol(object):
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
        Initialize a new Symbol.

        :param ctype: Construct type.
        :param cid: Hashable sequence serving as Construct ID.
        """

        if isinstance(ctype, str):
            ctype = ConstructType[ctype]
        elif isinstance(ctype, int):
            ctype = ConstructType(ctype)
        elif isinstance(ctype, ConstructType):
            # do nothing, all good.
            pass
        else:
            raise TypeError(
                "Unexpected type {} for arg ctype.".format(
                    type(ctype).__name__
                )
            )
        
        super().__setattr__('_data', (ctype, tuple(cid)))

    def __setattr__(self, name, value):

        raise AttributeError(
            "{} instance is immutable.".format(type(self).__name__)
        )
    
    def __hash__(self):

        return hash(self._data)

    def __eq__(self, other):

        if isinstance(other, Symbol):
            return self._data == other._data
        else:
            raise TypeError(
                "'==' not supported between instances of '{}' and '{}'.".format(
                    type(self).__name__, type(other).__name__
                )
            )
    
    def __lt__(self, other):

        if isinstance(other, Symbol) and self.ctype == other.ctype:
            return self._data < other._data
        else:
            raise TypeError(
                "'<' not supported between instances of '{}' and '{}'.".format(
                    type(self).__name__, type(other).__name__
                )
            ) 

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

    @property
    def dim(self) -> Optional[Hashable]:
        """Dimension of a feature."""

        if self.ctype in ConstructType.feature:
            return self.cid[0]
        else:
            raise TypeError(
                "Attribute 'dim' not defined for {}".format(self.ctype)
            )

    @property
    def val(self) -> Optional[Hashable]:
        """Value of a feature."""

        if self.ctype in ConstructType.feature:
            return self.cid[1]
        else:
            raise TypeError(
                "Attribute 'dim' not defined for {}".format(self.ctype)
            )


class MatchSet(object):
    """
    A unary predicate that applies to construct symbols.

    MatchSet objects are intended to facilitate checking if constructs satisfy 
    complex conditions. Such checks may be required, for example, to decide 
    whether or not to connect two construct realizers (see realizers.py). 
    In general, MatchSet objects may be used at any point where a (potentially 
    complex) predicate must be applied to construct symbols. 

    MatchSet objects support definition with respect to construct types or 
    arbitrary predicates, and by enumeration of matching constructs. They are 
    set-like in that they support __contains__, may be extended (through 
    addition) or contracted (through removal). However, unlike sets, MatchSet 
    objects do not support algebraic operators such as union, intersection, 
    difference etc.
    """

    def __init__(
        self, 
        ctype: ConstructType = None, 
        constructs: Iterable[Symbol] = None,
        predicates: Iterable[Callable[[Symbol], bool]] = None
    ) -> None:
        """
        Initialize a new Matcher instance.

        :param ctype: Acceptable construct type(s).
        :param constructs: Acceptable construct symbols.
        :param predicates: Custom custom predicates indicating acceptable 
            constructs. 
        """

        self.ctype = ConstructType.null_construct
        self.constructs: MutableSet[Symbol] = set()
        self.predicates: MutableSet[Callable[[Symbol], bool]] = set()
        self.add(ctype, constructs, predicates)

    def __repr__(self):

        ctr = "ConstructType({})".format(self.ctype.value)
        if self.ctype.name is not None:
            ctr = repr(self.ctype.name)
        r = "MatchSet(ctype={}, constructs={}, predicates={})".format(
            ctr, repr(self.constructs), repr(self.predicates)
        )

        return r


    def __contains__(self, key: Symbol) -> bool:
        """
        Return true if construct is in the match set.
        
        A construct is considered to be in the match set if:
            - Its construct type is in self.ctype OR
            - It is equal to a member of self.constructs OR
            - A predicate in self.predicates returns true when called on its 
              construct symbol.
        """

        val = False
        val |= key.ctype in self.ctype
        val |= key in self.constructs
        for predicate in self.predicates:
                val |= predicate(key)
        return val

    def __copy__(self):
        """Make a shallow copy of self."""

        return type(self)(
            ctype=self.ctype,
            constructs=self.constructs.copy(),
            predicates=self.predicates.copy()
        )

    def __ior__(self, other):

        # TODO: Type check? - Can
        self.add(other.ctype, other.constructs, other.predicates)
    
    def __isub__(self, other):

        # TODO: Type check? - Can
        self.remove(other.ctype, other.constructs, other.predicates)


    def add(
        self, 
        ctype: ConstructType = None, 
        constructs: Iterable[Symbol] = None,
        predicates: Iterable[Callable[[Symbol], bool]] = None
    ) -> None:
        """
        Extend the set of accepted constructs.
        
        See Predicate.__init__() for argument descriptions.
        """

        if ctype is not None:
            self.ctype |= ctype
        if constructs is not None:
            self.constructs |= set(constructs)
        if predicates is not None:
            self.predicates |= set(predicates)

    def remove(
        self, 
        ctype: ConstructType = None, 
        constructs: Iterable[Symbol] = None,
        predicates: Iterable[Callable[[Symbol], bool]] = None
    ) -> None:
        """
        Contract the set of accepted constructs.
        
        See Predicate.__init__() for argument descriptions.
        """

        if ctype is not None:
            self.ctype &= ~ctype
        if constructs is not None:
            self.constructs -= set(constructs)
        if predicates is not None:
            self.predicates -= set(predicates)


##################################
### Construct Symbol Factories ###
##################################

# These are convenience functions for constructing construct symbols.
# They simply wrap the appropriate Symbol constructor.


def feature(dim: Hashable, val: Hashable) -> Symbol:
    """
    Return a new feature symbol.

    :param dim: Dimension of feature.
    :param val: Value of feature.
    """

    return Symbol(ConstructType.feature, dim, val)


def chunk(name: Hashable) -> Symbol:
    """
    Return a new chunk symbol.

    :param cid: Chunk identifier.
    """

    return Symbol(ConstructType.chunk, name)


def flow_in(name: Hashable) -> Symbol:
    """
    Return a new input flow symbol.

    :param cid: Name of flow.
    """

    return Symbol(ConstructType.flow_in, name)


def flow_bt(name: Hashable) -> Symbol:
    """
    Return a new bottom-up flow symbol.

    :param cid: Name of flow.
    """

    return Symbol(ConstructType.flow_bt, name)


def flow_tb(name: Hashable) -> Symbol:
    """
    Return a new top-down flow symbol.

    :param cid: Name of flow.
    """

    return Symbol(ConstructType.flow_tb, name)


def flow_tt(name: Hashable) -> Symbol:
    """
    Return a new top-level flow symbol.

    :param cid: Name of flow.
    """

    return Symbol(ConstructType.flow_tt, name)


def flow_bb(name: Hashable) -> Symbol:
    """
    Return a new bottom-level flow symbol.

    :param cid: Name of flow.
    """

    return Symbol(ConstructType.flow_bb, name)


def terminus(name: Hashable) -> Symbol:
    """
    Return a new terminus symbol.

    :param name: Name of terminus.
    """

    return Symbol(ConstructType.terminus, name)


def buffer(name: Hashable) -> Symbol:
    """
    Return a new buffer symbol.

    :param name: Name of buffer.
    """

    return Symbol(ConstructType.buffer, name)


def subsystem(name: Hashable) -> Symbol:
    """
    Return a new subsystem symbol.

    :param name: Name of subsystem.
    """

    return Symbol(ConstructType.subsystem, name)


def agent(name: Hashable) -> Symbol:
    """
    Return a new agent symbol.

    :param name: Name of agent.
    """

    return Symbol(ConstructType.agent, name)
