"""Tools for naming, indexing, and referencing simulated constructs."""


# Notes For Readers

# This file consists of two major sections. The first section contains class 
# definitions; the second section contains construct symbol factory functions.


from typing import Hashable, Tuple, MutableSet, List, Callable, Iterable, cast
from enum import Flag, auto


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
        parameter: A construct parameter.
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
    parameter = auto()
    subsystem = auto()
    agent = auto()
    node = feature | chunk
    flow_bx = flow_bt | flow_bb 
    flow_tx = flow_tb | flow_tt 
    flow_xb = flow_tb | flow_bb 
    flow_xt = flow_bt | flow_tt 
    flow_h = flow_bb | flow_tt
    flow_v = flow_tb | flow_bt
    flow = flow_tb | flow_bt | flow_tt | flow_bb
    basic_construct = node | flow | response | buffer
    container_construct = subsystem | agent


class ConstructSymbol:
    """
    Symbolic representation of a Clarion construct.

    Abbreviated ConSym.
    
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


# Abbreviated handle for ConstructSymbol.
ConSym = ConstructSymbol


class FeatureSymbol(ConstructSymbol):
    """
    Symbolic representation of a feature node.

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


class ParameterSymbol(ConstructSymbol):
    """
    Symbolic representation of a Clarion construct parameter.

    Extends ConstructSymbol to support accessing client construct and name 
    attributes.
    """

    __slots__ = ()

    def __init__(self, construct: ConstructSymbol, name: Hashable) -> None:

        super().__init__(ConstructType.parameter, construct, name)

    @property
    def construct(self) -> ConstructSymbol:

        return cast(ConstructSymbol, self.cid[0])

    @property
    def name(self) -> Hashable:

        return self.cid[1]


class ConstructSymbolPredicate(object):
    """
    A unary predicate that applies to construct symbols.

    Abbreviated as ConSymPred.

    The predicate may be defined intensionally (with respect to construct types 
    or arbitrary predicates) or extensionally (by enumeration of matching 
    constructs). ConstructSymbolPredicate objects are set-like in that they 
    support __contains__, may be extended (through addition) or contracted 
    (through removal). However, unlike sets, ConstructSymbolPredicate objects do 
    not support algebraic operators such as union, intersection, difference etc.

    ConstructSymbolPredicate objects are intended to facilitate checking if 
    simulated constructs satisfy certain complex conditions. Such checks may be 
    required, for example, to decide whether or not to connect two construct 
    realizers (see pyClarion.realizers). In general ConstructSymbolPredicate 
    objects may be used at any point where a (potentially complex) predicate 
    must be applied to construct symbols. 
    """

    def __init__(
        self, 
        ctype: ConstructType = None, 
        constructs: Iterable[ConstructSymbol] = None,
        predicates: Iterable[Callable[[ConstructSymbol], bool]] = None
    ) -> None:
        """
        Initialize a new Matcher instance.

        :param ctype: Acceptable construct type(s).
        :param constructs: Acceptable construct symbols.
        :param predicates: Custom custom predicates indicating acceptable 
            constructs. 
        """

        self.ctype = ConstructType.null_construct
        self.constructs: MutableSet[ConstructSymbol] = set()
        self.predicates: MutableSet[Callable[[ConstructSymbol], bool]] = set()
        self.add(ctype, constructs, predicates)

    def __contains__(self, key: ConstructSymbol) -> bool:
        """
        Return true if construct is in the match set.
        
        A construct is considered to be in the match set if:
            - Its construct symbol is in self.ctype OR
            - It is equal to a member of self.constructs OR
            - A predicate in self.predicates returns true when called on the 
              construct.
        """

        val = False
        val |= key.ctype in self.ctype
        val |= key in self.constructs
        for predicate in self.predicates:
                val |= predicate(key)
        return val

    def add(
        self, 
        ctype: ConstructType = None, 
        constructs: Iterable[ConstructSymbol] = None,
        predicates: Iterable[Callable[[ConstructSymbol], bool]] = None
    ) -> None:
        """
        Extend the set of accepted constructs.
        
        See Matcher.__init__ for argument descriptions.
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
        constructs: Iterable[ConstructSymbol] = None,
        predicates: Iterable[Callable[[ConstructSymbol], bool]] = None
    ) -> None:
        """
        Contract the set of accepted constructs.
        
        See Matcher.__init__ for argument descriptions.
        """

        if ctype is not None:
            self.ctype ^= ctype
        if constructs is not None:
            self.constructs ^= set(constructs)
        if predicates is not None:
            self.predicates ^= set(predicates)
        raise NotImplementedError()


# Abbreviated handle for ConstructSymbolPredicate.
ConSymPred = ConstructSymbolPredicate


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


def parameter(construct: ConstructSymbol, name: Hashable) -> ConstructSymbol:
    """
    Return a new parameter symbol.

    :param name: Name of behavior.
    """

    return ParameterSymbol(construct, name)


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
