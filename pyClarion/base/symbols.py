"""Tools for naming, indexing, and selecting constructs."""


__all__ = [
    "ConstructType", "Token", "Symbol", "SymbolicAddress", "SymbolTrie",
    "feature", "chunk", "rule", "chunks", "features", "flow_in", "flow_bt", 
    "flow_tb", "flow_tt", "flow_bb", "terminus", "updater", "buffer", 
    "subsystem", "agent", "lag", "validate_address", "expand_address", "dims", 
    "tags", "lags"
]


from typing import (
    Hashable, Tuple, Union, Iterable, Callable, Dict, TypeVar, Iterator, 
    List, Set, FrozenSet, Optional, Any, cast, overload
)
from enum import Flag, auto
from itertools import zip_longest


dim = Tuple[Hashable, int]
# Address for a construct w/in a simulated agent or component.
SymbolicAddress = Union["Symbol", Tuple["Symbol", ...]]
SymbolTrie = Any


class ConstructType(Flag):
    """
    Represents construct types within Clarion theory.

    Signals the role of a construct.

    Basic members (and interpretations):
        null_construct: Empty construct type (corresponds to flag null).
        feature: Feature node.
        chunk: Chunk node.
        rule: A rule.
        features: A pool of feature nodes.
        chunks: A pool of chunk nodes.
        flow_in: Activation input to a subsystem.
        flow_tb: Activation flow from top to bottom level.
        flow_bt: Activation flow from bottom to top level.
        flow_tt: Activation flow within top level.
        flow_bb: Activation flow within bottom level.
        terminus: Subsystem output.
        buffer: Temporary store of activations.
        subsystem: A Clarion subsystem.
        agent: A Clarion agent.

    Other members: 
        node: A chunk or (micro)feature.
        nodes: A pool of chunk or (micro)feature nodes.
        flow_bx: Flow originating in bottom level.
        flow_tx: Flow originating in top level.
        flow_xb: Flow ending in bottom level.
        flow_xt: Flow ending in top level.
        flow_h: Horizontal (intra-level) flow.
        flow_v: Vertical (inter-level) flow.
        flow: Links among (micro)feature and/or chunk nodes.
        basic_construct: A feature or chunk or flow or terminus or buffer. 
        container_construct: Subsystem or agent.
        any_construct: Matches any construct type.
    """

    null_construct = 0
    feature = auto()
    chunk = auto()
    rule = auto()
    features = auto()
    chunks = auto()
    flow_in = auto()
    flow_tb = auto()
    flow_bt = auto()
    flow_tt = auto()
    flow_bb = auto()
    terminus = auto()
    updater = auto()
    buffer = auto()
    subsystem = auto()
    agent = auto()

    node = feature | chunk
    nodes = features | chunks
    flow_bx = flow_bt | flow_bb 
    flow_tx = flow_tb | flow_tt 
    flow_xb = flow_tb | flow_bb 
    flow_xt = flow_bt | flow_tt 
    flow_h = flow_bb | flow_tt
    flow_v = flow_tb | flow_bt
    flow = flow_tb | flow_bt | flow_tt | flow_bb | flow_in
    basic_construct = node | nodes | flow | terminus | updater | buffer
    container_construct = subsystem | agent
    any_construct = basic_construct | container_construct


class Token(object):
    """
    A symbolic token. 
    
    Intended as a base class for constructing symbolic structures.
    
    Constructs a hashable object from hashable args. Supports '=='. Does not 
    support mutation.
    """

    __slots__ = ("_args")

    _args: Tuple[Hashable, ...]

    def __init__(self, *args: Hashable) -> None:

        super().__setattr__("_args", tuple(args))

    def __hash__(self) -> int:

        return hash(self._args)

    def __repr__(self) -> str:

        cls_name = type(self).__name__
        args = ", ".join(repr(item) for item in self._args)
        
        return "{}({})".format(cls_name, args)

    def __setattr__(self, name, value):

        cls_name = type(self).__name__
        msg = "Mutation of {} instance not permitted.".format(cls_name)
        
        raise AttributeError(msg)

    def __eq__(self, other: object) -> bool:

        if isinstance(other, Token):
            return self._args == other._args
        else:
            return NotImplemented

    def __lt__(self, other: object) -> bool:

        if isinstance(other, Token):
            return self._args < other._args
        else:
            return NotImplemented


class Symbol(Token):
    """
    Symbolic label for Clarion constructs.

    Consists of a construct type (see ConstructType) and an identifier.
    """

    __slots__ = ()

    def __init__(
        self, ctype: Union[ConstructType, str, int], cid: Hashable
    ) -> None:
        """
        Initialize a new Symbol.

        :param ctype: Construct type.
        :param cid: Hashable sequence serving as identifier.
        """

        if isinstance(ctype, str):
            ctype = ConstructType[ctype]
        elif isinstance(ctype, int):
            ctype = ConstructType(ctype)
        elif isinstance(ctype, ConstructType):
            pass
        else:
            msg = "Unexpected type {} for arg ctype."
            raise TypeError(msg.format(type(ctype).__name__))

        super().__init__(ctype, cid)

    def __repr__(self) -> str:

        cls_name = type(self).__name__
        
        return "{}({})".format(cls_name, repr(self.cid))

    @property
    def ctype(self) -> ConstructType:
        """Construct type associated with self."""

        return cast(ConstructType, self._args[0])

    @property
    def cid(self) -> Hashable:
        """Construct identifier associated with self."""

        return self._args[1] 


class feature(Symbol):
    """
    A feature symbol.

    Each feature is identified by a dimensional tag, a value, and a lag. By 
    default, the value is set to "" and the lag is set to 0.

    In pyClarion, the dimension of a feature is considered to be its dimension 
    tag together with its lag value. That is to say, two features initialized 
    with identical dimension tags but different lag values will be considered 
    to be of different dimensions.
    """

    __slots__ = ()

    def __init__(
        self, tag: Hashable, val: Hashable = "", lag: int = 0
    ) -> None:
        """
        Initialize a new feature symbol.

        :param tag: Dimension tag.
        :param val: Value of feature.
        :param lag: Lag indicator.
        """

        super().__init__("feature", ((tag, lag), val))

    def __repr__(self) -> str:

        cls_name = type(self).__name__

        _args = [repr(self.tag)]
        if self.val != "":
            _args.append(repr(self.val))
        if self.lag != 0:
            _args.append("lag={}".format(repr(self.lag)))

        args = ", ".join(_args)
        
        return "{}({})".format(cls_name, args)

    @property
    def dim(self):
        """Feature dimension, equal to (self.tag, self.lag)."""

        return self.cid[0]
    
    @property
    def val(self):
        """Feature value."""
        
        return self.cid[1]

    @property
    def tag(self):
        """Dimension tag."""
        
        return self.cid[0][0]

    @property
    def lag(self):
        """Lag value."""
        
        return self.cid[0][1]


class chunk(Symbol):
    """A chunk symbol."""

    __slots__ = ()

    def __init__(self, cid: Hashable) -> None:
        """
        Initialize a new chunk symbol.

        :param cid: Chunk identifier.
        """

        super().__init__("chunk", cid)


class rule(Symbol):
    """A rule symbol."""

    __slots__ = ()

    def __init__(self, cid: Hashable) -> None:
        """
        Initialize a new rule symbol.

        :param cid: Rule identifier.
        """

        super().__init__("rule", cid)


class features(Symbol):
    """A feature pool symbol."""

    __slots__ = ()

    def __init__(self, cid: Hashable) -> None:
        """
        Initialize a new feature pool symbol.

        :param cid: Name of feature node pool.
        """

        super().__init__("features", cid)


class chunks(Symbol):
    """A chunk pool symbol."""

    __slots__ = ()

    def __init__(self, cid: Hashable) -> None:
        """
        Initialize a new chunk pool symbol.

        :param cid: Name of chunk node pool.
        """

        super().__init__("chunks", cid)


class flow_in(Symbol):
    """An input flow symbol."""

    __slots__ = ()

    def __init__(self, cid: Hashable) -> None:
        """
        Initialize a new input flow symbol.

        :param cid: Flow identifier.
        """

        super().__init__("flow_in", cid)


class flow_bt(Symbol):
    """A bottom-up flow symbol."""

    __slots__ = ()

    def __init__(self, cid: Hashable) -> None:
        """
        Initialize a new bottom-up flow symbol.

        :param cid: Flow identifier.
        """

        super().__init__("flow_bt", cid)


class flow_tb(Symbol):
    """A top-down flow symbol."""

    __slots__ = ()

    def __init__(self, cid: Hashable) -> None:
        """
        Initialize a new top-down flow symbol.

        :param cid: Flow identifier.
        """

        super().__init__("flow_tb", cid)


class flow_tt(Symbol):
    """A top level flow symbol."""

    __slots__ = ()

    def __init__(self, cid: Hashable) -> None:
        """
        Initialize a new top-level flow symbol.

        :param cid: Flow identifier.
        """

        super().__init__("flow_tt", cid)


class flow_bb(Symbol):
    """A bottom level flow symbol."""

    __slots__ = ()

    def __init__(self, cid: Hashable) -> None:
        """
        Initialize a new bottom level flow symbol.

        :param cid: Flow identifier.
        """

        super().__init__("flow_bb", cid)


class terminus(Symbol):
    """A terminus symbol."""

    __slots__ = ()

    def __init__(self, cid: Hashable) -> None:
        """
        Initialize a new terminus symbol.

        :param cid: Terminus identifier.
        """

        super().__init__("terminus", cid)


class updater(Symbol):
    """An updater symbol."""

    __slots__ = ()

    def __init__(self, cid: Hashable) -> None:
        """
        Initialize a new updater symbol.

        :param cid: Updater identifier.
        """

        super().__init__("updater", cid)


class buffer(Symbol):
    """A buffer symbol."""

    __slots__ = ()

    def __init__(self, cid) -> None:
        """
        Initialize a new buffer symbol.

        :param cid: Buffer identifier.
        """

        super().__init__("buffer", cid)


class subsystem(Symbol):
    """A subsystem symbol."""

    __slots__ = ()

    def __init__(self, cid) -> None:
        """
        Initialize a new subsystem symbol.

        :param cid: Subsystem identifier.
        """

        super().__init__("subsystem", cid)


class agent(Symbol):
    """An agent symbol."""

    __slots__ = ()

    def __init__(self, cid) -> None:
        """
        Initialize a new agent symbol.

        :param cid: Agent identifier.
        """

        super().__init__("agent", cid)


######################
### SYMBOLIC PATHS ###
######################


# These tuples represent allowable construct locators.
# TODO: Need to handle lone subsystems.
PATTERNS = [
    ("agent",),
    ("agent", "buffer"),
    ("agent", "updater"),
    ("agent", "subsystem"),
    ("agent", "subsystem", "features"),
    ("agent", "subsystem", "chunks"),
    ("agent", "subsystem", "flow"),
    ("agent", "subsystem", "terminus"),
    ("agent", "subsystem", "updater")
]


def validate_address(address: SymbolicAddress, strict: bool = False) -> None:
    """
    Check if construct address matches a valid pattern.

    Throws ValueError if address is invalid.
    
    :param address: Target address.
    :param strict: Option to reject partial addresses.
    """

    if isinstance(address, Symbol):
        address = (address,)

    seq = [symbol.ctype for symbol in address]
    cutoff = 0 if strict else -len(seq)
    stubs = [
        tuple([ConstructType[name] for name in path[cutoff:]])
        for path in PATTERNS
    ]
    
    for i, stub in enumerate(stubs):
        pairs = zip_longest(seq, stub, fillvalue=ConstructType.null_construct)
        if all(x in ref for x, ref in pairs):
            break
    else:
        raise ValueError("Address does not match any pattern.")


def expand_address(
    base: Tuple[Symbol, ...], partial: SymbolicAddress
) -> SymbolicAddress:
    """
    Return the nearest full address relative to base given a partial address.

    The nearest full address is...
    """

    validate_address(address=base, strict=True)

    if len(base) == 0:
        return partial

    if isinstance(partial, Symbol):
        partial = (partial,)

    seq = [symbol.ctype for symbol in partial]

    stubs = [
        tuple([ConstructType[name] for name in path[-len(seq):]])
        for path in PATTERNS
    ]

    candidates = []
    for i, stub in enumerate(stubs):
        split = len(PATTERNS[i]) - len(seq)
        pairs = zip_longest(seq, stub, fillvalue=ConstructType.null_construct)
        if split < len(base) and all(x in ref for x, ref in pairs):
            candidates.append((i, split))

    if len(candidates) == 0:
        msg = "Address {} does not match any pattern."
        raise ValueError(msg.format(partial))
    else:
        assert len(candidates) == 1, "Multiple patterns matched."
        (i, split), = candidates
        return base[:split] + partial


#########################
### UTILITY FUNCTIONS ###
#########################


def lag(f: feature, val: int = 1) -> feature:
    """Return a copy of feature with lag incremented by val."""
    
    return feature(f.tag, f.val, f.lag + val)


@overload
def dims(fs: Tuple[feature, ...]) -> Tuple[dim, ...]:
    ...

@overload
def dims(fs: List[feature]) -> List[dim]:
    ...

@overload
def dims(fs: Set[feature]) -> Set[dim]:
    ...

@overload
def dims(fs: FrozenSet[feature]) -> FrozenSet[dim]:
    ...

def dims(fs):
    """Extract dims from a collection of features."""

    return type(fs)(f.dim for f in fs)


@overload
def tags(fs: Tuple[feature, ...]) -> Tuple[Hashable, ...]:
    ...

@overload
def tags(fs: List[feature]) -> List[Hashable]:
    ...

@overload
def tags(fs: Set[feature]) -> Set[Hashable]:
    ...

@overload
def tags(fs: FrozenSet[feature]) -> FrozenSet[Hashable]:
    ...

def tags(fs):
    """Extract tags from a collection of features."""

    return type(fs)(f.tag for f in fs)


@overload
def lags(fs: Tuple[feature, ...]) -> Tuple[int, ...]:
    ...

@overload
def lags(fs: List[feature]) -> List[int]:
    ...

@overload
def lags(fs: Set[feature]) -> Set[int]:
    ...

@overload
def lags(fs: FrozenSet[feature]) -> FrozenSet[int]:
    ...

def lags(fs):
    """Extract lags from a collection of features."""

    return type(fs)(f.lag for f in fs)
