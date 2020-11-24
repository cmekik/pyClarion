"""Tools for naming, indexing, and selecting constructs."""


__all__ = [
    "ConstructType", "Token", "Symbol", "ConstructRef", "feature", 
    "chunk", "rule", "chunks", "features", "flow_in", "flow_bt", "flow_tb", 
    "flow_tt", "flow_bb", "terminus", "buffer", "subsystem", "agent", 
    "group_by", "group_by_ctype", "group_by_dims", "group_by_tags", 
    "group_by_lags", "lag"
]


from enum import Flag, auto
from typing import Hashable, Tuple, Union, Iterable, Callable, Dict, TypeVar


class ConstructType(Flag):
    """
    Represents construct types within Clarion theory.

    Signals the role of a construct for controlling processing logic.

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
        nodes: A pool of chunk or microfeature nodes.
        flow_bx: Flow originating in bottom level.
        flow_tx: Flow originating in top level.
        flow_xb: Flow ending in bottom level.
        flow_xt: Flow ending in top level.
        flow_h: Horizontal (intra-level) flow.
        flow_v: Vertical (inter-level) flow.
        flow: Links among (micro)feature and/or chunk nodes.
        basic_construct: A feature or chunk or flow or terminus or buffer. 
        container_construct: Subsystem or agent.
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
    basic_construct = node | flow | terminus | buffer
    container_construct = subsystem | agent


class Token(object):
    """
    A symbolic token. 
    
    Intended as a base class for constructing symbolic structures.
    
    Constructs a hashable object from hashable args. Supports '=='. Does not 
    support mutation.
    """

    __slots__ = ("_args")

    def __init__(self, *args: Hashable):

        super().__setattr__("_args", tuple(args))

    def __hash__(self):

        return hash(self._args)

    def __repr__(self):

        cls_name = type(self).__name__
        args = ", ".join(repr(item) for item in self._args)
        
        return "{}({})".format(cls_name, args)

    def __setattr__(self, name, value):

        cls_name = type(self).__name__
        msg = "Mutation of {} instance not permitted.".format(cls_name)
        
        raise AttributeError(msg)

    def __eq__(self, other):

        if isinstance(other, Token):
            return self._args == other._args
        else:
            return NotImplemented

    def __lt__(self, other):

        if isinstance(other, Token):
            return self._args < other._args
        else:
            return NotImplemented


class Symbol(Token):
    """
    Symbol for naming Clarion constructs.

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
            # do nothing, all good.
            pass
        else:
            raise TypeError(
                "Unexpected type {} for arg ctype.".format(
                    type(ctype).__name__
                )
            )

        super().__init__(ctype, cid)

    def __repr__(self):

        cls_name = type(self).__name__
        
        return "{}({})".format(cls_name, repr(self.cid))

    @property
    def ctype(self) -> ConstructType:
        """Construct type associated with self."""

        # NOTE: mypy complains that self._args not defined. So, ignore. - Can
        return self._args[0] # type: ignore

    @property
    def cid(self) -> Tuple[Hashable, ...]:
        """Construct identifier associated with self."""

        # NOTE: mypy complains that self._args not defined. So, ignore. - Can
        return self._args[1] # type: ignore


# Address for a construct w/in a simulated agent or component.
ConstructRef = Union[Symbol, Tuple[Symbol, ...]]


class feature(Symbol):
    """
    A feature symbol.

    Each feature is identified by a dimension label, a value, and a lag. By 
    default, the lag is set to 0.

    In pyClarion, the dimension of a feature is considered to be its dimension 
    label together with its lag value. That is to say, two features initialized 
    with identical dimension labels but different lag values will be considered 
    to be of different dimensions.
    """

    __slots__ = ()

    def __init__(
        self, tag: Hashable, val: Hashable = "", lag: int = 0
    ) -> None:
        """
        Initialize a new feature symbol.

        :param tag: Dimension label.
        :param val: Value of feature.
        :param lag: Lag indicator.
        """

        super().__init__("feature", ((tag, lag), val))

    def __repr__(self):

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
        """Dimension label."""
        
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
    """A pool of feature nodes."""

    __slots__ = ()

    def __init__(self, cid: Hashable) -> None:
        """
        Initialize a new feature pool symbol.

        :param cid: Name of feature node pool.
        """

        super().__init__("features", cid)


class chunks(Symbol):
    """A pool of chunk nodes."""

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


################
### FUNCTIONS ##
################


T = TypeVar("T")
K = TypeVar("K")
def group_by(
    iterable: Iterable[T], key: Callable[[T], K]
) -> Dict[K, Tuple[T, ...]]:
    """Return a dict grouping items in iterable by values of the key func."""

    groups: dict = {}
    for item in iterable:
        k = key(item)
        groups.setdefault(k, []).append(item)
    
    return {k: tuple(v) for k, v in groups.items()}


def group_by_ctype(
    symbols: Iterable[Symbol]
) -> Dict[ConstructType, Tuple[Symbol, ...]]:
    """
    Construct a dict grouping symbols by their construct types.
    
    Returns a dict where each construct type is mapped to a tuple of symbols of 
    that type. Does not check for duplicates.
    """

    # Ignore type of key due to mypy false alarm. - Can
    key = Symbol.ctype.fget # type: ignore 
    
    return group_by(iterable=symbols, key=key)


def group_by_dims(
    features: Iterable[feature]
) -> Dict[Hashable, Tuple[feature, ...]]:
    """
    Construct a dict grouping features by their dimensions.
    
    Returns a dict where each dim is mapped to a tuple of features of that dim.
    Does not check for duplicate features.

    :param features: An iterable of features to be grouped by dimension.
    """

    # Ignore type of key due to mypy false alarm. - Can
    key = feature.dim.fget # type: ignore 
    
    return group_by(iterable=features, key=key)


def group_by_tags(
    features: Iterable[feature]
) -> Dict[Hashable, Tuple[feature, ...]]:
    """
    Construct a dict grouping features by their dimensions.
    
    Returns a dict where each dim is mapped to a tuple of features of that dim.
    Does not check for duplicate features.

    :param features: An iterable of features to be grouped by dimension.
    """

    # Ignore type of key due to mypy false alarm. - Can
    key = feature.tag.fget # type: ignore 
    
    return group_by(iterable=features, key=key)


def group_by_lags(
    features: Iterable[feature]
) -> Dict[Hashable, Tuple[feature, ...]]:
    """
    Construct a dict grouping features by their dimensions.
    
    Returns a dict where each dim is mapped to a tuple of features of that dim.
    Does not check for duplicate features.

    :param features: An iterable of features to be grouped by dimension.
    """

    # Ignore type of key due to mypy false alarm. - Can
    key = feature.lag.fget # type: ignore 
    
    return group_by(iterable=features, key=key)


def lag(f: feature, val: int = 1) -> feature:
    """Return a copy of feature with lag incremented by val."""
    
    return feature(f.tag, f.val, f.lag + val)
