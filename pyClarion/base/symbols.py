"""Tools for naming, indexing, and selecting constructs."""


__all__ = [
    "ConstructType", "Token", "Symbol", "ConstructRef", "MatchSet", "feature", 
    "chunk", "rule", "chunks", "features", "flow_in", "flow_bt", "flow_tb", 
    "flow_tt", "flow_bb", "terminus", "buffer", "subsystem", "agent"
]


from enum import Flag, auto
from typing import Hashable, Tuple, Union, Iterable, Callable, MutableSet


class ConstructType(Flag):
    """
    Represents construct types within Clarion theory.

    Signals the role of a construct for controlling processing logic.

    Basic members (and interpretations):
        null_construct: Empty construct type (corresponds to flag null).
        feature: Feature node.
        chunk: Chunk node.
        rule: Rule node.
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
    
    Constructs a hashable object from hashable args. Supports '==' and '<'. 
    Does not support mutation.
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
            return False
    
    def __lt__(self, other):

        if isinstance(other, Token):
            return self._args < other._args
        else:
            template = "'<' not supported between instances of '{}' and '{}'."
            msg = template.format(type(self).__name__, type(other).__name__)
            raise TypeError(msg)


class Symbol(Token):
    """
    Symbol for naming Clarion constructs.

    Consists of a construct type (see ConstructType) and an identifier.
    """

    __slots__ = ()

    def __init__(
        self, ctype: Union[ConstructType, str, int], *cid: Hashable
    ) -> None:
        """
        Initialize a new Symbol.

        :param ctype: Construct type.
        :param cid: Hashable sequence serving as identifier.
        """

        if len(cid) == 0:
            raise ValueError("Must pass at least one identifier.")

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
        args = ", ".join(repr(item) for item in self.cid)
        
        return "{}({})".format(cls_name, args)

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

    def __init__(self, tag: Hashable, val: Hashable, lag: int = 0) -> None:
        """
        Initialize a new feature symbol.

        :param tag: Dimension label.
        :param val: Value of feature.
        :param lag: Lag indicator.
        """

        super().__init__("feature", (tag, lag), val)

    def __repr__(self):

        cls_name = type(self).__name__
        args = ", ".join(map(repr, (self.tag, self.val, self.lag)))
        
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
        Initialize a new chunk symbol.

        :param cid: Chunk identifier.
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

    def __init__(self, cid) -> None:
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


class MatchSet(object):
    """
    Matches construct symbols.

    Checks if construct symbols satisfy complex conditions. Supports checks 
    against construct types, reference symbol sets, or arbitrary predicates. 
    
    Supports usage with 'in' and addition/removal of criteria. Does NOT support 
    algebraic operators such as union, intersection, difference etc.
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
        Return true iff construct is in the match set.
        
        A construct is considered to be in the match set iff:
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
