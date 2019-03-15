"""Tools for defining the behavior of constructs within simulations."""


# Notes for Readers:

#   - Type aliases and helper functions and classes are defined first.
#   - There are two major types of construct realizer: basic construct 
#     realizers and container construct realizers. Definitions for each major 
#     realizer type are grouped together in marked sections.
#   - Last, some factory functions are defined for quick initialization of 
#     construct realizers.


__all__ = [
    "ConstructRealizer", "BasicConstructRealizer", "NodeRealizer", 
    "FlowRealizer", "ResponseRealizer", "BehaviorRealizer", "BufferRealizer",
    "ContainerConstructRealizer", "SubsystemRealizer", "AgentRealizer",
    "make_realizer", "make_subsystem", "make_agent"
]


from typing import (
    Any, Callable, Iterable, MutableMapping, Union, Optional, ClassVar, Text, 
    Type, Dict, Tuple, Iterator, Hashable, List, Mapping, Sequence, cast, Container
)
from operator import getitem, setitem, delitem
from itertools import combinations
from pyClarion.base.symbols import ConstructSymbol, ConstructType
from pyClarion.base.packets import (
    ConstructSymbolMapping, ActivationPacket, DecisionPacket, 
    ConstructSymbolCollection
)


# Types aliases used by helper classes

Packet = Union[ActivationPacket, DecisionPacket]
PullMethod = Union[
    Callable[[], ActivationPacket], Callable[[], DecisionPacket]
]
InputMapping = MutableMapping[ConstructSymbol, PullMethod]
PacketIterable = Union[Iterable[ActivationPacket], Iterable[DecisionPacket]]
WatchMethod = Callable[[ConstructSymbol, PullMethod], None]
DropMethod = Callable[[ConstructSymbol], None]
PullRule = Union[Container, ConstructType]

# Types used by ConstructRealizer instances

ComponentSpec = Iterable[str] 

# Types used by BasicConstructRealizer instances

Channel = Callable[[ConstructSymbolMapping], ConstructSymbolMapping]
Junction = Callable[[Iterable[ActivationPacket]], ConstructSymbolMapping]
Selector = Callable[
    [ConstructSymbolMapping], 
    Tuple[ConstructSymbolMapping, ConstructSymbolCollection]
]
Effector = Callable[[DecisionPacket], None]
Source = Callable[[], ConstructSymbolMapping]

# Types used by ContainerConstructRealizer instances

ConstructIndex = Union[ConstructSymbol, Tuple[ConstructSymbol, ...]]
MissingSpec = List[Tuple[ConstructIndex, ComponentSpec]]
ConstructSymbolTuple = Tuple[ConstructSymbol, ...]
MutableRealizerMapping = MutableMapping[ConstructSymbol, 'ConstructRealizer']
ConstructSymbolList = List[ConstructSymbol]
ConstructSymbolIterable = Iterable[ConstructSymbol]
ContainerConstructItems = Iterable[Tuple[ConstructSymbol, 'ConstructRealizer']]
HasInput = Union[
    'NodeRealizer', 'FlowRealizer', 'ResponseRealizer', 'BehaviorRealizer', 
    'SubsystemRealizer'
]
HasOutput = Union[
    'NodeRealizer', 'FlowRealizer', 'ResponseRealizer', 'BufferRealizer', 
]
PropagationRule = Callable[['SubsystemRealizer'], None]
Updater = Callable[[], None]
UpdaterList = List[Updater]
UpdaterIterable = Iterable[Callable[[], None]]


######################
### Helper Classes ###
######################


# The classes defined below support networking of basic construct realizers. 
# These classes allow basic construct realizers to listen for and emit 
# activation packets.


class InputMonitor(object):
    """Listens for basic construct realizer outputs."""

    def __init__(self) -> None:

        self.input_links: InputMapping = {}

    def pull(self) -> PacketIterable:
        """Pull activations from input constructs."""

        for view in self.input_links.values():
            v = view()
            if v is not None: 
                yield v

    def watch(self, csym: ConstructSymbol, pull_method: PullMethod) -> None:
        """Connect given construct as input to client."""

        self.input_links[csym] = pull_method

    def drop(self, csym: ConstructSymbol):
        """Disconnect given construct from client."""

        del self.input_links[csym]


class OutputView(object):
    """Exposes outputs of basic construct realizers."""

    def update(self, packet: Packet) -> None:
        """Update reported output of client construct."""
        
        self._buffer = packet

    def view(self) -> Optional[Packet]:
        """Emit current output of client construct."""
        
        try:
            return self._buffer
        except AttributeError:
            return None

    def clear(self) -> None:
        """Clear output buffer.""" 

        del self._buffer       


class SubsystemInputMonitor(object):
    """Listens for buffer outputs."""

    def __init__(self, watch: WatchMethod, drop: DropMethod) -> None:
        """
        Initialize a SubsystemInputMonitor.

        :param watch: Callable for informing client subsystem members to watch  
            a new input to client.
        :param drop: Callable for informing client subsystem members to drop an 
            input to client.
        """

        self.input_links: InputMapping = {}
        self._watch = watch
        self._drop = drop

    def watch(self, csym: ConstructSymbol, pull_method: PullMethod) -> None:
        """Connect given construct as input to client."""

        self.input_links[csym] = pull_method
        self._watch(csym, pull_method)

    def drop(self, csym: ConstructSymbol):
        """Disconnect given construct from client."""

        del self.input_links[csym]
        self._drop(csym)


######################################
### Construct Realizer Definitions ###
######################################


class ConstructRealizer(object):
    """
    Base class for construct realizers.

    Construct realizers are responsible for implementing the behavior of client 
    constructs. 
    """

    ctype: ClassVar[ConstructType] = ConstructType.null_construct
    __slots__: ComponentSpec = ("csym",)

    def __init__(self, csym: ConstructSymbol) -> None:
        """Initialize a new construct realizer.
        
        :param csym: Symbolic representation of client construct.
        """

        self._check_csym(csym)
        self.csym = csym

    def __repr__(self) -> Text:

        return "<{}: {}>".format(self.__class__.__name__, str(self.csym))

    def propagate(self) -> None:
        """Propagate activations."""

        raise NotImplementedError()

    def pulls_from(self, source: ConstructSymbol) -> bool:
        """Return true if self pulls information from source."""

        raise NotImplementedError()

    def clear_activations(self) -> None:
        """Clear activations."""

        raise NotImplementedError()

    def ready(self) -> bool:
        """Return true iff all necessary components have been defined."""

        return all(hasattr(self, component) for component in self.__slots__)

    def missing(self) -> ComponentSpec:
        """Return any missing components of self."""

        return tuple(
            component for component in self.__slots__ 
            if not hasattr(self, component)
        )

    def _check_csym(self, csym: ConstructSymbol) -> None:
        """Check if construct symbol matches realizer."""

        if csym.ctype not in type(self).ctype:
            raise ValueError(
                " ".join(
                    [   
                        type(self).__name__,
                        "expects construct symbol with ctype",
                        repr(type(self).ctype),
                        "but received symbol {} of ctype {}.".format(
                            str(csym), repr(csym.ctype)
                        )
                    ]
                )
            )


### Basic Construct Realizer Definitions ###


class BasicConstructRealizer(ConstructRealizer):
    """Base class for basic construct realizers."""

    itype: Optional[Type] = InputMonitor
    otype: Optional[Type] = OutputView

    def __init__(
        self, csym: ConstructSymbol, pull_rule: PullRule = None
    ) -> None:

        super().__init__(csym)

        # Initialize I/O
        if self.itype is not None:
            self.input = self.itype()
        if self.otype is not None:
            self.output = self.otype()
        self.pull_rule = pull_rule

    def pulls_from(self, source: ConstructSymbol) -> bool:
        """Check if self may pull data from given construct."""

        if self.pull_rule is not None and isinstance(
            self.pull_rule, ConstructType
        ):
            return source.ctype in self.pull_rule
        elif self.pull_rule is not None:
            return source in self.pull_rule
        else:
            return False

    def clear_activations(self) -> None:
        """Clear activations stored in output view."""

        try:
            self.output.clear()
        except AttributeError:
            pass


class NodeRealizer(BasicConstructRealizer):

    ctype = ConstructType.node
    __slots__ = ('junction',)

    def __init__(
        self, 
        csym: ConstructSymbol, 
        pull_rule: PullRule = None, 
        junction: Junction = None
    ) -> None:

        super().__init__(csym, pull_rule)
        if junction is not None: 
            self.junction = junction

    def propagate(self) -> None:
        """Output current strength of node."""

        strengths = self.junction(self.input.pull())
        packet = ActivationPacket(strengths=strengths, origin=self.csym)
        self.output.update(packet)


class FlowRealizer(BasicConstructRealizer):

    ctype = ConstructType.flow
    __slots__ = ('junction', 'channel')

    def __init__(
        self, 
        csym: ConstructSymbol,
        pull_rule: PullRule = None, 
        junction: Junction = None, 
        channel: Channel = None
    ) -> None:

        super().__init__(csym, pull_rule)
        if junction is not None: 
            self.junction = junction
        if channel is not None: 
            self.channel = channel

    def propagate(self) -> None:
        """Compute new node activations."""

        combined = self.junction(self.input.pull())
        strengths = self.channel(combined)
        packet = ActivationPacket(strengths=strengths, origin=self.csym)
        self.output.update(packet)


class ResponseRealizer(BasicConstructRealizer):

    ctype = ConstructType.response
    __slots__ = ('junction', 'selector')

    def __init__(
        self, 
        csym: ConstructSymbol, 
        pull_rule: PullRule = None,
        junction: Junction = None, 
        selector: Selector = None
    ) -> None:

        super().__init__(csym, pull_rule)
        if junction is not None: 
            self.junction = junction
        if selector is not None: 
            self.selector = selector

    def propagate(self) -> None:
        """Make and output a decision."""

        combined = self.junction(self.input.pull())
        strengths, chosen = self.selector(combined)
        decision_packet = DecisionPacket(
            strengths=strengths, chosen=chosen, origin=self.csym
        )
        self.output.update(decision_packet)


class BehaviorRealizer(BasicConstructRealizer):
    
    ctype = ConstructType.behavior
    otype = None
    __slots__ = ('effector',)

    def __init__(
        self, 
        csym: ConstructSymbol, 
        pull_rule: PullRule = None, 
        effector: Effector = None
    ) -> None:

        super().__init__(csym, pull_rule)
        if effector is not None: 
            self.effector = effector

    def propagate(self) -> None:
        """Execute selected callbacks."""

        self.effector(*self.input.pull())


class BufferRealizer(BasicConstructRealizer):

    ctype = ConstructType.buffer
    itype = None
    __slots__ = ('source',)

    def __init__(
        self, 
        csym: ConstructSymbol, 
        pull_rule: PullRule = None, 
        source: Source = None
    ) -> None:

        super().__init__(csym)
        if source is not None: 
            self.source = source

    def propagate(self) -> None:
        """
        Output stored activation pattern.
        
        .. warning:
           Output activation packet is constructed directly from output of 
           self.source. If source output is mutated (e.g., as part of a buffer 
           update), it *will* be reflected in the output packet. Possible cause 
           of unexpected behavior.
        """

        strengths = self.source()
        packet = ActivationPacket(strengths=strengths, origin=self.csym)
        self.output.update(packet)


### Container Construct Realizers ###


class ContainerConstructRealizer(MutableRealizerMapping, ConstructRealizer):
    """Base class for container construct realizers."""

    def __init__(self, csym: ConstructSymbol) -> None:

        super().__init__(csym)
        self._dict: Dict = dict()

    def __len__(self) -> int:

        return len(self._dict)

    def __contains__(self, obj: Any) -> bool:

        return obj in self._dict

    def __iter__(self) -> Iterator:

        return iter(self._dict)

    def __getitem__(self, key: Any) -> Any:

        if isinstance(key, ConstructSymbol):
            return self._dict[key]
        else:
            return self._consume_multiindex(
                key[:-1], lambda a: getitem(a, key[-1])
            )

    def __setitem__(self, key: Any, value: Any) -> None:

        if isinstance(key, ConstructSymbol):
            self._check_kv_pair(key, value)
            self._dict[key] = value
            self._connect(key, value)
        else:
            self._consume_multiindex(
                key[:-1], lambda a: setitem(a, key[-1], value)
            )

    def __delitem__(self, key: Any) -> None:

        if isinstance(key, ConstructSymbol):
            del self._dict[key]
            self._disconnect(key)
        else:
            self._consume_multiindex(key[:-1], lambda a: delitem(a, key[-1]))

    def execute(self) -> None:
        """Execute selected actions."""

        raise NotImplementedError()

    def may_contain(self, csym: ConstructSymbol) -> bool:
        """Return true if container construct may contain csym."""
        
        raise NotImplementedError()

    def may_connect(
        self, source: ConstructSymbol, target: ConstructSymbol
    ) -> bool:
        """Return true if source may send output to target."""

        raise NotImplementedError()

    def ready(self) -> bool:
        "Return true iff all necessary components defined for self and members."

        return super().ready() and all(r.ready() for r in self.values())

    def clear_activations(self) -> None:
        """Clear member activations"""

        for realizer in self.values():
            realizer.clear_activations()

    def missing_recursive(self) -> MissingSpec:
        """Return missing components in self and all member realizers."""

        missing: MissingSpec = [(self.csym, self.missing())]
        for csym, realizer in self.items():
            if isinstance(realizer, ContainerConstructRealizer):
                missing.extend(
                    (self._make_compound_index(index), missing_slots) 
                    for index, missing_slots in realizer.missing_recursive()
                )
            elif isinstance(realizer, BasicConstructRealizer):
                missing.append(((self.csym, csym), realizer.missing()))
        return missing

    def insert_realizers(self, *realizers: ConstructRealizer) -> None:
        """Add pre-initialized realizers to self."""

        for realizer in realizers:
            self[realizer.csym] = realizer

    def iter_ctype(self, ctype: ConstructType) -> ConstructSymbolIterable:
        """Return an iterator over all members matching ctype."""

        for csym in self: 
            if csym.ctype in ctype:
                yield csym

    def items_ctype(self, ctype: ConstructType) -> ContainerConstructItems:
        """Return an iterator over all csym-realizer pairs matching ctype."""

        for csym, realizer in self.items():
            if csym.ctype in ctype:
                yield csym, realizer

    def make_links(self) -> None:
        """Relink all constructs within self."""

        for r1, r2 in combinations(self.values(), 2):
            if r1.pulls_from(r2.csym):
                cast(HasInput, r1).input.watch(
                    r2.csym, cast(HasOutput, r2).output.view
                )
            if r2.pulls_from(r1.csym):
                cast(HasInput, r2).input.watch(
                    r1.csym, cast(HasOutput, r1).output.view
                )

    def _check_kv_pair(self, key: Any, value: Any) -> None:

        if not self.may_contain(key):
            raise ValueError(
                "{} may not contain {}; forbidden construct type.".format(
                    repr(self), repr(key)
                )
            )
        if key != value.csym:
            raise ValueError(
                "{} given key {} does not match value {}".format(
                    repr(self), repr(key), repr(value)
                )
            )

    def _consume_multiindex(
            self, multiindex: ConstructSymbolTuple, func: Callable[[Any], Any]
        ) -> Any:
        # type annotations need improvement.

        a = self
        for csym in multiindex:
            a = cast(ContainerConstructRealizer, a[csym])
        return func(a)

    def _connect(self, key: Any, value: Any) -> None:

        for csym, realizer in self.items():
            if value.pulls_from(csym):
                # EXPLAIN!!!!!!
                value = cast(HasInput, value)
                realizer = cast(HasOutput, realizer)
                value.input.watch(csym, realizer.output.view)
            if realizer.pulls_from(key):
                value = cast(HasOutput, value)
                realizer = cast(HasInput, realizer)
                realizer.input.watch(key, value.output.view)

    def _disconnect(self, key: Any) -> None:

        for csym, realizer in self.items():
            if self.may_connect(key, csym):
                cast(HasInput, realizer).input.drop(key)

    def _make_compound_index(self, index: ConstructIndex) -> ConstructIndex:
        
        if isinstance(index, ConstructSymbol):
            return (self.csym, index)
        else: # index must be a tuple of construct symbols
            return (self.csym,) + cast(Tuple[ConstructSymbol, ...], index)


class SubsystemRealizer(ContainerConstructRealizer):
    """A network of node, flow, apprasial, and behavior realizers."""

    ctype = ConstructType.subsystem
    itype = SubsystemInputMonitor
    __slots__ = ('propagation_rule', 'pull_rule')

    def __init__(
        self, 
        csym: ConstructSymbol, 
        pull_rule: PullRule = None, 
        propagation_rule: PropagationRule = None, 
    ) -> None:
        """
        Initialize a new subsystem realizer.
        
        :param construct: Client subsystem.
        :param propagation_rule: Function implementing desired activation 
            propagation sequence. Should expect a single SubsystemRealizer as 
            argument. 
        """

        super().__init__(csym)
        self.input = type(self).itype(self._watch, self._drop)
        self.pull_rule = pull_rule
        if propagation_rule is not None: 
            self.propagation_rule = propagation_rule

    def propagate(self) -> None:
        """Propagate activations among realizers owned by self."""

        self.propagation_rule(self)

    def execute(self) -> None:
        """Fire all selected actions."""

        for behavior in self.behaviors:
            self[behavior].propagate()

    def pulls_from(self, source: ConstructSymbol) -> bool:
        """
        Return true iff source may connect to target within self.
        
        Connects buffers to subsystems according to specifications in buffer 
        construct symbols.
        """

        if self.pull_rule is not None and isinstance(
            self.pull_rule, ConstructType
        ):
            return source.ctype in self.pull_rule
        elif self.pull_rule is not None:
            return source in self.pull_rule
        else:
            return False

    def may_contain(self, csym: ConstructSymbol) -> bool:
        """Return true if subsystem realizer may contain construct symbol."""

        return bool(
            csym.ctype & (
                ConstructType.node |
                ConstructType.flow |
                ConstructType.response |
                ConstructType.behavior
            )            
        )

    @property
    def nodes(self) -> ConstructSymbolList:
        
        return list(self.iter_ctype(ConstructType.node))

    @property
    def flows(self) -> ConstructSymbolList:
        
        return list(self.iter_ctype(ConstructType.flow))

    @property
    def responses(self) -> ConstructSymbolList:
        
        return list(self.iter_ctype(ConstructType.response))

    @property
    def behaviors(self) -> ConstructSymbolList:
        
        return list(self.iter_ctype(ConstructType.behavior))

    def _connect(self, key: Any, value: Any) -> None:

        super()._connect(key, value)

        if key.ctype in ConstructType.node:
            for buffer, pull_method in self.input.input_links.items():
                value.input.watch(buffer, pull_method)

    def _watch(
        self, identifier: ConstructSymbol, pull_method: PullMethod
    ) -> None:
        """Informs members of a new input to self."""

        for realizer in self.values():
            if cast(BasicConstructRealizer, realizer).pulls_from(identifier):
                cast(BasicConstructRealizer, realizer).input.watch(
                    identifier, pull_method
                )

    def _drop(self, identifier: ConstructSymbol) -> None:
        """Informs members of a dropped input to self."""

        for realizer in self.values():
            if cast(BasicConstructRealizer, realizer).pulls_from(identifier):
                cast(BasicConstructRealizer, realizer).input.drop(identifier)


class AgentRealizer(ContainerConstructRealizer):
    """Realizer for Agent constructs."""

    ctype = ConstructType.agent

    def __init__(self, csym: ConstructSymbol) -> None:
        """
        Initialize a new agent realizer.
        
        :param construct: Client agent.
        """

        super().__init__(csym)
        self._updaters: UpdaterList = []

    def propagate(self) -> None:
        """Propagate activations among realizers owned by self."""
        
        for buffer in self.buffers:
            self[buffer].propagate()
        for subsystem in self.subsystems:
            self[subsystem].propagate()

    def execute(self) -> None:
        """Execute all selected actions in all subsystems."""

        for subsystem in self.subsystems:
            cast(SubsystemRealizer, self[subsystem]).execute()

    def may_contain(self, csym: ConstructSymbol) -> bool:
        """Return true if agent realizer may contain csym."""

        return csym.ctype in ConstructType.subsystem | ConstructType.buffer

    def learn(self) -> None:
        """
        Update knowledge in all subsystems and all buffers.
        
        Issues update calls to each updater attached to self.  
        """

        for updater in self.updaters:
            updater()

    def attach(self, *updaters: Updater) -> None:
        """
        Add update managers to self.
        
        :param update_managers: Callables that manage updates to dynamic 
            knowledge components. Should take no arguments and return nothing.
        """

        for updater in updaters:
            self._updaters.append(updater)

    def make_links(self) -> None:

        super().make_links()
        for subsystem in self.subsystems:
            self[subsystem].make_links() 

    @property
    def updaters(self) -> UpdaterList:
        
        return list(updater for updater in self._updaters)

    @property
    def buffers(self) -> ConstructSymbolList:

        return list(self.iter_ctype(ConstructType.buffer))

    @property
    def subsystems(self) -> ConstructSymbolList:

        return list(self.iter_ctype(ConstructType.subsystem))


####################################
### Construct Realizer Factories ###
####################################


def make_realizer(csym: ConstructSymbol) -> ConstructRealizer:
    """Initialize empty construct realizer for given construct symbol."""

    if csym.ctype in ConstructType.node:
        return NodeRealizer(csym)
    elif csym.ctype in ConstructType.flow:
        return FlowRealizer(csym)
    elif csym.ctype == ConstructType.response:
        return ResponseRealizer(csym)
    elif csym.ctype == ConstructType.behavior:
        return BehaviorRealizer(csym)
    elif csym.ctype == ConstructType.buffer:
        return BufferRealizer(csym)
    elif csym.ctype == ConstructType.subsystem:
        return SubsystemRealizer(csym)
    elif csym.ctype == ConstructType.agent:
        return AgentRealizer(csym)
    else:
        raise ValueError("Unexpected construct type in {}".format(csym))


def make_subsystem(
    csym: ConstructSymbol, members: ConstructSymbolIterable
) -> SubsystemRealizer:
    """Initialize empty subsystem realizer with given members."""

    subsystem = SubsystemRealizer(csym)
    subsystem.insert_realizers(*(make_realizer(member) for member in members))
    return subsystem


def make_agent(
    csym: ConstructSymbol, 
    subsystems: Mapping[ConstructSymbol, ConstructSymbolIterable], 
    buffers: ConstructSymbolIterable
) -> AgentRealizer:
    """Initialize empty agent realizer with given subsystems and buffers."""

    agent = AgentRealizer(csym)
    agent.insert_realizers(
        *(
            make_subsystem(subsystem, members) 
            for subsystem, members in subsystems.items()
        )
    )
    agent.insert_realizers(*(make_realizer(buffer) for buffer in buffers))
    return agent
