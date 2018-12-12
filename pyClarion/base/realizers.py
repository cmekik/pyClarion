"""Tools for defining the behavior of constructs within simulations."""


# Notes for Readers:

#   - Type aliases and helper functions are defined first.
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
    Type, Dict, Tuple, Iterator, Hashable, List, Mapping, Sequence, cast
)
from pyClarion.base.symbols import (
    ConstructSymbol, ConstructType, FlowType, FlowID, ResponseID, BehaviorID, 
    BufferID
)
from pyClarion.base.packets import (
    ConstructSymbolMapping, Packet, ActivationPacket, DecisionPacket, 
    make_packet
)


# Types used by helper classes

PullMethod = Callable[[], Union[ActivationPacket, DecisionPacket]]
InputMapping = MutableMapping[ConstructSymbol, PullMethod]
PacketIterable = Union[Iterable[ActivationPacket], Iterable[DecisionPacket]]
WatchMethod = Callable[[ConstructSymbol, PullMethod], None]
DropMethod = Callable[[ConstructSymbol], None]

# Types used by ConstructRealizer instances

ComponentSpec = Sequence[str] 

# Types used by BasicConstructRealizer instances

Channel = Callable[[ConstructSymbolMapping], ConstructSymbolMapping]
Junction = Callable[[Iterable[ActivationPacket]], ConstructSymbolMapping]
Selector = Callable[[ConstructSymbolMapping], DecisionPacket]
Effector = Callable[[DecisionPacket], None]
Source = Callable[[], ConstructSymbolMapping]
PacketMaker = Callable[[ConstructSymbol, Any], Packet]

# Types used by ContainerConstructRealizer instances

MutableRealizerMapping = MutableMapping[ConstructSymbol, 'ConstructRealizer']
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
            if v is not None: yield v

    def watch(self, csym: ConstructSymbol, pull_method: PullMethod) -> None:
        """Connect given construct as input to client."""

        self.input_links[csym] = pull_method

    def drop(self, csym: ConstructSymbol):
        """Disconnect given construct from client."""

        del self.input_links[csym]


class OutputView(object):
    """Exposes outputs of basic construct realizers."""

    def update(self, packet: ActivationPacket) -> None:
        """Update reported output of client construct."""
        
        self._buffer = packet

    def view(self) -> Optional[ActivationPacket]:
        """Emit current output of client construct."""
        
        try:
            return self._buffer
        except AttributeError:
            return None        


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

    ctype: ClassVar[ConstructType]
    __slots__: ComponentSpec = ()

    def __init__(self, csym: ConstructSymbol) -> None:
        """Initialize a new construct realizer.
        
        :param csym: Symbolic representation of client construct.
        """

        self._check_csym(csym)
        self.csym = csym

    def __repr__(self) -> Text:

        return "{}({})".format(type(self).__name__, repr(self.csym))

    def propagate(self) -> None:

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

    itype: Type = InputMonitor
    otype: Type = OutputView
    has_input: bool = True
    has_output: bool = True
    make_packet: PacketMaker = make_packet

    def __init__(self, csym: ConstructSymbol) -> None:

        super().__init__(csym)

        # Initialize I/O
        if type(self).has_input:
            self.input = type(self).itype()
        if type(self).has_output:
            self.output = type(self).otype()


class NodeRealizer(BasicConstructRealizer):

    ctype = ConstructType.Node
    __slots__ = ('junction',)

    def __init__(
        self, csym: ConstructSymbol, junction: Junction = None
    ) -> None:

        super().__init__(csym)
        if junction is not None: self.junction = junction

    def propagate(self) -> None:
        """Output current strength of node."""

        smap = self.junction(self.input.pull())
        packet = type(self).make_packet(self.csym, smap)
        self.output.update(packet)


class FlowRealizer(BasicConstructRealizer):

    ctype = ConstructType.Flow
    __slots__ = ('junction', 'channel')

    def __init__(
        self, 
        csym: ConstructSymbol, 
        junction: Junction = None, 
        channel: Channel = None
    ) -> None:

        super().__init__(csym)
        if junction is not None: self.junction = junction
        if channel is not None: self.channel = channel

    def propagate(self) -> None:
        """Compute new node activations."""

        combined = self.junction(self.input.pull())
        strengths = self.channel(combined)
        packet = type(self).make_packet(self.csym, strengths)
        self.output.update(packet)


class ResponseRealizer(BasicConstructRealizer):

    ctype = ConstructType.Response
    __slots__ = ('junction', 'selector')

    def __init__(
        self, 
        csym: ConstructSymbol, 
        junction: Junction = None, 
        selector: Selector = None
    ) -> None:

        super().__init__(csym)
        if junction is not None: self.junction = junction
        if selector is not None: self.selector = selector

    def propagate(self) -> None:
        """Make and output a decision."""

        combined = self.junction(self.input.pull())
        appraisal_data = self.selector(combined)
        decision_packet = type(self).make_packet(self.csym, appraisal_data)
        self.output.update(decision_packet)


class BehaviorRealizer(BasicConstructRealizer):
    
    ctype = ConstructType.Behavior
    has_output = False
    __slots__ = ('effector',)

    def __init__(
        self, csym: ConstructSymbol, effector: Effector = None
    ) -> None:

        super().__init__(csym)
        if effector is not None: self.effector = effector

    def propagate(self) -> None:
        """Execute selected callbacks."""

        self.effector(*self.input.pull())


class BufferRealizer(BasicConstructRealizer):

    ctype = ConstructType.Buffer
    has_input = False
    __slots__ = ('source',)

    def __init__(self, csym: ConstructSymbol, source: Source = None) -> None:

        super().__init__(csym)
        if source is not None: self.source = source

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
        packet = type(self).make_packet(self.csym, strengths)
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

        return self._dict[key]

    def __setitem__(self, key: Any, value: Any) -> None:

        self._check_kv_pair(key, value)
        self._dict[key] = value
        self._connect(key, value)

    def __delitem__(self, key: Any) -> None:

        del self._dict[key]

        for csym, realizer in self.items():
            if self.may_connect(key, csym):
                cast(HasInput, realizer).input.drop(key)

    def ready(self) -> bool:
        "Return true iff all necessary components defined for self and members."

        return super().ready() and all(r.ready() for r in self.values())

    def missing_recursive(self):
        """Return missing components in self and all member realizers."""

        missing = (
            self.missing(), 
            {
                construct: (
                    realizer.missing_recursive() 
                    if isinstance(realizer, ContainerConstructRealizer) 
                    else realizer.missing()
                ) for construct, realizer in self.items()
            }
        )
        return missing

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

    def insert_realizers(self, *realizers: ConstructRealizer) -> None:
        """Add pre-initialized realizers to self."""

        for realizer in realizers:
            self[realizer.csym] = realizer

    def iter_ctype(self, ctype: ConstructType) -> ConstructSymbolIterable:
        """Return an iterator over all members matching ctype."""

        return (csym for csym in self if csym.ctype in ctype)

    def items_ctype(self, ctype: ConstructType) -> ContainerConstructItems:
        """Return an iterator over all csym-realizer pairs matching ctype."""

        return (
            (csym, realizer) for csym, realizer in self.items() 
            if csym.ctype in ctype
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

    def _connect(self, key: Any, value: Any) -> None:

        for csym, realizer in self.items():
            if self.may_connect(csym, key):
                value = cast(HasInput, value)
                realizer = cast(HasOutput, realizer)
                value.input.watch(csym, realizer.output.view)
            if self.may_connect(key, csym):
                value = cast(HasOutput, value)
                realizer = cast(HasInput, realizer)
                realizer.input.watch(key, value.output.view)


class SubsystemRealizer(ContainerConstructRealizer):
    """A network of node, flow, apprasial, and behavior realizers."""

    ctype = ConstructType.Subsystem
    itype = SubsystemInputMonitor
    __slots__ = ('propagation_rule',)

    def __init__(
        self, csym: ConstructSymbol, propagation_rule: PropagationRule = None, 
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
        if propagation_rule is not None: 
            self.propagation_rule = propagation_rule

    def __setitem__(self, key: Any, value: Any) -> None:
        """Add given construct and realizer pair to self."""

        super().__setitem__(key, value)

        if key.ctype in ConstructType.Node:
            for buffer, pull_method in self.input.input_links.items():
                value.input.watch(buffer, pull_method)

    def propagate(self) -> None:
        """Propagate activations among realizers owned by self."""

        self.propagation_rule(self)

    def execute(self) -> None:
        """Fire all selected actions."""

        for behavior in self.behaviors:
            self[behavior].propagate()

    def may_contain(self, csym: ConstructSymbol) -> bool:
        """Return true if subsystem realizer may contain construct symbol."""

        return bool(
            csym.ctype & (
                ConstructType.Node |
                ConstructType.Flow |
                ConstructType.Response |
                ConstructType.Behavior
            )            
        )

    def may_connect(
        self, source: ConstructSymbol, target: ConstructSymbol
    ) -> bool:
        """
        Return true iff source may connect to target within self.
        
        Connects chunk and microfeature nodes may connect to flows as inputs 
        or outputs according to flow direction. Connects of response and 
        behavior constructs according to contents of respective construct 
        symbols.
        """
        
        possibilities: List[bool] = [
            (
                target.ctype is ConstructType.Response and
                bool(source.ctype & cast(ResponseID, target.cid).itype)
            ),
            (
                target.ctype is ConstructType.Behavior and
                source == cast(BehaviorID, target.cid).response
            ),
            (
                source.ctype is ConstructType.Microfeature and
                target.ctype is ConstructType.Flow and
                bool(
                    cast(FlowID, target.cid).ftype & (FlowType.BB | FlowType.BT)
                )
            ),
            (
                source.ctype is ConstructType.Chunk and
                target.ctype is ConstructType.Flow and
                bool(
                    cast(FlowID, target.cid).ftype & (FlowType.TT | FlowType.TB)
                )
            ),
            (
                source.ctype is ConstructType.Flow and
                target.ctype is ConstructType.Microfeature and
                bool(
                    cast(FlowID, source.cid).ftype & (FlowType.BB | FlowType.TB)
                )
            ),
            (
                source.ctype is ConstructType.Flow and
                target.ctype is ConstructType.Chunk and
                bool(
                    cast(FlowID, source.cid).ftype & (FlowType.TT | FlowType.BT)
                )
            )
        ]
        return any(possibilities)

    def _watch(
        self, identifier: Hashable, pull_method: PullMethod
    ) -> None:
        """Informs members of a new input to self."""

        for _, realizer in self.items_ctype(ConstructType.Node):
            cast(BasicConstructRealizer, realizer).input.watch(
                identifier, pull_method
            )

    def _drop(self, identifier: Hashable) -> None:
        """Informs members of a dropped input to self."""

        for _, realizer in self.items_ctype(ConstructType.Node):
            cast(BasicConstructRealizer, realizer).input.drop(identifier)

    @property
    def nodes(self) -> ConstructSymbolIterable:
        
        return self.iter_ctype(ConstructType.Node)

    @property
    def flows(self) -> ConstructSymbolIterable:
        
        return self.iter_ctype(ConstructType.Flow)

    @property
    def responses(self) -> ConstructSymbolIterable:
        
        return self.iter_ctype(ConstructType.Response)

    @property
    def behaviors(self) -> ConstructSymbolIterable:
        
        return self.iter_ctype(ConstructType.Behavior)


class AgentRealizer(ContainerConstructRealizer):
    """Realizer for Agent constructs."""

    ctype = ConstructType.Agent

    def __init__(self, csym: ConstructSymbol) -> None:
        """
        Initialize a new agent realizer.
        
        :param construct: Client agent.
        """

        super().__init__(csym)
        self._updaters: UpdaterList = []

    def __getitem__(self, key):

        if isinstance(key, ConstructSymbol):
            return super().__getitem__(key)
        elif len(key) == 2:
            # if key is len 2, must be subsystem, since buffers are basic
            # constructs.
            subsystem, member = key
            return super().__getitem__(subsystem)[member]
        else:
            raise TypeError("Unexpected key {}".format(key))

    def propagate(self) -> None:
        """Propagate activations among realizers owned by self."""
        
        for buffer in self.buffers:
            self[buffer].propagate()
        for subsystem in self.subsystems:
            self[subsystem].propagate()

    def execute(self) -> None:
        """Execute all selected actions in all subsystems."""

        for subsystem in self.subsystems:
            self[subsystem].execute()

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

    def may_contain(self, csym: ConstructSymbol) -> bool:
        """Return true if agent realizer may contain csym."""

        return csym.ctype in ConstructType.Subsystem | ConstructType.Buffer

    def may_connect(
        self, source: ConstructSymbol, target: ConstructSymbol
    ) -> bool:
        """
        Return true iff source may connect to target within self.
        
        Connects buffers to subsystems according to specifications in buffer 
        construct symbols.
        """
        
        possibilities = [
            (
                source.ctype is ConstructType.Buffer and
                target.ctype is ConstructType.Subsystem and
                target in cast(BufferID, source.cid).outputs
            )
        ]

        return any(possibilities)

    @property
    def updaters(self) -> UpdaterIterable:
        
        return (updater for updater in self._updaters)

    @property
    def buffers(self) -> ConstructSymbolIterable:

        return self.iter_ctype(ConstructType.Buffer)

    @property
    def subsystems(self) -> ConstructSymbolIterable:

        return self.iter_ctype(ConstructType.Subsystem)


####################################
### Construct Realizer Factories ###
####################################


def make_realizer(csym: ConstructSymbol) -> ConstructRealizer:
    """Initialize empty construct realizer for given construct symbol."""

    if csym.ctype in ConstructType.Node:
        return NodeRealizer(csym)
    elif csym.ctype is ConstructType.Flow:
        return FlowRealizer(csym)
    elif csym.ctype is ConstructType.Response:
        return ResponseRealizer(csym)
    elif csym.ctype is ConstructType.Behavior:
        return BehaviorRealizer(csym)
    elif csym.ctype is ConstructType.Buffer:
        return BufferRealizer(csym)
    elif csym.ctype is ConstructType.Subsystem:
        return SubsystemRealizer(csym)
    elif csym.ctype is ConstructType.Agent:
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
