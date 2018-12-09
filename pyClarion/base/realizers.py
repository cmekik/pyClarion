"""Tools for defining the behavior of constructs within simulations."""


# Notes for Readers:

#   - Type aliases and helper functions are defined first.
#   - There are two major types of construct realizer: basic construct 
#     realizers and container construct realizers. Definitions for each major 
#     realizer type are grouped together in marked sections.


__all__ = [
    "ConstructRealizer", "BasicConstructRealizer", "NodeRealizer", 
    "FlowRealizer", "AppraisalRealizer", "BehaviorRealizer", "BufferRealizer",
    "ContainerConstructRealizer", "SubsystemRealizer", "AgentRealizer"
]


from typing import (
    Any, Callable, Iterable, MutableMapping, Union, Optional, ClassVar, Text, 
    Type, Dict, Tuple, Iterator, Hashable, List, cast
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
    'NodeRealizer', 'FlowRealizer', 'AppraisalRealizer', 'BehaviorRealizer', 
    'SubsystemRealizer'
]
HasOutput = Union[
    'NodeRealizer', 'FlowRealizer', 'AppraisalRealizer', 'BufferRealizer', 
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

    def _check_csym(self, csym: ConstructSymbol) -> None:
        """Check if construct symbol matches realizer."""

        if csym.ctype not in type(self).ctype:
            raise ValueError(
                " ".join(
                    [   
                        type(self).__name__,
                        "expects construct symbol with ctype",
                        repr(type(self).ctype),
                        "but received symbol of ctype {}.".format(
                            repr(csym.ctype)
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

    def __init__(self, csym: ConstructSymbol, junction: Junction) -> None:

        super().__init__(csym)
        self.junction = junction

    def propagate(self) -> None:
        """Output current strength of node."""

        smap = self.junction(self.input.pull())
        packet = type(self).make_packet(self.csym, smap)
        self.output.update(packet)


class FlowRealizer(BasicConstructRealizer):

    ctype = ConstructType.Flow

    def __init__(
        self, csym: ConstructSymbol, junction: Junction, channel: Channel
    ) -> None:

        super().__init__(csym)
        self.junction = junction
        self.channel = channel

    def propagate(self) -> None:
        """Compute new node activations."""

        combined = self.junction(self.input.pull())
        strengths = self.channel(combined)
        packet = type(self).make_packet(self.csym, strengths)
        self.output.update(packet)


class AppraisalRealizer(BasicConstructRealizer):

    ctype = ConstructType.Response

    def __init__(
        self, csym: ConstructSymbol, junction: Junction, selector: Selector
    ) -> None:

        super().__init__(csym)
        self.junction = junction
        self.selector = selector

    def propagate(self) -> None:
        """Make and output a decision."""

        combined = self.junction(self.input.pull())
        appraisal_data = self.selector(combined)
        decision_packet = type(self).make_packet(self.csym, appraisal_data)
        self.output.update(decision_packet)


class BehaviorRealizer(BasicConstructRealizer):
    
    ctype = ConstructType.Behavior
    has_output = False

    def __init__(self, csym: ConstructSymbol, effector: Effector) -> None:

        super().__init__(csym)
        self.effector = effector

    def propagate(self) -> None:
        """Execute selected callbacks."""

        self.effector(*self.input.pull())


class BufferRealizer(BasicConstructRealizer):

    ctype = ConstructType.Buffer
    has_input = False

    def __init__(self, csym: ConstructSymbol, source: Source) -> None:

        super().__init__(csym)
        self.source = source

    def propagate(self) -> None:
        """Output stored activation pattern."""

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

    def __init__(
        self, csym: ConstructSymbol, propagation_rule: PropagationRule, 
    ) -> None:
        """
        Initialize a new subsystem realizer.
        
        :param construct: Client subsystem.
        :param propagation_rule: Function implementing desired activation 
            propagation sequence. Should expect a single SubsystemRealizer as 
            argument. 
        """

        super().__init__(csym)
        self.propagation_rule = propagation_rule
        self.input = type(self).itype(self._watch, self._drop)

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
        
        possibilities = [
            (
                target.ctype is ConstructType.Response and
                source.ctype & cast(ResponseID, target.cid).itype
            ),
            (
                target.ctype is ConstructType.Behavior and
                source is cast(BehaviorID, target.cid).response
            ),
            (
                source.ctype is ConstructType.Microfeature and
                target.ctype is ConstructType.Flow and
                cast(FlowID, target.cid).ftype & FlowType.BB | FlowType.BT
            ),
            (
                source.ctype is ConstructType.Chunk and
                target.ctype is ConstructType.Flow and
                cast(FlowID, target.cid).ftype & FlowType.TT | FlowType.TB
            ),
            (
                source.ctype is ConstructType.Flow and
                target.ctype is ConstructType.Microfeature and
                cast(FlowID, source.cid).ftype & FlowType.BB | FlowType.TB
            ),
            (
                source.ctype is ConstructType.Flow and
                target.ctype is ConstructType.Chunk and
                cast(FlowID, source.cid).ftype & FlowType.TT | FlowType.BT
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
