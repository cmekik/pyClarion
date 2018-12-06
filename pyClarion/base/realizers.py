"""Tools for defining the behavior of constructs within simulations."""


# Notes for Readers:

#   - Type aliases and helper functions are defined first.
#   - There are two major types of construct realizer: basic construct 
#     realizers and container construct realizers. Definitions for each major 
#     realizer type are grouped together in marked sections.


import typing as typ
import weakref
import pyClarion.base.symbols as sym
import pyClarion.base.packets as pkt


__all__ = [
    "ConstructRealizer",
    "BasicConstructRealizer",
    "NodeRealizer",
    "FlowRealizer",
    "AppraisalRealizer",
    "BehaviorRealizer",
    "BufferRealizer",
    "ContainerConstructRealizer",
    "SubsystemRealizer",
    "AgentRealizer"
]


####################
### Type Aliases ###
####################


Channel = typ.Callable[
    [pkt.ConstructSymbolMapping], pkt.ConstructSymbolMapping
]
Junction = typ.Callable[
    [typ.Iterable[pkt.ActivationPacket]], 
    pkt.ConstructSymbolMapping
]
Selector = typ.Callable[[pkt.ConstructSymbolMapping], pkt.DecisionPacket]
Effector = typ.Callable[[pkt.DecisionPacket], None]
Source = typ.Callable[[], pkt.ConstructSymbolMapping]

PullMethod = typ.Callable[
    [], typ.Union[pkt.ActivationPacket, pkt.DecisionPacket]
]
InputBase = typ.MutableMapping[sym.ConstructSymbol, PullMethod]
PacketMaker = typ.Callable[[sym.ConstructSymbol, typ.Any], typ.Any]
Packets = typ.Union[
    typ.Iterable[pkt.ActivationPacket],
    typ.Iterable[pkt.DecisionPacket]
]

PropagationRule = typ.Callable[['SubsystemRealizer'], None]
ConnectivityPredicate = typ.Callable[
    [sym.ConstructSymbol, sym.ConstructSymbol], bool
]


######################
### Helper Classes ###
######################


# The classes defined below support networking of basic construct realizers. 
# These classes allow basic construct realizers to listen for and emit 
# activation packets.


class InputMonitor(object):
    """Listens for basic construct realizer outputs."""

    def __init__(self) -> None:

        self._input_links: InputBase = {}

    def pull(self) -> Packets:
        """Pull activations from input constructs."""

        for view in self._input_links.values():
            v = view()
            if v is not None: yield v

    def watch(
        self, csym: sym.ConstructSymbol, pull_method: PullMethod
    ) -> None:
        """Connect given construct as input to client."""

        self._input_links[csym] = pull_method

    def drop(self, csym: sym.ConstructSymbol):
        """Disconnect given construct from client."""

        del self._input_links[csym]

    @property
    def inputs(self) -> typ.Iterable[sym.ConstructSymbol]:
        """Inputs to client construct."""

        return (csym for csym in self._input_links)


class OutputView(object):
    """Exposes outputs of basic construct realizers."""

    def update(self, packet: pkt.ActivationPacket) -> None:
        """Update reported output of client construct."""
        
        self._buffer = packet

    def view(self) -> typ.Optional[pkt.ActivationPacket]:
        """Emit current output of client construct."""
        
        try:
            return self._buffer
        except AttributeError:
            return None        


######################################
### Construct Realizer Definitions ###
######################################


class ConstructRealizer(object):
    """
    Base class for construct realizers.

    Construct realizers are responsible for implementing the behavior of client 
    constructs. 
    """

    ctype: typ.ClassVar[sym.ConstructType]
    
    def __init__(self, csym: sym.ConstructSymbol) -> None:
        """Initialize a new construct realizer.
        
        :param csym: Symbolic representation of client construct.
        """

        self._check_csym(csym)
        self.csym = csym

    def __repr__(self) -> typ.Text:

        return "{}({})".format(type(self).__name__, repr(self.csym))

    def _check_csym(self, csym: sym.ConstructSymbol) -> None:
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

    itype: typ.Type = InputMonitor
    otype: typ.Type = OutputView
    has_input: bool = True
    has_output: bool = True
    make_packet: PacketMaker = pkt.make_packet

    def __init__(self, csym: sym.ConstructSymbol) -> None:

        super().__init__(csym)

        # Initialize I/O
        if type(self).has_input:
            self.input = type(self).itype()
        if type(self).has_output:
            self.output = type(self).otype()


class NodeRealizer(BasicConstructRealizer):

    ctype = sym.ConstructType.Node

    def __init__(self, csym: sym.ConstructSymbol, junction: Junction) -> None:

        super().__init__(csym)
        self.junction = junction

    def propagate(self) -> None:
        """Output current strength of node."""

        smap = self.junction(self.input.pull())
        packet = type(self).make_packet(self.csym, smap)
        self.output.update(packet)


class FlowRealizer(BasicConstructRealizer):

    ctype = sym.ConstructType.Flow

    def __init__(
        self, csym: sym.ConstructSymbol, junction: Junction, channel: Channel
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

    ctype = sym.ConstructType.Appraisal

    def __init__(
        self, csym: sym.ConstructSymbol, junction: Junction, selector: Selector
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
    
    ctype = sym.ConstructType.Behavior
    has_output = False

    def __init__(self, csym: sym.ConstructSymbol, effector: Effector) -> None:

        super().__init__(csym)
        self.effector = effector

    def propagate(self) -> None:
        """Execute selected callbacks."""

        self.effector(*self.input.pull())


class BufferRealizer(BasicConstructRealizer):

    ctype = sym.ConstructType.Buffer
    has_input = False

    def __init__(self, csym: sym.ConstructSymbol, source: Source) -> None:

        super().__init__(csym)
        self.source = source

    def propagate(self) -> None:
        """Output stored activation pattern."""

        strengths = self.source()
        packet = type(self).make_packet(self.csym, strengths)
        self.output.update(packet)


### Container Construct Realizers ###


class ContainerConstructRealizer(
    typ.MutableMapping[sym.ConstructSymbol, ConstructRealizer], 
    ConstructRealizer
):
    """Base class for container construct realizers."""

    def __init__(self, csym: sym.ConstructSymbol) -> None:

        super().__init__(csym)
        self._dict: typ.Dict = dict()

    def __len__(self) -> int:

        return len(self._dict)

    def __contains__(self, obj: typ.Any) -> bool:

        return obj in self._dict

    def __iter__(self) -> typ.Iterator:

        return iter(self._dict)

    def __getitem__(self, key: typ.Any) -> typ.Any:

        return self._dict[key]

    def __setitem__(self, key: typ.Any, value: typ.Any) -> None:

        self._check_kv_pair(key, value)
        self._dict[key] = value

    def __delitem__(self, key: typ.Any) -> None:

        del self._dict[key]

    def may_contain(self, csym: sym.ConstructSymbol) -> bool:
        """Return true if container construct may contain csym."""
        
        return False

    def insert_realizers(self, *realizers: ConstructRealizer) -> None:
        """Add pre-initialized realizers to self."""

        for realizer in realizers:
            self[realizer.csym] = realizer

    def csym_iterable(
        self, ctype: sym.ConstructType
    ) -> typ.Iterable[sym.ConstructSymbol]:
        """Return an iterator over all members matching ctype."""

        return (csym for csym in self._dict if csym.ctype in ctype)


    def _check_kv_pair(self, key: typ.Any, value: typ.Any) -> None:

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


class SubsystemRealizer(ContainerConstructRealizer):
    """A network of node, flow, apprasial, and behavior realizers."""

    ctype = sym.ConstructType.Subsystem

    def __init__(
        self, 
        csym: sym.ConstructSymbol, 
        propagation_rule: PropagationRule, 
        may_connect: ConnectivityPredicate
    ) -> None:
        """
        Initialize a new subsystem realizer.
        
        :param construct: Client subsystem.
        :param propagation_rule: Function implementing desired activation 
            propagation sequence. Should expect a single SubsystemRealizer as 
            argument. 
        :param may_connect: Predicate determining whether a source construct 
            may send activation packets to a target construct. Used to 
            automatically connect new constructs to existing constructs. Must 
            be a callable accepting two arguments, where the first argument is 
            the source construct and the second argument is the target 
            construct.
        """

        super().__init__(csym)
        self.propagation_rule = propagation_rule
        self.may_connect = may_connect

    def __setitem__(self, key: typ.Any, value: typ.Any) -> None:
        """
        Add given construct and realizer pair to self.
        
        Automatically integrates realizer I/O into existing network. 
        """

        super().__setitem__(key, value)

        for csym, realizer in self._dict.items():
            if self.may_connect(csym, key):
                value.input.watch(csym, realizer.output.view)
            if self.may_connect(key, csym):
                realizer.input.watch(key, value.output.view)

    def __delitem__(self, key: typ.Any) -> None:
        """
        Remove given construct from self.
        
        Any links within self to/from deleted construct will be dropped. 
        External links will not be touched.
        """

        super().__delitem__(key)

        for csym, realizer in self._dict.items():
            if self.may_connect(key, csym):
                realizer.input.drop(key)

    def propagate(self):
        """Propagate activations among realizers owned by self."""

        self.propagation_rule(self)

    def execute(self) -> None:
        """Fire all selected actions."""

        for behavior in self.behaviors:
            self[behavior].propagate()

    def may_contain(self, csym: sym.ConstructSymbol) -> bool:
        """Return true if subsystem realizer may contain construct symbol."""

        return bool(
            csym.ctype & (
                sym.ConstructType.Node |
                sym.ConstructType.Flow |
                sym.ConstructType.Appraisal |
                sym.ConstructType.Behavior
            )            
        )

    @property
    def nodes(self) -> typ.Iterable[sym.ConstructSymbol]:
        
        return self.csym_iterable(sym.ConstructType.Node)

    @property
    def flows(self) -> typ.Iterable[sym.ConstructSymbol]:
        
        return self.csym_iterable(sym.ConstructType.Flow)

    @property
    def appraisals(self) -> typ.Iterable[sym.ConstructSymbol]:
        
        return self.csym_iterable(sym.ConstructType.Appraisal)

    @property
    def behaviors(self) -> typ.Iterable[sym.ConstructSymbol]:
        
        return self.csym_iterable(sym.ConstructType.Behavior)


class AgentRealizer(ContainerConstructRealizer):
    """Realizer for Agent constructs."""

    ctype = sym.ConstructType.Agent

    def __init__(self, csym: sym.ConstructSymbol) -> None:
        """
        Initialize a new agent realizer.
        
        :param construct: Client agent.
        """

        super().__init__(csym)
        self._updaters : typ.List[typ.Callable[[], None]] = []

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

    def attach(self, *updaters: typ.Callable[[], None]) -> None:
        """
        Add update managers to self.
        
        :param update_managers: Callables that manage updates to dynamic 
            knowledge components. Should take no arguments and return nothing.
        """

        for updater in updaters:
            self._updaters.append(updater)

    def may_contain(self, csym: sym.ConstructSymbol) -> bool:
        """Return true if agent realizer may contain csym."""

        return csym.ctype in (
            sym.ConstructType.Subsystem |
            sym.ConstructType.Buffer
        )

    @property
    def updaters(self) -> typ.Iterable[typ.Callable[[], None]]:
        
        return (updater for updater in self._updaters)

    @property
    def buffers(self) -> typ.Iterable[sym.ConstructSymbol]:

        return self.csym_iterable(sym.ConstructType.Buffer)

    @property
    def subsystems(self) -> typ.Iterable[sym.ConstructSymbol]:

        return self.csym_iterable(sym.ConstructType.Subsystem)
