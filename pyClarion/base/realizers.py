"""
Tools for defining the behavior of constructs within simulations.

This module defines construct realizers for every major construct type. 
Construct realizer behavior is designed to be as uniform as possible, but two 
major types may be distinguished: basic construct realizers and container 
construct realizers. The former generally act as nodes within a network 
propagating and processing activation packets. The latter own subordinate 
construct realizers and control their behavior.
"""


# Notes for Readers:

#   - Type hints signal intended usage.
#   - Type aliases and helper functions are defined first.
#   - There are two major types of construct realizer, reflecting the two major 
#     construct types: basic construct realizers and container construct 
#     realizers. Definitions for each major realizer type are grouped together 
#     in marked sections.


import typing as typ
import enum
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


Channel = typ.Callable[[pkt.ActivationPacket], pkt.ActivationPacket]
Junction = typ.Callable[[typ.Iterable[pkt.ActivationPacket]], pkt.ActivationPacket]
Selector = typ.Callable[[pkt.ActivationPacket], pkt.DecisionPacket]
Effector = typ.Callable[[pkt.DecisionPacket], None]
Source = typ.Callable[[], pkt.ActivationPacket]

PullMethod = typ.Callable[[], pkt.ActivationPacket]
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

        self._input_links: typ.Dict[sym.ConstructSymbol, PullMethod] = dict()

    def pull(self) -> typ.Iterable[pkt.ActivationPacket]:
        """Pull activations from input constructs."""

        return (view() for view in self._input_links.values())

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
    constructs. As a rule of thumb, every simulated construct can be expected to 
    have at least one realizer within a model.
    """

    ctype: typ.ClassVar[sym.ConstructType]
    
    def __init__(self, csym: sym.ConstructSymbol) -> None:
        """Initialize a new construct realizer.
        
        :param csym: Construct symbol representing client construct.
        """

        self.check_csym(csym)
        self.csym = csym

    def __repr__(self) -> typ.Text:

        return "{}({})".format(type(self).__name__, repr(self.csym))

    def check_csym(self, csym: sym.ConstructSymbol) -> None:
        """Check if construct symbol matches realizer."""

        if csym.ctype & type(self).ctype:
            raise ValueError(
                " ".join(
                    [   
                        type(self).__name__,
                        " expects construct symbol with ctype ",
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

    def __init__(self, csym: sym.ConstructSymbol) -> None:

        super().__init__(csym)

        # Initialize I/O
        if type(self).has_input:
            self.input = type(self).itype()
        if type(self).has_output:
            self.output = type(self).otype()


class NodeRealizer(BasicConstructRealizer):
    """Realizer for node constructs."""

    ctype = sym.ConstructType.Node

    def __init__(self, csym: sym.ConstructSymbol, junction: Junction) -> None:

        super().__init__(csym)
        self.junction = junction

    def propagate(self) -> None:
        """Output current strength of node."""

        packet = self.junction(self.input.pull())
        packet.origin = self.csym
        self.output.update(packet)


class FlowRealizer(BasicConstructRealizer):
    """Realizer for flow constructs."""

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
        packet = self.channel(combined)
        packet.origin = self.csym
        self.output.update(packet)


class AppraisalRealizer(BasicConstructRealizer):
    """Realizer for appraisal constructs."""

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
        decision = self.selector(combined)
        decision.origin = self.csym
        self.output.update(decision)


class BehaviorRealizer(BasicConstructRealizer):
    """Realizer for behavior constructs."""
    
    ctype = sym.ConstructType.Behavior
    has_output = False

    def __init__(self, csym: sym.ConstructSymbol, effector: Effector) -> None:

        super().__init__(csym)
        self.effector = effector

    def propagate(self) -> None:
        """Execute selected callbacks."""

        self.effector(*self.input.pull())


class BufferRealizer(BasicConstructRealizer):
    """Realizer for buffer constructs."""

    ctype = sym.ConstructType.Buffer
    has_input = False

    def __init__(self, csym: sym.ConstructSymbol, source: Source) -> None:

        super().__init__(csym)
        self.source = source

    def propagate(self) -> None:
        """Output activation pattern in buffer."""

        packet = self.source()
        packet.origin = self.csym
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

        if not self.may_contain(key):
            raise ValueError(
                "{} may not contain {}; forbidden construct type.".format(
                    repr(self), repr(key)
                )
            )
            
        if key != value.construct:
            raise ValueError(
                "{} given key {} does not match value {}".format(
                    repr(self), repr(key), repr(value)
                )
            )

        self._dict[key] = value

    def __delitem__(self, key: typ.Any) -> None:

        del self._dict[key]

    def may_contain(self, csym: sym.ConstructSymbol) -> bool:
        """Return true if container construct may contain csym."""
        
        return False

    def _make_csym_iterable(
        self, ctype: sym.ConstructType
    ) -> typ.Iterable[sym.ConstructSymbol]:

        return (csym for csym in self._dict if csym.ctype is ctype)


class SubsystemRealizer(ContainerConstructRealizer):
    """
    Realizer for Subsystem constructs.
    
    Contains a network of interconnected node, flow, apprasial, and 
    behavior realizers and controls their activation cycles.
    """

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
        :param may_connect: Predicate determining whether a source construct may 
            send activation packets to a target construct. Used to automatically 
            connect new constructs to existing constructs. Must be a callable 
            accepting two arguments. The first argument is assumed to be the 
            source construct and the second argument to be the target construct.
        """

        super().__init__(csym)
        self.propagation_rule = propagation_rule
        self.may_connect = may_connect

    def __setitem__(self, key: typ.Any, value: typ.Any) -> None:
        """
        Add given construct, realizer pair to self.
        
        New links will be established between the given construct and existing 
        constructs according to ``self.may_connect``. 
        """

        super().__setitem__(key, value)

        for construct, realizer in self._dict.items():
            if self.may_connect(construct, key):
                value.input.watch(construct, realizer.output.view)
            if self.may_connect(key, construct):
                realizer.input.watch(key, value.output.view)

    def __delitem__(self, key: typ.Any) -> None:
        """
        Remove given construct from self.
        
        Any links to/from deleted construct will be dropped (uses 
        ``self.may_connect``).
        """

        super().__delitem__(key)

        for construct, realizer in self._dict.items():
            if self.may_connect(key, construct):
                realizer.input.drop(key)

    def propagate(self):
        """
        Propagate activations among realizers owned by self.
        
        Calls ``self.propgation_rule`` on self.
        """

        self.propagation_rule(self)

    def execute(self) -> None:
        """Fire all selected actions."""

        for behavior in self.behaviors:
            self.__getitem__(behavior).propagate()

    def may_contain(self, key: sym.ConstructSymbol) -> bool:
        """Return true if subsystem realizer may contain csym."""

        return bool(
            key.ctype & (
                sym.ConstructType.Node |
                sym.ConstructType.Flow |
                sym.ConstructType.Appraisal |
                sym.ConstructType.Behavior
            )            
        )

    @property
    def nodes(self) -> typ.Iterable[sym.ConstructSymbol]:
        """Subsystem nodes."""
        
        return self._make_csym_iterable(sym.ConstructType.Node)

    @property
    def flows(self) -> typ.Iterable[sym.ConstructSymbol]:
        """Subsystem flows."""
        
        return self._make_csym_iterable(sym.ConstructType.Flow)

    @property
    def appraisals(self) -> typ.Iterable[sym.ConstructSymbol]:
        """Subsystem appraisals."""
        
        return self._make_csym_iterable(sym.ConstructType.Appraisal)

    @property
    def behaviors(self) -> typ.Iterable[sym.ConstructSymbol]:
        """Subsystem behaviors."""
        
        return self._make_csym_iterable(sym.ConstructType.Behavior)


class AgentRealizer(ContainerConstructRealizer):
    """Realizer for Agent constructs."""

    def __init__(self, csym: sym.ConstructSymbol) -> None:
        """
        Initialize a new agent realizer.
        
        :param construct: Client agent.
        """

        super().__init__(csym)
        self._update_managers : typ.List[typ.Callable[[], None]] = []

    def propagate(self) -> None:
        """
        Propagate activations among realizers owned by self.
        
        First propagates activations from buffers, then propagates activations 
        within each constituent subsystem.
        """
        
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
        
        Issues update calls to each update manager in ``self.update_managers``.  
        """

        for update_manager in self.update_managers:
            update_manager()

    def attach(self, *update_managers: typ.Callable[[], None]) -> None:
        """
        Add update managers to self.
        
        :param update_managers: Callable objects that manage updates to dynamic 
            knowledge components. Should take no arguments and return nothing.
        """

        for update_manager in update_managers:
            self._update_managers.append(update_manager)

    def may_contain(self, key):
        """Return true if agent realizer may contain csym."""

        return bool( 
            key.ctype & (
                sym.ConstructType.Subsystem |
                sym.ConstructType.Buffer
            )
        )

    @property
    def update_managers(self) -> typ.Iterable[typ.Callable[[], None]]:
        """Update managers attached to self."""
        
        return (update_manager for update_manager in self._update_managers)

    @property
    def buffers(self) -> typ.Iterable[sym.ConstructSymbol]:
        """Agent buffers."""

        return self._make_csym_iterable(sym.ConstructType.Buffer)

    @property
    def subsystems(self) -> typ.Iterable[sym.ConstructSymbol]:
        """Agent subsystems."""

        return self._make_csym_iterable(sym.ConstructType.Subsystem)
