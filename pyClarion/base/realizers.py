"""Tools for defining the behavior of constructs within simulations."""


# ATTN READERS:
#   Here is some general information that may help reading. Imports and type 
#   aliases are grouped at the head of the file, followed by an abstract base 
#   class for all construct realizers. Broadly, there are two types of construct 
#   realizer, reflecting the two major construct types: basic construct 
#   realizers and container construct realizers. Definitions for each major 
#   realizer type are grouped together in marked sections; within sections, 
#   helper classes (if they exist) are immediately prior to their use.


###############
### IMPORTS ###
###############


import abc
import typing as typ
import pyClarion.base.symbols as sym
import pyClarion.base.packets as pkt
import pyClarion.base.processors as proc


####################
### TYPE ALIASES ###
####################


Ct = typ.TypeVar('Ct', bound=sym.ConstructSymbol)
Bt = typ.TypeVar('Bt', bound=sym.BasicConstructSymbol)
Xt = typ.TypeVar('Xt', bound=sym.ContainerConstructSymbol)

PullMethod = typ.Callable[
    [typ.Optional[typ.Iterable[sym.Node]]], pkt.ActivationPacket
]

PropagationRule = typ.Callable[['SubsystemRealizer'], None]
ConnectivityPredicate = typ.Callable[[typ.Any, typ.Any], bool]


###################################################
### ABSTRACT BASE CLASS FOR CONSTRUCT REALIZERS ###
###################################################


class ConstructRealizer(typ.Generic[Ct], abc.ABC):
    """
    Abstract base class for construct realizers.

    Construct realizers are responsible for implementing the behavior of their 
    client constructs. As a rule of thumb, every construct can be expected to 
    have at least one realizer within a model.
    """
    
    def __init__(self, construct: Ct) -> None:
        """Initialize a new construct realizer.
        
        :param construct: Client construct of self.
        """

        self.construct = construct

    @abc.abstractmethod
    def propagate(self) -> None:
        """Execute input/output routine associated with client construct."""
        
        pass

    @staticmethod
    def check_construct(construct: sym.ConstructSymbol, type_: typ.Type):
        """Check if construct matches given type."""

        if not isinstance(construct, type_):
            raise TypeError("Unexpected construct type {}".format(str(construct)))


#################################
### BASIC CONSTRUCT REALIZERS ### 
#################################


# This section has three parts:
#    1. Some helper classes are defined for handling communication between 
#       various basic constructs. 
#    2. An abstract base class for basic construct realizers is defined.
#    3. A construct realizer is defined for every major type of basic construct: 
#       Node, Flow, Appraisal, Behavior, and Buffer


### HELPER CLASSES FOR BASIC CONSTRUCT REALIZERS ###


# The classes defined below support networking of basic construct realizers. 
# These classes allow basic construct realizers to listen for and emit 
# activation packets. These classes are used in the private initialization 
# method BasicConstructRealizer._init_io(), defined further below.


class InputMonitor(object):
    """Listens for inputs to `BasicConstructRealizer` objects."""

    def __init__(self) -> None:

        self.input_links: typ.Dict[typ.Hashable, PullMethod] = dict()

    def pull(
        self, keys: typ.Iterable[sym.Node] = None
    ) -> typ.Iterable[pkt.ActivationPacket]:
        
        return [
            pull_method(keys) for pull_method in self.input_links.values()
        ] 

    def watch(self, identifier: typ.Hashable, pull_method: PullMethod) -> None:

        self.input_links[identifier] = pull_method

    def drop(self, identifier: typ.Hashable):

        del self.input_links[identifier]


class OutputView(object):
    """Exposes outputs of `BasicConstructRealizer` objects."""

    def __init__(
        self, default_activation: pkt.DefaultActivation = None
    ) -> None:

        self.default_activation = default_activation

    def update(self, output: pkt.ActivationPacket) -> None:
        
        self._output_buffer = output

    def view(self, keys: typ.Iterable[sym.Node] = None) -> pkt.ActivationPacket:
        
        if keys:
            out = self.output_buffer.subpacket(keys, self.default_activation)
        else:
            out = self.output_buffer.copy()
        return out

    @property
    def output_buffer(self) -> pkt.ActivationPacket:

        if self._output_buffer is not None:
            return self._output_buffer
        else:
            raise AttributeError()


### BASE CLASS FOR BASIC CONSTRUCT REALIZERS ###
    
    
class BasicConstructRealizer(ConstructRealizer[Bt]):
    """
    Base class for basic construct realizers.

    Provides initialization routines common to all basic construct realizers.
    """

    def __init__(self, construct: Bt) -> None:

        super().__init__(construct)

    def _init_io(
        self, 
        has_input: bool = True, 
        has_output: bool = True, 
        default_activation: pkt.DefaultActivation = None
    ) -> None:

        if has_input:
            self.input = InputMonitor()
        if has_output:
            self.output = OutputView(default_activation)
            self.propagate()


### CONCRETE BASIC CONSTRUCT REALIZER DEFINITIONS ###


class NodeRealizer(BasicConstructRealizer[sym.Node]):
    """Realizer for Node constructs."""

    def __init__(
        self, 
        construct: sym.Node, 
        junction: proc.Junction, 
        default_activation: pkt.DefaultActivation
    ) -> None:
        """Initialize a new node realizer.

        :param construct: Client node.
        :param junction: Junction for combining or selecting output value 
            recommendations for client node.
        """
        
        self.check_construct(construct, sym.Node)
        super().__init__(construct)
        self.junction: proc.Junction = junction
        self._init_io(default_activation=default_activation)

    def propagate(self) -> None:
        """
        Compute current strength of client node.
        
        Combines strength recommendations of input sources using 
        ``self.junction``.
        """

        inputs = self.input.pull([self.construct])
        output = self.junction(*inputs)
        output.origin = self.construct
        self.output.update(output)


class FlowRealizer(BasicConstructRealizer[sym.Flow]):
    """Realizer for Flow constructs."""

    def __init__(
        self, 
        construct: sym.Flow, 
        junction: proc.Junction, 
        channel: proc.Channel, 
        default_activation: pkt.DefaultActivation
    ) -> None:
        """
        Initialize a new flow realizer.
        
        :param construct: Client flow.
        :param junction: Combines input packets.
        :param channel: Computes output.
        :param default_activation: Computes default outputs.
        """

        self.check_construct(construct, sym.Flow)
        super().__init__(construct)
        self.junction: proc.Junction = junction
        self.channel: proc.Channel = channel
        self._init_io(default_activation=default_activation)

    def propagate(self):
        """Construct strength recommendations for output nodes.

        Outputs are computed from strengths of input nodes using 
        ``self.channel``. Strengths of input nodes are first combined by 
        ``self.junction``.
        """

        inputs = self.input.pull()
        combined = self.junction(*inputs)
        output = self.channel(combined)
        output.origin = self.construct
        self.output.update(output)


class AppraisalRealizer(BasicConstructRealizer[sym.Appraisal]):
    """Realizer for Appraisal constructs."""

    def __init__(
        self, 
        construct: sym.Appraisal, 
        junction: proc.Junction, 
        selector: proc.Selector
    ) -> None:
        """
        Initialize a new appraisal realizer.
        
        :param construct: Client appraisal.
        :param junction: Combines incoming input packets.
        :param selector: Computes output.
        """

        self.check_construct(construct, sym.Appraisal)
        super().__init__(construct)
        self.junction: proc.Junction = junction
        self.selector: proc.Selector = selector
        self._init_io()

    def propagate(self):
        """
        Construct a decision packet from strengths of input nodes.
        
        Decision packets are constructed using ``self.selector``. Strengths 
        reported by input nodes are combined using ``self.junction``.
        """

        inputs = self.input.pull()
        combined = self.junction(*inputs)
        output = self.selector(combined)
        output.origin = self.construct
        self.output.update(output)


class BufferRealizer(BasicConstructRealizer[sym.Buffer]):
    """Realizer for Buffer constructs."""

    def __init__(
        self, 
        construct: sym.Buffer, 
        source: proc.Source, 
        default_activation: pkt.DefaultActivation
    ) -> None:
        """
        Initialize a new buffer realizer.
        
        :param construct: Client buffer.
        :param source: Computes output.
        :param default_activation: Computes default output.
        """

        self.check_construct(construct, sym.Buffer)
        super().__init__(construct)
        self.source: proc.Source = source
        self._init_io(has_input=False, default_activation=default_activation)

    def propagate(self):
        """
        Output activation pattern recommended by an external source.

        Activations are obtained from ``self.source``.
        """

        output = self.source()
        output.origin = self.construct
        self.output.update(output)


class BehaviorRealizer(BasicConstructRealizer[sym.Behavior]):
    """Realizer for a Behavior construct."""
    
    def __init__(
        self, construct: sym.Behavior, effector: proc.Effector
    ) -> None:
        """
        Initialize a new behavior realizer.
        
        :param construct: Client behavior.
        :param effector: Executes action callbacks.
        """
        
        self.check_construct(construct, sym.Behavior)
        super().__init__(construct)
        self.effector: proc.Effector = effector
        self._init_io(has_output=False)

    def propagate(self):
        """
        Execute callbacks based on input decision packet(s).

        Callback execution handled by ``self.effector``.
        """

        input_ = self.input.pull()
        self.effector(*input_)

    
#####################################
### CONTAINER CONSTRUCT REALIZERS ###
#####################################


class ContainerConstructRealizer(
    typ.MutableMapping[sym.ConstructSymbol, ConstructRealizer], 
    ConstructRealizer[Xt]
):
    """Generic construct realizer for container constructs."""

    def __init__(self, construct: Xt) -> None:

        super().__init__(construct)
        self.dict: typ.Dict = dict()

    def __len__(self) -> int:

        return len(self.dict)

    def __contains__(self, obj: typ.Any) -> bool:

        return obj in self.dict

    def __iter__(self) -> typ.Iterator:

        return self.dict.__iter__()

    def __getitem__(self, key: typ.Any) -> typ.Any:

        return self.dict[key]

    def __setitem__(self, key: typ.Any, value: typ.Any) -> None:

        if not self.may_contain(self.construct, key):
            raise TypeError("Unexpected type {}".format(type(key)))

        if key != value.construct:
            raise ValueError("Mismatch between key and realizer construct.")

        self.dict[key] = value

    def __delitem__(self, key: typ.Any) -> None:

        del self.dict[key]

    @staticmethod
    def may_contain(container: typ.Any, element: typ.Any) -> bool:
        """Return true if container construct may contain element."""
        
        possibilities = [
            (
                isinstance(container, sym.Subsystem) and
                (
                    isinstance(element, sym.Node) or
                    isinstance(element, sym.Flow) or
                    isinstance(element, sym.Appraisal) or
                    isinstance(element, sym.Behavior)
                )
            ),
            (
                isinstance(container, sym.Agent) and
                (
                    isinstance(element, sym.Subsystem) or
                    isinstance(element, sym.Buffer)
                )
            )
        ]
        return any(possibilities)



class SubsystemRealizer(ContainerConstructRealizer[sym.Subsystem]):
    """
    Realizer for Subsystem constructs.
    
    Contains a network of interconnected node, flow, apprasial, and 
    behavior realizers and controls their activation cycles.
    """

    def __init__(
        self, 
        construct: sym.Subsystem, 
        propagation_rule: PropagationRule, 
        may_connect: ConnectivityPredicate
    ) -> None:
        """
        Initialize a new subsystem realizer.
        
        :param construct: Client subsystem.
        :param propagation_rule: Function implementing desired activation 
            propagation sequence. Should expect a single SubsystemRealizer as 
            argument (will be passed self). Desired activation sequences may be 
            captured by a sequence of propagate calls to basic constructs owned 
            by self. 
        :param may_connect: Predicate determining whether a source construct may 
            send activation packets to a target construct. Used to automatically 
            connect new constructs to existing constructs. Must be a callable 
            accepting two arguments. The first argument is assumed to be the 
            source construct and the second argument to be the target construct.
        """

        self.check_construct(construct, sym.Subsystem)
        super().__init__(construct)
        self.propagation_rule = propagation_rule
        self.may_connect = may_connect

    def __setitem__(self, key: typ.Any, value: typ.Any) -> None:
        """
        Add given construct, realizer pair to self.
        
        New links will be established between the given construct and existing 
        constructs according to ``self.may_connect``. 
        """

        super().__setitem__(key, value)

        for construct, realizer in self.dict.items():
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

        for construct, realizer in self.dict.items():
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

    @property
    def nodes(self) -> typ.Iterable[sym.Node]:
        """Node constructs in self."""
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, sym.Node)
        }

    @property
    def flows(self) -> typ.Iterable[sym.Flow]:
        """Flow constructs in self."""
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, sym.Flow)
        }

    @property
    def appraisals(self) -> typ.Iterable[sym.Appraisal]:
        """Appraisal constructs in self."""
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, sym.Appraisal)
        }

    @property
    def behaviors(self) -> typ.Iterable[sym.Behavior]:
        """Behavior constructs in self."""
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, sym.Behavior)
        }


class UpdateManager(abc.ABC):
    """
    Abstract base class for management of updates to agent owned constructs.

    Supports the ``AgentRealizer`` class.

    Provides an interface for routines that monitor the activity of agent 
    constructs and that direct learning and forgetting. These routines may add, 
    remove or modify construct realizers owned by client agent.

    All client constructs should be passed to ``UpdateManager`` instances at 
    initialization time. This recommendation is not enforced.
    """

    @abc.abstractmethod
    def update(self) -> None:
        """
        Update client constructs.
        
        This method should trigger processes such as weight updates in neural 
        networks, creation/deletion of chunk nodes, adjustment of parameters, 
        and other routines associated with the maintenance and management of 
        simulated constructs throughout simulation time.
        """

        pass


class AgentRealizer(ContainerConstructRealizer):
    """Realizer for Agent constructs."""

    def __init__(self, construct: sym.Agent) -> None:
        """
        Initialize a new agent realizer.
        
        :param construct: Client agent.
        """

        self.check_construct(construct, sym.Agent)
        super().__init__(construct)
        self._update_managers : typ.List[UpdateManager] = []

    def propagate(self) -> None:
        """
        Propagate activations among realizers owned by self.
        
        First propagates activations from buffers, then propagates activations 
        within each constituent subsystem.
        """
        
        for construct, realizer in self.items():
            if isinstance(construct, sym.Buffer):
                realizer.propagate()
        for construct, realizer in self.items():
            if isinstance(construct, sym.Subsystem):
                realizer.propagate()

    def execute(self) -> None:
        """Execute all selected actions in all subsystems."""

        for construct, realizer in self.items():
            if isinstance(construct, sym.Subsystem):
                # Casting necessary below b/c, although it is expected, it is 
                # not guaranteed that: 
                #   isinstance(construct, Subsystem) iff 
                #   isinstance(realizer, SubsystemRealizer)
                # This is indirectly and partially enforced by may_contain() 
                # and self.__setitem__(). self.__setitem__() only accepts 
                # Subsystem or Buffer instances as keys, and the only provided 
                # realizer definition that accepts Subsystem instances as 
                # constructs is SubsystemRealizer. Unless a custom realizer 
                # class accepting Subsystem instances as constructs but not 
                # providing an execute() method is in use, the call to 
                # realizer.execute() should succeed.
                typ.cast(SubsystemRealizer, realizer).execute()

    def learn(self) -> None:
        """
        Update knowledge in all subsystems and all buffers.
        
        Issues update calls to each update manager in ``self.update_managers``.  
        """

        for update_manager in self.update_managers:
            update_manager.update()

    def attach(self, *update_managers: UpdateManager) -> None:
        """
        Add update managers to self.
        
        :param update_managers: Update managers for dynamic knowledge 
            components.
        """

        for update_manager in update_managers:
            self._update_managers.append(update_manager)

    @property
    def update_managers(self) -> typ.List[UpdateManager]:
        """Update managers attached to self."""
        
        return list(self._update_managers)
