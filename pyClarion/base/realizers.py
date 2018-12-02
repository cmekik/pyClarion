"""Tools for defining the behavior of theoretcally relevant constructs."""


from abc import ABC, abstractmethod
from typing import (
    TypeVar, Generic, MutableMapping, Dict, Any, Iterator, Iterable, List, 
    Callable, cast
)
from pyClarion.base.symbols import (
    ConstructSymbol, BasicConstructSymbol, ContainerConstructSymbol, 
    Node, Flow, Appraisal, Behavior, Buffer, Subsystem, Agent
)
from pyClarion.base.processors import (
    Channel, Junction, Selector, Effector, Source
)
from pyClarion.base.utils import may_contain, check_construct
from pyClarion.base.packets import DefaultActivation
from pyClarion.base.links import BasicInputMonitor, BasicOutputView


Ct = TypeVar('Ct', bound=ConstructSymbol)
Bt = TypeVar('Bt', bound=BasicConstructSymbol)
Xt = TypeVar('Xt', bound=ContainerConstructSymbol)


####################
### ABSTRACTIONS ###
####################


class ConstructRealizer(Generic[Ct], ABC):
    """
    A generic construct realizer.

    Construct realizers are responsible for implementing the behavior of their 
    client constructs. As a rule of thumb, every construct can be expected to 
    have at least one realizer within a model.
    """
    
    def __init__(self, construct: Ct) -> None:
        """Initialize a new construct realizer.
        
        :param construct: Client construct of self.
        """

        self.construct = construct

    @abstractmethod
    def propagate(self) -> None:
        """Execute input/output routine associated with client construct."""
        
        pass


class BasicConstructRealizer(ConstructRealizer[Bt]):
    """Generic construct realizer for basic constructs"""

    def __init__(self, construct: Bt) -> None:

        super().__init__(construct)

    def _init_io(
        self, 
        has_input: bool = True, 
        has_output: bool = True, 
        default_activation: DefaultActivation = None
    ) -> None:

        if has_input:
            self.input = BasicInputMonitor()
        if has_output:
            self.output = BasicOutputView(default_activation)
            self.propagate()


class ContainerConstructRealizer(
    MutableMapping[ConstructSymbol, ConstructRealizer], ConstructRealizer[Xt]
):
    """Generic construct realizer for container constructs."""

    def __init__(self, construct: Xt) -> None:

        super().__init__(construct)
        self.dict: Dict = dict()

    def __len__(self) -> int:

        return len(self.dict)

    def __contains__(self, obj: Any) -> bool:

        return obj in self.dict

    def __iter__(self) -> Iterator:

        return self.dict.__iter__()

    def __getitem__(self, key: Any) -> Any:

        return self.dict[key]

    def __setitem__(self, key: Any, value: Any) -> None:

        if not may_contain(self.construct, key):
            raise TypeError("Unexpected type {}".format(type(key)))

        if key != value.construct:
            raise ValueError("Mismatch between key and realizer construct.")

        self.dict[key] = value

    def __delitem__(self, key: Any) -> None:

        del self.dict[key]


#################################
### BASIC CONSTRUCT REALIZERS ### 
#################################


class NodeRealizer(BasicConstructRealizer[Node]):
    """Realizer for Node constructs."""

    def __init__(self, construct: Node, junction: Junction) -> None:
        """Initialize a new node realizer.

        :param construct: Client node.
        :param junction: Junction for combining or selecting output value 
            recommendations for client node.
        """
        
        check_construct(construct, Node)
        super().__init__(construct)
        self.junction: Junction = junction
        self._init_io()

    def propagate(self) -> None:
        """
        Compute current strength of client node.
        
        Combines strength recommendations of input sources using 
        ``self.junction``.
        """

        inputs = self.input.pull([self.construct])
        output = self.junction(*inputs)
        self.output.update(output)


class FlowRealizer(BasicConstructRealizer[Flow]):
    """Realizer for Flow constructs."""

    def __init__(
        self, 
        construct: Flow, 
        junction: Junction, 
        channel: Channel, 
        default_activation: DefaultActivation
    ) -> None:
        """
        Initialize a new flow realizer.
        
        :param construct: Client flow.
        :param junction: Combines input packets.
        :param channel: Computes output.
        :param default_activation: Computes default outputs.
        """

        check_construct(construct, Flow)
        super().__init__(construct)
        self.junction: Junction = junction
        self.channel: Channel = channel
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
        self.output.update(output)


class AppraisalRealizer(BasicConstructRealizer[Appraisal]):
    """Realizer for Appraisal constructs."""

    def __init__(
        self, construct: Appraisal, junction: Junction, selector: Selector
    ) -> None:
        """
        Initialize a new appraisal realizer.
        
        :param construct: Client appraisal.
        :param junction: Combines incoming input packets.
        :param selector: Computes output.
        """

        check_construct(construct, Appraisal)
        super().__init__(construct)
        self.junction: Junction = junction
        self.selector: Selector = selector
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
        self.output.update(output)


class BufferRealizer(BasicConstructRealizer[Buffer]):
    """Realizer for Buffer constructs."""

    def __init__(
        self, 
        construct: Buffer, 
        source: Source, 
        default_activation: DefaultActivation
    ) -> None:
        """
        Initialize a new buffer realizer.
        
        :param construct: Client buffer.
        :param source: Computes output.
        :param default_activation: Computes default output.
        """

        check_construct(construct, Buffer)
        super().__init__(construct)
        self.source: Source = source
        self._init_io(has_input=False, default_activation=default_activation)

    def propagate(self):
        """
        Output activation pattern recommended by an external source.

        Activations are obtained from ``self.source``.
        """

        output = self.source()
        self.output.update(output)


class BehaviorRealizer(BasicConstructRealizer[Behavior]):
    """Realizer for a Behavior construct."""
    
    def __init__(
        self, construct: Behavior, effector: Effector
    ) -> None:
        """
        Initialize a new behavior realizer.
        
        :param construct: Client behavior.
        :param effector: Executes action callbacks.
        """
        
        check_construct(construct, Behavior)
        super().__init__(construct)
        self.effector: Effector = effector
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


PropagationRule = Callable[['SubsystemRealizer'], None]
ConnectivityPredicate = Callable[[Any, Any], bool]


class SubsystemRealizer(ContainerConstructRealizer[Subsystem]):
    """
    Realizer for Subsystem constructs.
    
    Contains a network of interconnected node, flow, apprasial, and 
    behavior realizers and controls their activation cycles.
    """

    def __init__(
        self, 
        construct: Subsystem, 
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

        check_construct(construct, Subsystem)
        super().__init__(construct)
        self.propagation_rule = propagation_rule
        self.may_connect = may_connect

    def __setitem__(self, key: Any, value: Any) -> None:
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

    def __delitem__(self, key: Any) -> None:
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
    def nodes(self) -> Iterable[Node]:
        """Node constructs in self."""
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, Node)
        }

    @property
    def flows(self) -> Iterable[Flow]:
        """Flow constructs in self."""
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, Flow)
        }

    @property
    def appraisals(self) -> Iterable[Appraisal]:
        """Appraisal constructs in self."""
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, Appraisal)
        }

    @property
    def behaviors(self) -> Iterable[Behavior]:
        """Behavior constructs in self."""
        
        return {
            construct for construct in self.dict 
            if isinstance(construct, Behavior)
        }


class AgentRealizer(ContainerConstructRealizer):
    """Realizer for Agent constructs."""

    def __init__(self, construct: Agent) -> None:
        """
        Initialize a new agent realizer.
        
        :param construct: Client agent.
        """

        check_construct(construct, Agent)
        super().__init__(construct)
        self._update_managers : List[UpdateManager] = []

    def propagate(self) -> None:
        """
        Propagate activations among realizers owned by self.
        
        First propagates activations of buffers, then propagates activations 
        within each constituent subsystem.
        """
        
        for construct, realizer in self.items():
            if isinstance(construct, Buffer):
                realizer.propagate()
        for construct, realizer in self.items():
            if isinstance(construct, Subsystem):
                realizer.propagate()

    def execute(self) -> None:
        """Execute all selected actions in all subsystems."""

        for construct, realizer in self.items():
            if isinstance(construct, Subsystem):
                cast(SubsystemRealizer, realizer).execute()

    def learn(self) -> None:
        """
        Update knowledge in all subsystems and all buffers.
        
        Issues update calls to each update manager in ``self.update_managers``.  
        """

        for update_manager in self.update_managers:
            update_manager.update()

    def attach(self, *update_managers: 'UpdateManager') -> None:
        """
        Add update managers to self.
        
        :param update_managers: Update managers for dynamic knowledge 
            components.
        """

        for update_manager in update_managers:
            self._update_managers.append(update_manager)

    @property
    def update_managers(self) -> List['UpdateManager']:
        """Update managers attached to self."""
        
        return list(self._update_managers)


class UpdateManager(ABC):
    """
    Manages updates to constructs owned by an agent.

    Monitors subsystem and buffer activity and directs learning and forgetting 
    routines. May add, remove or modify construct realizers owned by client 
    agent.

    All client constructs should be passed to UpdateManager at initialization 
    time.
    """

    @abstractmethod
    def update(self) -> None:
        """
        Update client constructs.
        
        This method should trigger processes such as weight updates in neural 
        networks, creation/deletion of chunk nodes, adjustment of parameters, 
        and other routines associated with the maintenance and management of 
        simulated constructs throughout simulation time.
        """

        pass
