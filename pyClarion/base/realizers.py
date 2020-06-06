"""Tools for defining the behavior of constructs within simulations."""


# Notes for Readers:

# There are two major types of construct realizer: basic construct realizers 
# and container construct realizers. Definitions for each major 
# realizer type are grouped together in marked sections.


from pyClarion.base.symbols import ConstructType, ConstructSymbol
from pyClarion.base.packets import (
    ActivationPacket, DecisionPacket, SubsystemPacket
)
from itertools import combinations, combinations_with_replacement
from collections import ChainMap, OrderedDict
from types import MappingProxyType
from typing import (
    TypeVar, Union, Container, Tuple, Dict, List, Callable, Hashable, Sequence, 
    Generic, Any, ClassVar, Optional, Type, Text, Iterator, Mapping
)
from abc import abstractmethod

__all__ = [
    "MatchArg", "UpdaterArg", "MissingSpec", "PullFuncs", "Inputs", "Updater",
    "Proc", "ConstructRealizer", "BasicConstruct", "ContainerConstruct", 
    "Node", "Flow", "Response", "Buffer", "Subsystem", "Agent"
]

It = TypeVar('It') # type variable for inputs to construct realizers
Ot = TypeVar('Ot') # type variable for outputs to construct realizers
Rt = TypeVar('Rt') # type variable representing a construct realizer 
MatchArg = Union[ConstructType, Container[ConstructSymbol]] # explain scope of this type variable
ConstructRef = Union[ConstructSymbol, Tuple[ConstructSymbol, ...]]
MissingSpec = Dict[ConstructRef, List[str]]
PullFuncs = Dict[ConstructSymbol, Callable[[], It]]
Inputs = Dict[ConstructSymbol, It]
Updater = Callable[[Rt], None]
# updater may be a pure ordered dict, or a list of identifier-updater pairs
UpdaterArg = Union[
    # This should be an OrderedDict, but in 3.6 generic ordered dicts are not 
    # supported (only python 3.7 and up).
    Dict[Hashable, Updater[Rt]], 
    Sequence[Tuple[Hashable, Updater[Rt]]]
]


#It was necessary to define It2, Ot2 to satisfy contravariance and covariance 
# requirements for Proc class.
It2 = TypeVar('It2', contravariant=True) # type variable for proc inputs
Ot2 = TypeVar('Ot2', covariant=True) # type variable for proc outputs
class Proc(Generic[It2, Ot2]):
    """
    Abstract class for BasicConstruct propagation callbacks.

    BasicConstruct propagation callbacks define how BasicConstruct instances 
    process inputs and set outputs upon a call to the `propagate()` method.

    This class is a generic class, taking two types It2, Ot2. These types are 
    fixed by the BasicConstruct owning the propagation callback.
    """

    def __call__(
        self, construct: ConstructSymbol, inputs: PullFuncs[It2], **kwargs: Any
    ) -> Ot2:
        """
        Execute basic construct propagation callback on call to self.

        Pulls data from input constructs and delegates processing to 
        self.call().
        """

        inputs_ = {source: pull_func() for source, pull_func in inputs.items()}
        return self.call(construct=construct, inputs=inputs_, **kwargs)

    @abstractmethod
    def call(
        self, construct: ConstructSymbol, inputs: Inputs[It2], **kwargs: Any
    ) -> Ot2:
        """
        Execute an activation propagation procedure (i.e., forward pass).

        In words, the signature says that propagation callbacks should expect 
        to receive a `construct` argument of type ConstructSymbol, an `inputs` 
        argument of type It2 and a keyword arguments dictionary of value type 
        Any and that they should have a return value of type Ot2.

        :param construct: The construct symbol associated with the realizer 
            owning to the propagation callback. 
        :param inputs: A datastructure specifying input passed to the owning 
            construct by other constructs within the simulation. 
        :param kwargs: Any additional parameters passed to the call to owning 
            construct's propagate() method through the `options` argument. These 
            arguments may be used to contextually modify proc behavior within an 
            activation cycle, pass in external inputs, override construct 
            behavior etc. For ease of debugging and as a precaution against 
            subtle bugs (e.g., failing to set an option due to misspelled 
            keyword argument name) it is recommended that callbacks implementing 
            the Proc protocol throw errors upon receipt of unexpected keyword 
            arguments.
        """
        pass
    

class ConstructRealizer(Generic[It, Ot]):
    """
    Base class for construct realizers.

    Construct realizers are responsible for implementing construct behavior.
    """

    ctype: ClassVar[ConstructType] = ConstructType.null_construct
    packet_constructor: ClassVar[Optional[Type[Ot]]] 

    def __init__(
        self, 
        name: Hashable, 
        matches: MatchArg = None,
        updaters: UpdaterArg[Any] = None
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param name: Identifier for construct, may be a ConstructSymbol, str, 
            tuple, or list.
        :param matches: Specification of constructs from which self may accept 
            input.
        """

        self.matches = matches
        self._construct = self._parse_name(name=name)
        self._inputs: Dict[ConstructSymbol, Callable[[], It]] = {}
        self._output: Optional[Ot] = None

        # This doesn't seem very safe...
        self.updaters: OrderedDict[Hashable, Updater[Any]]
        if updaters is None:
            self.updaters = OrderedDict()
        elif isinstance(updaters, OrderedDict):
            self.updaters = updaters
        else:
            self.updaters = OrderedDict(updaters)

    def __repr__(self) -> Text:

        return "<{}: {}>".format(self.__class__.__name__, str(self.construct))

    def propagate(self, args: Dict = None) -> None:
        """Propagate activations."""

        raise NotImplementedError()

    def learn(self) -> None:
        """Execute learning routines."""
        
        for updater in self.updaters.values():
            updater(self)

    def accepts(self, source: ConstructSymbol) -> bool:
        """Return true if self pulls information from source."""

        if self.matches is not None:
            if isinstance(self.matches, ConstructType):
                return source.ctype in self.matches
            else:
                return source in self.matches
        else:
            return False

    def watch(
        self, construct: ConstructSymbol, callback: Callable[[], It]
    ) -> None:
        """Set given construct as an input to self."""

        self._inputs[construct] = callback

    def drop(self, construct: ConstructSymbol) -> None:
        """Disconnect given construct from self."""

        if construct in self._inputs:
            del self._inputs[construct]

    def drop_all(self) -> None:
        """Disconnect self from all linked constructs."""

        self._inputs.clear()

    def view(self) -> Ot:
        """Return current output of self."""
        
        return self.output

    def update_output(self, output: Ot) -> None:
        """Update output of self."""

        self._output = output

    def clear_output(self) -> None:
        """Clear output."""

        self._output = None

    @property
    def construct(self) -> ConstructSymbol:
        """Client construct of self."""

        return self._construct

    @property
    def inputs(self) -> Dict[ConstructSymbol, Callable[[], It]]:
        """
        Mapping from input constructs to pull funcs.
        
        Warning: Direct in-place modification of this dict may result in 
        corrupted model behavior.
        """

        return self._inputs

    @property
    def output(self) -> Ot:
        """"Current output of self."""

        # Emit output if available.
        if self._output is not None:
            return self._output
        # Upon failure, try to construct empty output datastructure, 
        # if constructor is available.
        elif self.packet_constructor is not None:
            self._output = self.packet_constructor()
            return self._output
        # Upon a second failure, throw error.
        else:
            raise AttributeError('Output of {} not defined.'.format(repr(self)))

    @property
    def missing(self) -> MissingSpec:
        """
        Return any missing components of self.

        This attribute may be used to check if any important components were 
        forgotten at setup.
        """

        d: MissingSpec = {}
        if self.construct is None:
            d.setdefault(self.construct, []).append('construct')
        if self.matches is None:
            d.setdefault(self.construct, []).append('matches')
        return d

    def _check_construct(self, construct: ConstructSymbol) -> None:
        """Check if construct symbol matches realizer."""

        if construct.ctype not in type(self).ctype:
            raise ValueError(
                " ".join(
                    [   
                        type(self).__name__,
                        "expects construct symbol with ctype",
                        repr(type(self).ctype),
                        "but received symbol {} of ctype {}.".format(
                            str(construct), repr(construct.ctype)
                        )
                    ]
                )
            )

    def _parse_name(self, name: Hashable) -> ConstructSymbol:

        construct: ConstructSymbol

        # This implementation seems very wrong. Should simply check for 
        # hashability instead of complex type set, if input is not a construct 
        # symbol already. -CSM
        if isinstance(name, ConstructSymbol):
            self._check_construct(name)
            construct = name
        elif isinstance(name, str):
            construct = ConstructSymbol(type(self).ctype, name)
        elif isinstance(name, (tuple, list)):
            construct = ConstructSymbol(type(self).ctype, *name)
        else:
            raise TypeError(
                "Argument `name` to ConstructRealizer must be of type"
                "ConstructSymbol, str, tuple, or list."
            )
        
        return construct


############################################
### Basic Construct Realizer Definitions ###
############################################


class BasicConstruct(ConstructRealizer[It, Ot]):
    """Base class for basic construct realizers."""

    ctype: ClassVar[ConstructType] = ConstructType.basic_construct

    def __init__(
        self, 
        name: Hashable, 
        matches: MatchArg = None,
        proc: Callable = None,
        updaters: UpdaterArg[Any] = None,
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param construct: Symbolic representation of client construct.
        :param matches: Specification of constructs from which self may accept 
            input.
        :param proc: Activation processor associated with client construct.
        """

        super().__init__(name=name, matches=matches, updaters=updaters)
        self.proc = proc

    def propagate(self, args: Dict = None) -> None:
        """Update output of self with result of processor on current input."""

        if self.proc is not None:
            packet: Ot
            if args is not None:
                packet = self.proc(self.construct, self.inputs, **args)
            else:
                packet = self.proc(self.construct, self.inputs)
            self.update_output(packet)
        else:
            raise TypeError("'NoneType' object is not callable")

    @property
    def missing(self) -> MissingSpec:

        d = super().missing
        if self.proc is None:
            d.setdefault(self.construct, []).append('proc')
        return d


class Node(BasicConstruct[ActivationPacket, ActivationPacket]):

    ctype: ClassVar[ConstructType] = ConstructType.node
    packet_constructor: ClassVar[Type[ActivationPacket]] = ActivationPacket

    def __init__(
        self, 
        name: Hashable, 
        matches: MatchArg = None,
        proc: Proc[ActivationPacket, ActivationPacket] = None,
        updaters: UpdaterArg['Node'] = None,
    ) -> None:
        """Initialize a new node realizer."""

        super().__init__(
            name=name, matches=matches, proc=proc, updaters=updaters
        )

    @property
    def output_value(self) -> Any:
        
        return self.output[self.construct]


class Flow(BasicConstruct[ActivationPacket, ActivationPacket]):

    ctype: ClassVar[ConstructType] = ConstructType.flow
    packet_constructor: ClassVar[Type[ActivationPacket]] = ActivationPacket

    def __init__(
        self, 
        name: Hashable,
        matches: MatchArg = None,
        proc: Proc[ActivationPacket, ActivationPacket] = None,
        updaters: UpdaterArg['Flow'] = None,
    ) -> None:
        """Initialize a new flow realizer."""

        super().__init__(
            name=name, matches=matches, proc=proc, updaters=updaters
        )


class Response(BasicConstruct[ActivationPacket, DecisionPacket]):

    ctype: ClassVar[ConstructType] = ConstructType.response
    packet_constructor: ClassVar[Type[DecisionPacket]] = DecisionPacket

    def __init__(
        self,
        name: Hashable,
        matches: MatchArg = None,
        proc: Proc[ActivationPacket, DecisionPacket] = None,
        updaters: UpdaterArg['Response'] = None,
        effector: Callable[[DecisionPacket], None] = None
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param construct: Symbolic representation of client construct.
        :param matches: Specification of constructs from which self may accept 
            input.
        :param proc: Activation processor associated with client construct.
        :param effector: Routine for executing selected actions.
        """

        super().__init__(name=name, matches=matches, proc=proc, updaters=updaters)
        self.effector = effector

    def execute(self) -> None:
        """Execute any currently selected actions."""

        if self.effector is not None:
            self.effector(self.view())

    @property
    def missing(self) -> MissingSpec:

        d = super().missing
        if self.effector is None:
            d.setdefault(self.construct, []).append('effector')
        return d


class Buffer(BasicConstruct[SubsystemPacket, ActivationPacket]):

    ctype: ClassVar[ConstructType] = ConstructType.buffer
    packet_constructor: ClassVar[Type[ActivationPacket]] = ActivationPacket

    def __init__(
        self, 
        name: Hashable, 
        matches: MatchArg = None,
        proc: Proc[None, ActivationPacket] = None,
        updaters: UpdaterArg['Buffer'] = None,
    ) -> None:
        """Initialize a new buffer realizer."""

        super().__init__(name=name, matches=matches, proc=proc, updaters=updaters)


#####################################
### Container Construct Realizers ###
#####################################


class ContainerConstruct(ConstructRealizer[It, Ot]):
    """Base class for container construct realizers."""

    ctype: ClassVar[ConstructType] = ConstructType.container_construct

    def __init__(
        self, 
        name: Hashable, 
        matches: MatchArg = None,
        shared: Dict = None,
        updaters: UpdaterArg[Any] = None,
    ) -> None:
        """
        Initialize a new container realizer.
        """

        super().__init__(name=name, matches=matches, updaters=updaters)
        # shared is a simple dict for storing/housing resources shared among 
        # different components of the container including updaters and other 
        # realizers. There is no sophisticated semantics, just a convenient 
        # handle for storing handles to datastructures such as chunk databases 
        # and bla information. It is the user's responsibility to make sure 
        # shared resources are shared and used as intended.
        self.shared = shared if shared is not None else dict()

    def __contains__(self, key: ConstructSymbol) -> bool:

        try:
            self.__getitem__(key)
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[ConstructSymbol]:

        raise NotImplementedError()

    def __getitem__(self, key: ConstructSymbol) -> Any:

        raise NotImplementedError()

    def __delitem__(self, key: ConstructSymbol) -> None:

        raise NotImplementedError()

    def learn(self):
        """
        Execute learning routines in self and all members.
        
        Issues update calls to each updater attached to self.  
        """

        super().learn()
        for realizer in self.values():
            realizer.learn()

    def execute(self) -> None:
        """Execute currently selected actions."""

        raise NotImplementedError()

    def add(self, *realizers: ConstructRealizer) -> None:
        """Add a set of realizers to self."""

        raise NotImplementedError()

    def remove(self, *constructs: ConstructSymbol) -> None:
        """Remove a set of constructs from self."""

        for construct in constructs:
            self.__delitem__(construct)

    def clear(self):
        """Remove all constructs in self."""

        # make a copy of self.keys() first so as not to modify self during 
        # iteration over self.
        keys = tuple(self.keys())
        for construct in keys:
            del self[construct]

    def keys(self) -> Iterator[ConstructSymbol]:
        """Return iterator over all construct symbols in self."""

        for construct in self:
            yield construct

    def values(self) -> Iterator[ConstructRealizer]:
        """Return iterator over all construct realizers in self."""

        for construct in self:
            yield self[construct]

    def items(self) -> Iterator[Tuple[ConstructSymbol, ConstructRealizer]]:
        """Return iterator over all symbol, realizer pairs in self."""

        for construct in self:
            yield construct, self[construct]

    def link(self, source: ConstructSymbol, target: ConstructSymbol) -> None:
        """Link source construct to target construct."""

        self[target].watch(source, self[source].view)

    def unlink(self, source: ConstructSymbol, target: ConstructSymbol) -> None:
        """Unlink source construct from target construct."""

        self[target].drop(source)

    def watch(
        self, construct: ConstructSymbol, callback: Callable[[], It]
    ) -> None:
        """
        Add construct as an input to self. 
        
        Also adds construct as input to any interested construct in self.
        """

        super().watch(construct, callback)
        for realizer in self.values():
            if realizer.accepts(construct):
                realizer.watch(construct, callback)

    def drop(self, construct: ConstructSymbol) -> None:
        """
        Remove construct as an input to self. 
        
        Also removes construct as an input from any listening member in self.
        """

        super().drop(construct)
        for realizer in self.values():
            if realizer.accepts(construct):
                realizer.drop(construct)

    def drop_all(self) -> None:
        """
        Remove all inputs to self. 
        
        Also removes all inputs to self from any constructs in self that may be 
        listening to them.
        """

        for construct in self._inputs:
            for realizer in self.values():
                if realizer.accepts(construct):
                    realizer.drop(construct)
        super().drop_all()           

    def weave(self) -> None:
        """
        Add any acceptable links among constructs in self.
        
        A link is considered acceptable by a member construct if 
        member.accepts() returns True.

        Will also add links from inputs to self to any accepting member 
        construct.
        """

        # self-links
        for realizer in self.values(): 
            if realizer.accepts(realizer.construct):
                realizer.watch(realizer.construct, realizer.view)
        # pairwise links
        for realizer1, realizer2 in combinations(self.values(), 2):
            if realizer1.accepts(realizer2.construct):
                realizer1.watch(realizer2.construct, realizer2.view)
            if realizer2.accepts(realizer1.construct):
                realizer2.watch(realizer1.construct, realizer1.view)
        # links to subsystem input buffers
        for construct, callback in self._inputs.items():
            for realizer in self.values():
                if realizer.accepts(construct):
                    realizer.watch(construct, callback)

    def unweave(self) -> None:
        """
        Remove all links to and among constructs in self.
        
        Will also remove any links from inputs to self to member constructs.
        """

        for realizer in self.values():
            realizer.drop_all()

    def reweave(self) -> None:
        """Bring links among constructs in compliance with current specs."""

        self.unweave()
        self.weave()

    def clear_output(self) -> None:
        """Clear output of self and all members."""

        self._output = None
        for realizer in self.values():
            realizer.clear_output()

    @property
    def missing(self) -> MissingSpec:
        """Return missing components in self or in member constructs."""

        d = super().missing
        for realizer in self.values():
            d_realizer = realizer.missing
            for k, v in d_realizer.items():
                new_k: Tuple[ConstructSymbol, ...]
                if isinstance(k, ConstructSymbol):
                    new_k = (self.construct, k)
                else:
                    new_k = (self.construct, *k)
                d[new_k] = v
        return d

    def _update_links(self, new_realizer: ConstructRealizer) -> None:
        """Add any acceptable links associated with a new realizer."""

        for construct, realizer in self.items():
            if realizer.accepts(new_realizer.construct):
                realizer.watch(new_realizer.construct, new_realizer.view)
            if new_realizer.accepts(construct):
                new_realizer.watch(construct, realizer.view)
        for construct, callback in self._inputs.items():
            if new_realizer.accepts(construct):
                new_realizer.watch(construct, callback)

    def _drop_links(self, construct: ConstructSymbol) -> None:
        """Remove construct from inputs of any accepting member constructs."""

        for realizer in self.values():
            if realizer.accepts(construct):
                realizer.drop(construct)


class Subsystem(ContainerConstruct[ActivationPacket, SubsystemPacket]):

    ctype: ClassVar[ConstructType] = ConstructType.subsystem
    packet_constructor: ClassVar[Type[SubsystemPacket]] = SubsystemPacket

    def __init__(
        self, 
        name: Hashable, 
        matches: MatchArg = None,
        proc: Callable[['Subsystem', Optional[Dict]], None] = None,
        shared: Dict = None,
        updaters: UpdaterArg['Subsystem'] = None
    ) -> None:

        super().__init__(
            name=name, matches=matches, shared=shared, updaters=updaters
        )
        self.proc = proc
        self._features: Dict[ConstructSymbol, Node] = {}
        self._chunks: Dict[ConstructSymbol, Node] = {}
        self._nodes: Mapping[ConstructSymbol, Node] = ChainMap(
            self._features, self._chunks
        )
        self._flows: Dict[ConstructSymbol, Flow] = {}
        self._responses: Dict[ConstructSymbol, Response] = {}


    def __iter__(self) -> Iterator[ConstructSymbol]:

        for construct in self._responses:
            yield construct
        for construct in self._flows:
            yield construct
        for construct in self._chunks:
            yield construct
        for construct in self._features:
            yield construct

    def __getitem__(self, key: ConstructSymbol) -> Any:

        if key.ctype in ConstructType.feature:
            return self._features[key]
        elif key.ctype in ConstructType.chunk:
            return self._chunks[key]
        elif key.ctype in ConstructType.flow:
            return self._flows[key]
        elif key.ctype in ConstructType.response:
            return self._responses[key]
        else:
            raise ValueError(
                "{} does not contain constructs of type {}".format(
                    self.__class__.__name__, repr(key.ctype)
                )
            ) 

    def __delitem__(self, key: ConstructSymbol) -> None:

        if key.ctype in ConstructType.feature:
            del self._features[key]
        elif key.ctype in ConstructType.chunk:
            del self._chunks[key]
        elif key.ctype in ConstructType.flow:
            del self._flows[key]
        elif key.ctype in ConstructType.response:
            del self._responses[key]
        else:
            raise ValueError(
                "{} does not contain constructs of type {}".format(
                    self.__class__.__name__, repr(key.ctype)
                )
            )

    def add(self, *realizers: ConstructRealizer) -> None:

        for i, realizer in enumerate(realizers):
            # Link new realizer with existing realizers
            self._update_links(realizer)
            # Store new realizer
            if isinstance(realizer, Node):
                if realizer.construct.ctype in ConstructType.feature:
                    self._features[realizer.construct] = realizer
                elif realizer.construct.ctype in ConstructType.chunk:
                    self._chunks[realizer.construct] = realizer
            elif isinstance(realizer, Flow):
                self._flows[realizer.construct] = realizer
            elif isinstance(realizer, Response):
                self._responses[realizer.construct] = realizer
            else:
                # Unacceptable realizer type passed to self
                # Restore self to state prior to call to add() and
                # raise a TypeError
                self._drop_links(realizer.construct)
                for new_realizer in realizers[:i]:
                    del self[new_realizer.construct]
                raise TypeError(
                    "{} may not contain realizer of type {}".format(
                        self.__class__.__name__, realizer.__class__.__name__
                    )
                )

    def propagate(self, args: Dict = None) -> None:

        if self.proc is not None:
            self.proc(self, args)
        else:
            raise TypeError("'NoneType' object is not callable")
        
        packet = self._construct_subsystem_packet()
        self.update_output(packet)

    def execute(self) -> None:

        for realizer in self._responses.values():
            realizer.execute()

    def _construct_subsystem_packet(self) -> SubsystemPacket:

        strengths = {
            symb: node.output_value for symb, node in self.nodes.items()
        }
        decisions = {
            symb: node.output for symb, node in self.responses.items()
        }

        return SubsystemPacket(strengths=strengths, decisions=decisions)

    @property
    def missing(self) -> MissingSpec:
        """Return missing components of self and all members."""

        d = super().missing
        if self.proc is None:
            d.setdefault(self.construct, []).append('proc')
        return d

    @property
    def features(self) -> Mapping[ConstructSymbol, Node]:

        return MappingProxyType(self._features)

    @property
    def chunks(self) -> Mapping[ConstructSymbol, Node]:

        return MappingProxyType(self._chunks)

    @property
    def nodes(self) -> Mapping[ConstructSymbol, Node]:

        return MappingProxyType(self._nodes)

    @property
    def flows(self) -> Mapping[ConstructSymbol, Flow]:

        return MappingProxyType(self._flows)

    @property
    def responses(self) -> Mapping[ConstructSymbol, Response]:

        return MappingProxyType(self._responses)


class Agent(ContainerConstruct[None, None]):

    ctype: ClassVar[ConstructType] = ConstructType.agent
    packet_constructor: ClassVar[None] = None

    def __init__(
        self, 
        name: Hashable, 
        matches: MatchArg = None, 
        shared: Dict = None,
        updaters: UpdaterArg['Agent'] = None
    ) -> None:

        super().__init__(
            name=name, matches=matches, shared=shared, updaters=updaters
        )
        self._buffers: Dict[ConstructSymbol, Buffer] = {}
        self._subsystems: Dict[ConstructSymbol, Subsystem] = {}

    def __iter__(self) -> Iterator[ConstructSymbol]:

        for construct in self._buffers:
            yield construct
        for construct in self._subsystems:
            yield construct

    def __getitem__(self, key: ConstructSymbol) -> Any:

        if key.ctype in ConstructType.buffer:
            return self._buffers[key]
        elif key.ctype in ConstructType.subsystem:
            return self._subsystems[key]
        else:
            raise ValueError(
                "{} does not contain constructs of type {}".format(
                    self.__class__.__name__, repr(key.ctype)
                )
            )

    def __delitem__(self, key: ConstructSymbol) -> None:

        if key.ctype in ConstructType.buffer:
            del self._buffers[key]
        elif key.ctype in ConstructType.subsystem:
            del self._subsystems[key]
        else:
            raise ValueError(
                "{} does not contain constructs of type {}".format(
                    self.__class__.__name__, repr(key.ctype)
                )
            )

    def add(self, *realizers: ConstructRealizer) -> None:
        """Add a set of realizers to self or a member of self."""

        for i, realizer in enumerate(realizers):
            # Link new realizer with existing realizers
            self._update_links(realizer)
            # Store new realizer
            if isinstance(realizer, Buffer):
                self._buffers[realizer.construct] = realizer
            elif isinstance(realizer, Subsystem):
                self._subsystems[realizer.construct] = realizer
            else:
                # Unacceptable realizer type passed to self
                # Restore self to state prior to call to add() and
                # raise a TypeError
                self._drop_links(realizer.construct)
                for new_realizer in realizers[:i]:
                    del self[new_realizer.construct]
                raise TypeError(
                    "{} may not contain realizer of type {}".format(
                        self.__class__.__name__, realizer.__class__.__name__
                    )
                )

    def propagate(self, args: Dict = None) -> None:

        args = args or dict()
        realizer: ConstructRealizer
        for construct, realizer in self._buffers.items():
            realizer.propagate(args=args.get(construct))
        for construct, realizer in self._subsystems.items():
            realizer.propagate(args=args.get(construct))

    def execute(self) -> None:
        """Execute currently selected actions."""

        for subsys in self._subsystems.values():
            for resp in subsys.responses.values():
                resp.execute()

    def weave(self) -> None:

        super().weave()
        for realizer in self._subsystems.values():
            realizer.weave()

    def unweave(self) -> None:

        super().unweave()
        for realizer in self._subsystems.values():
            realizer.unweave()

    @property
    def buffers(self) -> Mapping[ConstructSymbol, Buffer]:

        return MappingProxyType(self._buffers)

    @property
    def subsystems(self) -> Mapping[ConstructSymbol, Subsystem]:

        return MappingProxyType(self._subsystems)
