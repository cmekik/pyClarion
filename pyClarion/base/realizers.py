"""Provides tools for defining the behavior of constructs within simulations."""


__all__ = [
    "MatchArg", "UpdaterArg", "MissingSpec", "PullFuncs", "Inputs", "Updater",
    "ConstructRealizer", "BasicConstruct", "ContainerConstruct", "Node", "Flow", 
    "Response", "Buffer", "Subsystem", "Agent", "Assets"
]


from pyClarion.base.symbols import ConstructType, ConstructSymbol, MatchSpec
from pyClarion.base.packets import (
    ActivationPacket, DecisionPacket, SubsystemPacket
)
from pyClarion.base.propagators import (
    Propagator, PropagatorA, PropagatorD, PropagatorB
)
from itertools import combinations, combinations_with_replacement
from collections import ChainMap, OrderedDict
from types import MappingProxyType
from typing import (
    TypeVar, Union, Container, Tuple, Dict, List, Callable, Hashable, Sequence, 
    Generic, Any, ClassVar, Optional, Type, Text, Iterator, Mapping,
    cast, no_type_check
)
from abc import abstractmethod


It = TypeVar('It') # type variable for inputs to construct realizers
Ot = TypeVar('Ot') # type variable for outputs to construct realizers
Rt = TypeVar('Rt') # type variable representing a construct realizer 

Pt = TypeVar("Pt") # type variable for basic construct propagators
At_co = TypeVar("At_co", covariant=True) # type variable for container construct assets

# explain scope of this type variable
MatchArg = Union[ConstructType, Container[ConstructSymbol], MatchSpec] 
ConstructRef = Union[ConstructSymbol, Tuple[ConstructSymbol, ...]]
MissingSpec = Dict[ConstructRef, List[str]]
PullFuncs = Dict[ConstructSymbol, Callable[[], It]]
Inputs = Dict[ConstructSymbol, It]
# Could type annotations for updaters be improved? 
# Right now,  
# - Can
Updater = Callable[[Rt], None] 
# updater may be a pure ordered dict, or a list of identifier-updater pairs
UpdaterArg = Union[
    # This should be an OrderedDict, but in 3.6 generic ordered dicts are not 
    # supported (only python 3.7 and up).
    Dict[Hashable, Updater[Rt]], 
    Sequence[Tuple[Hashable, Updater[Rt]]]
]

# Shorthands
APkt = ActivationPacket
DPkt = DecisionPacket
SPkt = SubsystemPacket


class ConstructRealizer(Generic[It, Ot]):
    """
    Base class for construct realizers.

    Construct realizers are facilitate communication between constructs by 
    providing a standard interface for creating, inspecting, modifying and 
    propagating information across construct networks. 

    Message passing among constructs follows a pull-based architecture. A 
    realizer decides what constructs to pull information from through its 
    `matches` attribute, which may be set on initialization.
    """

    _CRt = TypeVar("_CRt", bound="ConstructRealizer")
    ctype: ClassVar[ConstructType] = ConstructType.null_construct
    packet_constructor: ClassVar[Optional[Type[Ot]]]

    def __init__(
        self: _CRt, 
        name: Hashable, 
        matches: MatchArg = None,
        updaters: UpdaterArg[_CRt] = None
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


class BasicConstruct(ConstructRealizer[It, Ot], Generic[It, Ot, Pt]):
    """Base class for basic construct realizers."""

    # _CRt is defined to ensure that the type of a construct's updater is 
    # compatible with the construct itself. Not sure if this implementation is 
    # correct; needs confirmation. - Can
    _CRt = TypeVar("_CRt", bound="BasicConstruct")
    ctype: ClassVar[ConstructType] = ConstructType.basic_construct

    def __init__(
        self: _CRt, 
        name: Hashable, 
        matches: MatchArg = None,
        propagator: Pt = None,
        updaters: UpdaterArg[_CRt] = None,
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param construct: Symbolic representation of client construct.
        :param matches: Specification of constructs from which self may accept 
            input.
        :param propagator: Activation processor associated with client 
            construct.
        """

        super().__init__(name=name, matches=matches, updaters=updaters)
        self.propagator = propagator

    def propagate(self, args: Dict = None) -> None:
        """Update output of self with result of processor on current input."""

        if self.propagator is not None:
            packet: Ot
            # I'm not sure I like the cast below. Propagator is generic, but 
            # it's unclear to me that the variables are dealt with correctly. 
            # This should be checked.
            # If they are dealt with correctly, It may be fine. Correctness 
            # could also be enforced through class variables storing the desired 
            # Propagator type. 
            # Ideally, there would be a bound on the type variable, but doing so 
            # results in the propagator being upcast. This is not ideal because 
            # then the typing information is lost downstream and cannot be 
            # exploited by linters. 
            # Maybe there is a better way, but this was the simplest and least 
            # complicated I could come up with. The documentation should make 
            # the expected Propagator type *very* explicit. - Can  
            propagator = cast(Propagator, self.propagator)
            if args is not None:
                packet = propagator(self.construct, self.inputs, **args)
            else:
                packet = propagator(self.construct, self.inputs)
            self.update_output(packet)
        else:
            raise TypeError("'NoneType' object is not callable")

    @property
    def missing(self) -> MissingSpec:

        d = super().missing
        if self.propagator is None:
            d.setdefault(self.construct, []).append('propagator')
        return d


class Node(BasicConstruct[ActivationPacket, ActivationPacket, Pt]):

    _CRt = TypeVar("_CRt", bound="Node")
    ctype: ClassVar[ConstructType] = ConstructType.node
    packet_constructor: ClassVar[Type[ActivationPacket]] = ActivationPacket

    def __init__(
        self: _CRt, 
        name: Hashable, 
        matches: MatchArg = None,
        propagator: Pt = None,
        updaters: UpdaterArg[_CRt] = None,
    ) -> None:
        """Initialize a new node realizer."""

        super().__init__(
            name=name, matches=matches, propagator=propagator, updaters=updaters
        )

    @property
    def output_value(self) -> Any:
        
        return self.output[self.construct]


class Flow(BasicConstruct[ActivationPacket, ActivationPacket, Pt]):

    _CRt = TypeVar("_CRt", bound="Flow")
    ctype: ClassVar[ConstructType] = ConstructType.flow
    packet_constructor: ClassVar[Type[ActivationPacket]] = ActivationPacket

    def __init__(
        self: _CRt, 
        name: Hashable,
        matches: MatchArg = None,
        propagator: Pt = None,
        updaters: UpdaterArg[_CRt] = None,
    ) -> None:
        """Initialize a new flow realizer."""

        super().__init__(
            name=name, matches=matches, propagator=propagator, updaters=updaters
        )


class Response(BasicConstruct[ActivationPacket, DecisionPacket, Pt]):

    _CRt = TypeVar("_CRt", bound="Response")
    ctype: ClassVar[ConstructType] = ConstructType.response
    packet_constructor: ClassVar[Type[DecisionPacket]] = DecisionPacket

    def __init__(
        self: _CRt,
        name: Hashable,
        matches: MatchArg = None,
        propagator: Pt = None,
        updaters: UpdaterArg[_CRt] = None,
        effector: Callable[[DecisionPacket], None] = None
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param construct: Symbolic representation of client construct.
        :param matches: Specification of constructs from which self may accept 
            input.
        :param propagator: Activation propagator associated with client 
            construct.
        :param effector: Routine for executing selected actions.
        """

        super().__init__(
            name=name, matches=matches, propagator=propagator, updaters=updaters
        )
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


class Buffer(BasicConstruct[SubsystemPacket, ActivationPacket, Pt]):

    _CRt = TypeVar("_CRt", bound="Buffer")
    ctype: ClassVar[ConstructType] = ConstructType.buffer
    packet_constructor: ClassVar[Type[ActivationPacket]] = ActivationPacket

    def __init__(
        self: _CRt, 
        name: Hashable, 
        matches: MatchArg = None,
        propagator: Pt = None,
        updaters: UpdaterArg[_CRt] = None,
    ) -> None:
        """Initialize a new buffer realizer."""

        super().__init__(
            name=name, matches=matches, propagator=propagator, updaters=updaters
        )


#####################################
### Container Construct Realizers ###
#####################################


# Decorator is meant to disable type_checking for the class (but not sub- or 
# superclasses). @no_type_check is not supported on mypy as of 2020-06-10.
# Disabling type checks is required here to prevent the typechecker from 
# complaining about dynamically set attributes. - Can
# @no_type_check
class Assets(object):
    """
    Provides a namespace for ContainerConstruct assets.
    
    The main purpose of `Assets` objects is to provide handles for various
    datastructures such as chunk databases, rule databases, bla information, 
    etc. In general, all resources shared among different components of a 
    container construct are considered assets. 
    
    It is the user's responsibility to make sure shared resources are shared 
    and used as intended. 
    """
    
    def __init__(self, **kwds: Any) -> None:
        """
        Initialize a new Assets instance.

        :param kwds: A sequence of named to assets. Each asset in 
            will be set to the attribute given by the corresponding name.
        """

        for k, v in kwds.items():
            self.__setattr__(k, v)


class ContainerConstruct(ConstructRealizer[It, Ot], Generic[It, Ot, At_co]):
    """Base class for container construct realizers."""

    _CRt = TypeVar("_CRt", bound="ContainerConstruct")
    ctype: ClassVar[ConstructType] = ConstructType.container_construct

    def __init__(
        self: _CRt, 
        name: Hashable, 
        matches: MatchArg = None,
        assets: At_co = None,
        updaters: UpdaterArg[_CRt] = None,
    ) -> None:
        """
        Initialize a new container realizer.
        """

        super().__init__(name=name, matches=matches, updaters=updaters)
        # In case assets argument is None self.assets is given type Any to 
        # prevent type checkers from complaining about missing attributes. This 
        # occurs b/c attributes of Assets objects are set dynamically.
        self.assets: At_co = assets if assets is not None else Assets()

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
        # Would this cause trouble for parallelization? - Can
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


class Subsystem(ContainerConstruct[ActivationPacket, SubsystemPacket, At_co]):

    _CRt = TypeVar("_CRt", bound="Subsystem")
    ctype: ClassVar[ConstructType] = ConstructType.subsystem
    packet_constructor: ClassVar[Type[SubsystemPacket]] = SubsystemPacket

    def __init__(
        self: _CRt, 
        name: Hashable, 
        matches: MatchArg = None,
        propagator: Callable[[_CRt, Optional[Dict]], None] = None,
        assets: At_co = None,
        updaters: UpdaterArg[_CRt] = None
    ) -> None:

        super().__init__(
            name=name, matches=matches, assets=assets, updaters=updaters
        )
        self.propagator = propagator
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

    def propagate(self: _CRt, args: Dict = None) -> None:

        if self.propagator is not None:
            self.propagator(self, args)
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
        if self.propagator is None:
            d.setdefault(self.construct, []).append('propagator')
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


class Agent(ContainerConstruct[None, None, At_co]):

    _CRt = TypeVar("_CRt", bound="Agent")
    ctype: ClassVar[ConstructType] = ConstructType.agent
    packet_constructor: ClassVar[None] = None

    def __init__(
        self: _CRt, 
        name: Hashable, 
        matches: MatchArg = None, 
        assets: At_co = None,
        updaters: UpdaterArg[_CRt] = None
    ) -> None:

        super().__init__(
            name=name, matches=matches, assets=assets, updaters=updaters
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
