"""Tools for defining the behavior of constructs within simulations."""


# Notes for Readers:

# There are two major types of construct realizer: basic construct realizers 
# and container construct realizers. Definitions for each major 
# realizer type are grouped together in marked sections.


from pyClarion.base.symbols import *
from pyClarion.base.packets import *
from itertools import combinations
from types import MappingProxyType
from typing import (
    ClassVar, Any, Text, Union, Container, Callable, TypeVar, Generic, Dict,
    Optional, Hashable, List, Iterable, Sequence, MutableMapping, Iterator, 
    Mapping
)


It = TypeVar('It')
Ot = TypeVar('Ot')
MatchSpec = Union[ConstructType, Container[ConstructSymbol]]
ConstructRef = Union[ConstructSymbol, Tuple[ConstructSymbol, ...]]
MissingSpec = Dict[ConstructRef, List[str]]
ProcInput = Mapping[ConstructSymbol, Callable[[], It]]
Proc = Callable[[ConstructSymbol, ProcInput[It]], Ot]


class ConstructRealizer(Generic[It, Ot]):
    """
    Base class for construct realizers.

    Construct realizers are responsible for implementing construct behavior.
    """

    ctype: ClassVar[ConstructType] = ConstructType.null_construct

    def __init__(
        self, construct: ConstructSymbol, matches: MatchSpec = None
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param construct: Symbolic representation of client construct.
        :param matches: Specification of constructs from which self may accept 
            input.
        """

        self._check_construct(construct)
        self.construct = construct
        self.matches = matches
        self._inputs: Dict[ConstructSymbol, Callable[[], It]] = {}
        self._output: Optional[Ot] = None

    def __repr__(self) -> Text:

        return "<{}: {}>".format(self.__class__.__name__, str(self.construct))

    def propagate(self) -> None:
        """Propagate activations."""

        raise NotImplementedError()

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
    def inputs(self) -> Mapping[ConstructSymbol, Callable[[], It]]:
        """Mapping from input constructs to pull funcs."""

        return MappingProxyType(self._inputs)

    @property
    def output(self) -> Ot:
        """"Current output of self."""

        if self._output is not None:
            return self._output
        else:
            raise AttributeError('Output of {} not set.'.format(repr(self)))

    @property
    def missing(self) -> MissingSpec:
        """Return any missing components of self."""

        return {}

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


############################################
### Basic Construct Realizer Definitions ###
############################################


class BasicConstructRealizer(ConstructRealizer[It, Ot]):
    """Base class for basic construct realizers."""

    ctype: ClassVar[ConstructType] = ConstructType.basic_construct

    def __init__(
        self, 
        construct: ConstructSymbol, 
        matches: MatchSpec = None,
        proc: Proc[It, Ot] = None,
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param construct: Symbolic representation of client construct.
        :param matches: Specification of constructs from which self may accept 
            input.
        :param proc: Activation processor associated with client construct.
        """

        super().__init__(construct, matches)
        self.proc = proc

    def propagate(self) -> None:
        """Update output of self with result of processor on current input."""

        if self.proc is not None:
            packet: Ot = self.proc(self.construct, self.inputs)
        else:
            raise TypeError("'NoneType' object is not callable")
        self.update_output(packet)

    @property
    def missing(self) -> MissingSpec:

        d = super().missing
        if self.proc is None:
            d.setdefault(self.construct, []).append('proc')
        return d


class NodeRealizer(BasicConstructRealizer[ActivationPacket, ActivationPacket]):

    ctype: ClassVar[ConstructType] = ConstructType.node


class FlowRealizer(BasicConstructRealizer[ActivationPacket, ActivationPacket]):

    ctype: ClassVar[ConstructType] = ConstructType.flow


class ResponseRealizer(
    BasicConstructRealizer[ActivationPacket, DecisionPacket]
):

    ctype: ClassVar[ConstructType] = ConstructType.response

    def __init__(
        self,
        construct: ConstructSymbol,
        matches: MatchSpec = None,
        proc: Proc[ActivationPacket, DecisionPacket] = None,
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

        super().__init__(construct, matches, proc)
        self.effector = effector

    def execute(self) -> None:
        """Execute any currently selected actions."""

        if self.effector is not None:
            self.effector(self.view())
        else:
            raise TypeError("'NoneType' object is not callable")

    @property
    def missing(self) -> MissingSpec:

        d = super().missing
        if self.effector is None:
            d.setdefault(self.construct, []).append('effector')
        return d


class BufferRealizer(BasicConstructRealizer[None, ActivationPacket]):

    ctype: ClassVar[ConstructType] = ConstructType.buffer


class UpdaterRealizer(BasicConstructRealizer[None, None]):

    ctype: ClassVar[ConstructType] = ConstructType.updater


#####################################
### Container Construct Realizers ###
#####################################


class ContainerConstructRealizer(ConstructRealizer[It, None]):
    """Base class for container construct realizers."""

    ctype: ClassVar[ConstructType] = ConstructType.container_construct

    def __contains__(self, key: ConstructSymbol) -> bool:

        try:
            self.__getitem__(key)
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[ConstructSymbol]:

        raise NotImplementedError()

    def __getitem__(self, key: ConstructSymbol) -> ConstructRealizer:

        raise NotImplementedError()

    def __delitem__(self, key: ConstructSymbol) -> None:

        raise NotImplementedError()

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

        for realizer1, realizer2 in combinations(self.values(), 2):
            if realizer1.accepts(realizer2.construct):
                realizer1.watch(realizer2.construct, realizer2.view)
            if realizer2.accepts(realizer1.construct):
                realizer2.watch(realizer1.construct, realizer1.view)
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

        super().clear_output()
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


    def _drop_links(self, construct: ConstructSymbol) -> None:
        """Remove construct from inputs of any accepting member constructs."""

        for realizer in self.values():
            if realizer.accepts(construct):
                realizer.drop(construct)


class SubsystemRealizer(ContainerConstructRealizer[ActivationPacket]):

    ctype: ClassVar[ConstructType] = ConstructType.subsystem

    def __init__(
        self, 
        construct: ConstructSymbol, 
        matches: MatchSpec = None,
        proc: Callable[..., None] = None
    ) -> None:

        super().__init__(construct, matches)
        self.proc = proc
        self._nodes: Dict[ConstructSymbol, NodeRealizer] = {}
        self._flows: Dict[ConstructSymbol, FlowRealizer] = {}
        self._responses: Dict[ConstructSymbol, ResponseRealizer] = {}

    def __iter__(self):

        for construct in self._behaviors:
            yield construct
        for construct in self._responses:
            yield construct
        for construct in self._flows:
            yield construct
        for construct in self._nodes:
            yield construct

    def __getitem__(self, key: ConstructSymbol) -> ConstructRealizer:

        if key.ctype in ConstructType.node:
            return self._nodes[key]
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

        if key.ctype in ConstructType.node:
            del self._nodes[key]
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
            if isinstance(realizer, NodeRealizer):
                self._nodes[realizer.construct] = realizer
            elif isinstance(realizer, FlowRealizer):
                self._flows[realizer.construct] = realizer
            elif isinstance(realizer, ResponseRealizer):
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

    def propagate(self) -> None:

        if self.proc is not None:
            self.proc(self)
        else:
            raise TypeError("'NoneType' object is not callable")

    def execute(self) -> None:

        for realizer in self._responses.values():
            realizer.execute()

    @property
    def missing(self) -> MissingSpec:
        """Return missing components of self and all members."""

        d = super().missing
        if self.proc is None:
            d.setdefault(self.construct, []).append('proc')
        return d

    @property
    def nodes(self) -> Mapping[ConstructSymbol, NodeRealizer]:

        return MappingProxyType(self._nodes)

    @property
    def flows(self) -> Mapping[ConstructSymbol, FlowRealizer]:

        return MappingProxyType(self._flows)

    @property
    def responses(self) -> Mapping[ConstructSymbol, ResponseRealizer]:

        return MappingProxyType(self._responses)


class AgentRealizer(ContainerConstructRealizer[None]):

    ctype: ClassVar[ConstructType] = ConstructType.agent

    def __init__(
        self, construct: ConstructSymbol, matches: MatchSpec = None
    ) -> None:

        super().__init__(construct, matches)
        self._buffers: Dict[ConstructSymbol, BufferRealizer] = {}
        self._subsystems: Dict[ConstructSymbol, SubsystemRealizer] = {}
        self._updaters: Dict[ConstructSymbol, UpdaterRealizer] = {}

    def __iter__(self) -> Iterator[ConstructSymbol]:

        for construct in self._buffers:
            yield construct
        for construct in self._subsystems:
            yield construct
        for construct in self._updaters:
            yield construct

    def __getitem__(self, key: ConstructSymbol) -> ConstructRealizer:

        if key.ctype in ConstructType.buffer:
            return self._buffers[key]
        elif key.ctype in ConstructType.subsystem:
            return self._subsystems[key]
        elif key.ctype in ConstructType.updater:
            return self._updaters[key]
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
        elif key.ctype in ConstructType.updater:
            del self._updaters[key]
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
            if isinstance(realizer, BufferRealizer):
                self._buffers[realizer.construct] = realizer
            elif isinstance(realizer, SubsystemRealizer):
                self._subsystems[realizer.construct] = realizer
            elif isinstance(realizer, UpdaterRealizer):
                self._updaters[realizer.construct] = realizer
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

    def propagate(self) -> None:

        realizer: ConstructRealizer
        for realizer in self._buffers.values():
            realizer.propagate()
        for realizer in self._subsystems.values():
            realizer.propagate()

    def execute(self) -> None:
        """Execute currently selected actions."""

        for subsys in self._subsystems.values():
            for resp in subsys.responses.values():
                resp.execute()

    def learn(self) -> None:
        """
        Update knowledge in all subsystems and all buffers.
        
        Issues update calls to each updater attached to self.  
        """

        for realizer in self._updaters.values():
            realizer.propagate()

    def weave(self) -> None:

        super().weave()
        for realizer in self._subsystems.values():
            realizer.weave()

    def unweave(self) -> None:

        super().unweave()
        for realizer in self._subsystems.values():
            realizer.unweave()

    @property
    def buffers(self) -> Mapping[ConstructSymbol, BufferRealizer]:

        return MappingProxyType(self._buffers)

    @property
    def subsystems(self) -> Mapping[ConstructSymbol, SubsystemRealizer]:

        return MappingProxyType(self._subsystems)

    @property
    def updaters(self) -> Mapping[ConstructSymbol, UpdaterRealizer]:

        return MappingProxyType(self._updaters)
