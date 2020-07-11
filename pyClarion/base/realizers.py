"""Provides tools for defining the behavior of simulated constructs."""


__all__ = [
    "MatchArg", "UpdaterArg", "MissingSpec", "PullFuncs", "Inputs", "Updater",
    "Realizer", "Construct", "Structure", "Assets"
]


from pyClarion.base.symbols import ConstructType, ConstructSymbol, MatchSpec
from pyClarion.base.packets import (
    ActivationPacket, ResponsePacket, SubsystemPacket,
)
from pyClarion.base.propagators import Propagator, Cycle
from itertools import combinations, combinations_with_replacement, chain
from collections import ChainMap, OrderedDict
from functools import lru_cache
from types import MappingProxyType, SimpleNamespace
from typing import (
    TypeVar, Union, Container, Tuple, Dict, List, Callable, Hashable, Sequence, 
    Generic, Any, ClassVar, Optional, Type, Text, Iterator, Mapping,
    cast, no_type_check
)
from abc import abstractmethod


It = TypeVar('It') # type variable for inputs to construct realizers
Ot = TypeVar('Ot') # type variable for outputs to construct realizers
Rt = TypeVar('Rt') # type variable representing a construct realizer 
At_co = TypeVar("At_co", covariant=True) # type variable for sturcture assets

# explain scope of this type variable
MatchArg = Union[ConstructType, Container[ConstructSymbol], MatchSpec] 
ConstructRef = Union[ConstructSymbol, Tuple[ConstructSymbol, ...]]
MissingSpec = Dict[ConstructRef, List[str]]
PullFuncs = Dict[ConstructSymbol, Callable[[], It]]
Inputs = Dict[ConstructSymbol, It]
# Could type annotations for updaters be improved? - Can
Updater = Callable[[Rt], None] 
# updater may be a pure ordered dict, or a list of identifier-updater pairs
UpdaterArg = Union[
    # This should be an OrderedDict, but in 3.6 generic ordered dicts are not 
    # supported (only python 3.7 and up).
    Dict[Hashable, Updater[Rt]], 
    Sequence[Tuple[Hashable, Updater[Rt]]]
]


class Realizer(Generic[It, Ot]):
    """
    Base class for construct realizers.

    Construct realizers are facilitate communication between constructs by 
    providing a standard interface for creating, inspecting, modifying and 
    propagating information across construct networks. 

    Message passing among constructs follows a pull-based architecture. A 
    realizer decides what constructs to pull information from through its 
    `matches` attribute, which may be set on initialization.
    """

    Self = TypeVar("Self", bound="Realizer")

    def __init__(
        self: Self, 
        name: Hashable, 
        updaters: UpdaterArg[Self] = None
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param name: Identifier for construct, may be a ConstructSymbol, str, 
            tuple, or list. If a construct symbol is given, its construct type 
            must match the construct type associated with the realizer class. 
            If a str, tuple, or list is given, a `ConstructSymbol` will be 
            created with the given values as construct identifiers and the 
            class `ctype` as its `ConstructType`.  
        :param matches: Specification of constructs from which self may accept 
            input. Expects a `ConstructType` or a container of construct 
            symbols. For complex matching patterns see `symbols.MatchSpec`.
        :param propagator: Activation processor associated with client 
            construct. Propagates strengths based on inputs from linked 
            constructs. It is expected that this argument will behave like a 
            `Propagator` object; this expectation is not enforced.
        :param updaters: A dict-like object containing procedures for updating 
            construct knowledge.
        """

        if not isinstance(name, ConstructSymbol):
            raise TypeError(
                "Agrument 'name' must be of type ConstructSymbol"
                "got {} instead.".format(type(name))
            )
        self._construct = name
        self._inputs: Dict[ConstructSymbol, Callable[[], It]] = {}
        self._output: Optional[Ot] = None

        # This doesn't seem very safe...
        self.updaters: OrderedDict[Hashable, Updater[Any]]
        if updaters is None: self.updaters = OrderedDict()
        elif isinstance(updaters, OrderedDict): self.updaters = updaters
        else: self.updaters = OrderedDict(updaters)

    def __repr__(self) -> Text:

        return "<{}: {}>".format(self.__class__.__name__, str(self.construct))

    def propagate(self, args: Dict = None) -> None:
        """
        Propagate activations.

        :param args: A dict containing optional arguments for self and 
            subordinate constructs (if any).
        """

        raise NotImplementedError()

    def learn(self) -> None:
        """Execute learning routines."""
        
        for updater in self.updaters.values():
            updater(self)

    def accepts(self, source: ConstructSymbol) -> bool:
        """Return true if self pulls information from source."""

        raise NotImplementedError()

    def watch(
        self, construct: ConstructSymbol, callback: Callable[[], It]
    ) -> None:
        """
        Set given construct as an input to self.
        
        :param construct: Symbol for target construct.
        :param callback: A callable that returns a `Packet` representing the 
            output of the target construct. Typically this will be the `view()` 
            method of a construct realizer.
        """

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

    @property # type: ignore
    @lru_cache(maxsize=1)
    def inputs(self) -> Mapping[ConstructSymbol, Callable[[], It]]:
        """Mapping from input constructs to pull funcs."""

        return MappingProxyType(self._inputs)

    @property
    def output(self) -> Ot:
        """"Current output of self."""

        # Emit output if available.
        if self._output is not None:
            return self._output
        # Upon failure, throw output error.
        else:
            cls = type(self)
            repr_ = repr(self)
            raise cls.OutputError('Output of {} not defined.'.format(repr_))

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
        return d

    class OutputError(Exception):
        """Raised when a realizer has no output"""
        pass


############################################
### Basic Construct Realizer Definitions ###
############################################


class Construct(Realizer[It, Ot]):
    """
    Base class for basic construct realizers.
    
    `BasicConstruct` objects are leaves in the construct realizer containment 
    hierarchy. That is to say they contain no other realizers and are generally 
    responsible for defining the behaviour of a single construct or type of 
    construct in their context.
    """

    Self = TypeVar("Self", bound="Construct")

    def __init__(
        self: Self,
        name: Hashable,
        propagator: Propagator[It, Any, Ot] = None,
        updaters: UpdaterArg[Self] = None,
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param name: Identifier for construct, may be a ConstructSymbol, str, 
            tuple, or list. If a construct symbol is given, its construct type 
            must match the construct type associated with the realizer class. 
            If a str, tuple, or list is given, a `ConstructSymbol` will be 
            created with the given values as construct identifiers and the 
            class `ctype` as its `ConstructType`.  
        :param propagator: Activation processor associated with client 
            construct. Propagates strengths based on inputs from linked 
            constructs. It is expected that this argument will behave like a 
            `Propagator` object; this expectation is not enforced.
        :param updaters: A dict-like object containing procedures for updating 
            construct knowledge.
        """

        super().__init__(
            name=name, 
            updaters=updaters
        )
        self.propagator = propagator

    def accepts(self, source: ConstructSymbol) -> bool:
        """
        Return true if self pulls information from source.
        
        Self is deemed to pull information from source iff self.propagator 
        expects information from source.
        """

        if self.propagator is not None:
            return self.propagator.expects(construct=source)
        else:
            raise TypeError("Missing propagator.")

    def propagate(self, args: Dict = None) -> None:
        """Update output of self with result of propagator on current input."""

        if self.propagator is not None:
            packet: Ot
            inputs = cast(Any, self.inputs) # mypy complains about lru_cache
            if args is not None:
                packet = self.propagator(self.construct, inputs, **args)
            else:
                packet = self.propagator(self.construct, inputs)
            self.update_output(packet)
        else:
            raise TypeError("'NoneType' object is not callable")

    @property
    def output(self) -> Ot:
        """"Current output of self."""

        try:
            return super().output
        except super().OutputError:
            # Try to construct empty output datastructure, if constructor is 
            # available.
            if self.propagator is not None:
                self._output = cast(Propagator, self.propagator).make_packet()
                return cast(Ot, self._output)
            else:
                raise 

    @property
    def missing(self) -> MissingSpec:

        d = super().missing
        if self.propagator is None:
            d.setdefault(self.construct, []).append('propagator')
        return d


#####################################
### Container Construct Realizers ###
#####################################


# Decorator is meant to disable type_checking for the class (but not sub- or 
# superclasses). @no_type_check is not supported on mypy as of 2020-06-10.
# Disabling type checks is required here to prevent the typechecker from 
# complaining about dynamically set attributes. 
# 'type: ignore' is set to prevent mypy from complaining until the issue is 
# resolved.
# - Can
@no_type_check
class Assets(SimpleNamespace): # type: ignore
    """
    A namespace for ContainerConstruct assets.
    
    The main purpose of `Assets` objects is to provide handles for various
    datastructures such as chunk databases, rule databases, bla information, 
    etc. In general, all resources shared among different components of a 
    container construct are considered assets. 
    
    It is the user's responsibility to make sure shared resources are shared 
    and used as intended. 
    """
    pass


class Structure(Realizer[It, Ot]):
    """Base class for container construct realizers."""

    Self = TypeVar("Self", bound="Structure")

    def __init__(
        self: Self, 
        name: Hashable, 
        cycle: Cycle[It, Any, Ot] = None,
        assets: Any = None,
        updaters: UpdaterArg[Self] = None,
    ) -> None:
        """
        Initialize a new container realizer.
        """

        super().__init__(name=name, updaters=updaters)
        self._dict: Dict = {}

        self.cycle = cycle
        self.assets = assets if assets is not None else Assets()

    def __contains__(self, key: ConstructSymbol) -> bool:

        try:
            self.__getitem__(key)
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[ConstructSymbol]:

        for construct in chain(*self._dict.values()):
            yield construct

    def __getitem__(self, key: ConstructSymbol) -> Any:

        return self._dict[key.ctype][key]

    def __delitem__(self, key: ConstructSymbol) -> None:

        del self._dict[key.ctype][key]

    def accepts(self, source: ConstructSymbol) -> bool:
        """
        Return true if self pulls information from source.
        
        Self is deemed to pull information from source iff self.cycle expects 
        information from source.
        """

        if self.cycle is not None:
            return self.cycle.expects(construct=source)
        else:
            raise TypeError("Missing cycle.")

    def propagate(self: Self, args: Dict = None) -> None:

        args = args or dict()
        if self.cycle is not None:
            for ctype in self.cycle.sequence:
                for c in self.values(ctype=ctype):
                    c.propagate(args=args.get(c.construct))
            l = []
            data: Any
            if self.cycle.output is not None:
                for ctype in self.cycle.output:
                    l.append(
                        {sym: c.output for sym, c in self.items(ctype=ctype)}
                    )
                data = tuple(l)
            else:
                data = None
            packet = self.cycle.make_packet(data)
            self.update_output(packet)
        else:
            raise TypeError("Missing cycle.")

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

    def add(self, *realizers: Realizer) -> None:
        """Add a set of realizers to self."""

        for realizer in realizers:
            ctype = realizer.construct.ctype
            d = self._dict.setdefault(ctype, {})
            d[realizer.construct] = realizer
            self._update_links(realizer)

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

    def keys(self, ctype: ConstructType = None) -> Iterator[ConstructSymbol]:
        """Return iterator over all construct symbols in self."""

        for ct in self._dict:
            if ctype is None or bool(ct & ctype):
                for construct in self._dict[ct]:
                    yield construct

    def values(
        self, ctype: ConstructType = None
    ) -> Iterator[Realizer]:
        """Return iterator over all construct realizers in self."""

        for ct in self._dict:
            if ctype is None or bool(ct & ctype):
                for realizer in self._dict[ct].values():
                    yield realizer

    def items(
        self, ctype: ConstructType = None
    ) -> Iterator[Tuple[ConstructSymbol, Realizer]]:
        """Return iterator over all symbol, realizer pairs in self."""

        for ct in self._dict:
            if ctype is None or bool(ct & ctype):
                for construct, realizer in self._dict[ct].items():
                    yield construct, realizer

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
        for realizer in self.values():
            if isinstance(realizer, Structure):
                realizer.weave() 

    def unweave(self) -> None:
        """
        Remove all links to and among constructs in self.
        
        Will also remove any links from inputs to self to member constructs.
        """

        for realizer in self.values():
            realizer.drop_all()
            if isinstance(realizer, Structure):
                realizer.unweave()

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
        if self.cycle is None:
            d.setdefault(self.construct, []).append('cycle')
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

    def _update_links(self, new_realizer: Realizer) -> None:
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
