"""Provides tools for defining the behavior of simulated constructs."""


__all__ = [
    "Realizer", "Construct", "Structure", "Updater", "UpdaterArg", "PullFunc", 
    "PullFuncs", "ConstructRef"
]


from pyClarion.base.symbols import ConstructType, ConstructSymbol, MatchSpec
from pyClarion.base.propagators import Propagator, Cycle, Assets
from itertools import combinations, chain
from collections import ChainMap, OrderedDict
from types import MappingProxyType, SimpleNamespace
from typing import (
    TypeVar, Union, Container, Tuple, Dict, List, Callable, Hashable, Sequence, 
    Generic, Any, ClassVar, Optional, Type, Text, Iterator, Mapping,
    cast, no_type_check
)
from abc import abstractmethod


Rt = TypeVar('Rt') # type variable representing a construct realizer 

ConstructRef = Union[ConstructSymbol, Tuple[ConstructSymbol, ...]]
PullFunc = Callable[[], Any]
PullFuncs = Dict[ConstructSymbol, PullFunc]

# Could type annotations for updaters be improved? - Can
Updater = Callable[[Rt], None] 
# updater may be a pure ordered dict, or a list of identifier-updater pairs
UpdaterArg = Union[
    # This should be an OrderedDict, but in 3.6 generic ordered dicts are not 
    # supported (only python 3.7 and up).
    Dict[Hashable, Updater[Rt]], 
    Sequence[Tuple[Hashable, Updater[Rt]]]
]


class Realizer(object):
    """
    Base class for construct realizers.

    Construct realizers facilitate communication between constructs by 
    providing a standard interface for creating, inspecting, modifying and 
    propagating information across construct networks. 

    Message passing among constructs follows a pull-based architecture. 
    """

    Self = TypeVar("Self", bound="Realizer")

    def __init__(
        self: Self, 
        name: Hashable, 
        updaters: UpdaterArg[Self] = None
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param name: Identifier for construct.  
        :param updaters: A dict-like object containing procedures for updating 
            construct knowledge.
        """

        if not isinstance(name, ConstructSymbol):
            raise TypeError(
                "Agrument 'name' must be of type ConstructSymbol"
                "got {} instead.".format(type(name))
            )
        self._construct = name
        self._inputs: PullFuncs = {}
        self._output: Optional[Any] = None

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

    def update(self) -> None:
        """Execute learning routines."""
        
        for updater in self.updaters.values():
            updater(self)

    def accepts(self, source: ConstructSymbol) -> bool:
        """Return true if self pulls information from source."""

        raise NotImplementedError()

    def watch(
        self, construct: ConstructSymbol, callback: PullFunc
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

    def view(self) -> Any:
        """Return current output of self."""
        
        return self.output

    def update_output(self, output: Any) -> None:
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
    def inputs(self) -> Mapping[ConstructSymbol, PullFunc]:
        """Mapping from input constructs to pull funcs."""

        return MappingProxyType(self._inputs)

    @property
    def output(self) -> Any:
        """"Current output of self."""

        # Emit output if available.
        if self._output is not None:
            return self._output
        # Upon failure, throw output error.
        else:
            cls = type(self)
            repr_ = repr(self)
            raise cls.OutputError('Output of {} not defined.'.format(repr_))

    class OutputError(Exception):
        """Raised when a realizer has no output"""
        pass


# Autocomplete only works properly when bound is passed as str. Why? - Can
Pt = TypeVar("Pt", bound="Propagator")
class Construct(Realizer, Generic[Pt]):
    """
    Represents basic constructs.
    
    `Construct` objects are leaves in the construct realizer containment 
    hierarchy. That is to say they contain no other realizers and are generally 
    responsible for defining the behaviour of a single construct.
    """

    Self = TypeVar("Self", bound="Construct")

    def __init__(
        self: Self,
        name: Hashable,
        propagator: Pt,
        updaters: UpdaterArg[Self] = None,
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param name: Identifier for construct.  
        :param propagator: Activation processor associated with client 
            construct. Propagates strengths based on inputs from linked 
            constructs.
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

        return self.propagator.expects(construct=source)

    def propagate(self, args: Dict = None) -> None:
        """Update output of self with result of propagator on current input."""

        inputs = self.inputs
        args = args or dict()
        packet = self.propagator(self.construct, inputs, **args)
        self.update_output(packet)

    @property
    def output(self) -> Any:
        """"Current output of self."""

        try:
            return super().output
        except super().OutputError:
            # Try to construct empty output datastructure, if constructor is 
            # available.
            self._output = self.propagator.make_packet()
            return self._output


Ct = TypeVar("Ct", bound="Cycle")
class Structure(Realizer, Generic[Ct]):
    """Base class for container construct realizers."""

    Self = TypeVar("Self", bound="Structure")

    def __init__(
        self: Self, 
        name: Hashable, 
        cycle: Ct,
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

    def __contains__(self, key: ConstructRef) -> bool:

        try:
            self.__getitem__(key)
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[ConstructSymbol]:

        for construct in chain(*self._dict.values()):
            yield construct

    def __getitem__(self, key: ConstructRef) -> Any:

        if isinstance(key, tuple):
            if len(key) == 0:
                raise KeyError("Key sequence must be of length 1 at least.")
            elif len(key) == 1:
                return self[key[0]]
            else:
                # Catch & output more informative error here? - Can
                head = self[key[0]]
                return head[key[1:]] 
        else:
            return self._dict[key.ctype][key]

    def __delitem__(self, key: ConstructSymbol) -> None:

        # Should probably be recursive like getitem. - Can
        self.drop_links(construct=key)
        del self._dict[key.ctype][key]

    def accepts(self, source: ConstructSymbol) -> bool:
        """
        Return true if self pulls information from source.
        
        Self is deemed to pull information from source iff self.cycle expects 
        information from source.
        """

        return self.cycle.expects(construct=source)

    def propagate(self: Self, args: Dict = None) -> None:

        args = args or dict()
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

    def update(self):
        """
        Execute any knowledge updates in self and all members.
        
        Issues update calls to each updater attached to self.  
        """

        super().update()
        for realizer in self.values():
            realizer.update()

    def execute(self) -> None:
        """Execute currently selected actions."""

        raise NotImplementedError()

    def add(self, *realizers: Realizer) -> None:
        """Add realizers to self."""

        for realizer in realizers:
            ctype = realizer.construct.ctype
            d = self._dict.setdefault(ctype, {})
            d[realizer.construct] = realizer
            self.update_links(construct=realizer.construct)

    def remove(self, *constructs: ConstructSymbol) -> None:
        """Remove a set of constructs from self."""

        for construct in constructs:
            del self[construct]

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

    def watch(
        self, construct: ConstructSymbol, callback: PullFunc
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

    def update_links(self, construct: ConstructSymbol) -> None:
        """Add any acceptable links associated with a new realizer."""

        target = self[construct]
        # target._inputs.clear() # For case where connectivity is narrowed.
        for c, realizer in self.items():
            if realizer.accepts(target.construct):
                realizer.watch(target.construct, target.view)
            if target.accepts(c):
                target.watch(c, realizer.view)
        for c, callback in self._inputs.items():
            if target.accepts(c):
                target.watch(c, callback)

    def drop_links(self, construct: ConstructSymbol) -> None:
        """Remove construct from inputs of any accepting member constructs."""

        for realizer in self.values():
            if realizer.accepts(construct):
                realizer.drop(construct)

    @property
    def output(self) -> Any:
        """"Current output of self."""

        try:
            return super().output
        except super().OutputError:
            # Try to construct empty output datastructure, if constructor is 
            # available.
            self._output = self.cycle.make_packet()
            return self._output
