"""Provides tools for defining the behavior of simulated constructs."""


__all__ = [
    "Realizer", "Construct", "Structure", "Updater", "PullFunc", "PullFuncs", 
    "ConstructRef"
]


from pyClarion.base.symbols import ConstructType, Symbol
from pyClarion.base.propagators import Propagator, Cycle, Assets
from itertools import combinations, chain
from types import MappingProxyType
from typing import (
    TypeVar, Union, Tuple, Dict, Callable, Hashable, Generic, Any, Optional, 
    Text, Iterator, Mapping,
)


ConstructRef = Union[Symbol, Tuple[Symbol, ...]]
PullFunc = Callable[[], Any]
PullFuncs = Dict[Symbol, PullFunc]
Rt = TypeVar('Rt', bound="Realizer") 
Updater = Callable[[Rt], None] # Could this be improved? - Can


R = TypeVar("R", bound="Realizer")
class Realizer(object):
    """
    Base class for construct realizers.

    Construct realizers facilitate communication between constructs by 
    providing a standard interface for creating, inspecting, modifying and 
    propagating information across construct networks. 

    Message passing among constructs follows a pull-based architecture. 
    """


    def __init__(
        self: R, name: Symbol, updater: Updater[R] = None
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param name: Identifier for construct.  
        :param updater: A dict-like object containing procedures for updating 
            construct knowledge.
        """

        if not isinstance(name, Symbol):
            raise TypeError(
                "Agrument 'name' must be of type Symbol"
                "got {} instead.".format(type(name))
            )

        self._construct = name
        self._inputs: PullFuncs = {}
        self._output: Optional[Any] = None

        self.updater = updater


    def __repr__(self) -> Text:

        return "<{}: {}>".format(self.__class__.__name__, str(self.construct))

    def propagate(self, args: Dict = None) -> None:
        """
        Propagate activations.

        :param args: A dict containing optional arguments for self and 
            subordinate constructs (if any).
        """

        raise NotImplementedError()

    def update(self: R) -> None:
        """Execute learning routines."""
        
        if self.updater is not None:
            self.updater(self)

    def accepts(self, source: Symbol) -> bool:
        """Return true if self pulls information from source."""

        raise NotImplementedError()

    def watch(
        self, construct: Symbol, callback: PullFunc
    ) -> None:
        """
        Set given construct as an input to self.
        
        :param construct: Symbol for target construct.
        :param callback: A callable that returns a `Packet` representing the 
            output of the target construct. Typically this will be the `view()` 
            method of a construct realizer.
        """

        self._inputs[construct] = callback

    def drop(self, construct: Symbol) -> None:
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
    def construct(self) -> Symbol:
        """Client construct of self."""

        return self._construct

    @property 
    def inputs(self) -> Mapping[Symbol, PullFunc]:
        """Mapping from input constructs to pull funcs."""

        return MappingProxyType(self._inputs)

    @property
    def output(self) -> Any:
        """"Current output of self."""

        if self._output is not None:
            return self._output
        else:
            cls, repr_ = type(self), repr(self)
            raise cls.OutputError('Output of {} not defined.'.format(repr_))

    class OutputError(Exception):
        """Raised when a realizer has no output"""
        pass


# Autocomplete only works properly when bound is passed as str. Why? - Can
Pt = TypeVar("Pt", bound="Propagator")
C = TypeVar("C", bound="Construct")
class Construct(Realizer, Generic[Pt]):
    """
    Represents basic constructs.
    
    `Construct` objects are leaves in the construct realizer containment 
    hierarchy. That is to say they contain no other realizers and are generally 
    responsible for defining the behaviour of a single construct.
    """

    def __init__(
        self: C,
        name: Symbol,
        propagator: Pt,
        updater: Updater[C] = None,
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param name: Identifier for construct.  
        :param propagator: Activation processor associated with client 
            construct. Propagates strengths based on inputs from linked 
            constructs.
        :param updater: A dict-like object containing procedures for updating 
            construct knowledge.
        """

        super().__init__(name=name, updater=updater)
        self.propagator = propagator

    def accepts(self, source: Symbol) -> bool:
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
        output = self.propagator(self.construct, inputs, **args)
        self.update_output(output)

    @property
    def output(self) -> Any:
        """"Current output of self."""

        try:
            return super().output
        except super().OutputError:
            self._output = self.propagator.emit() # Default/empty output.
            return self._output


Ct = TypeVar("Ct", bound="Cycle")
S = TypeVar("S", bound="Structure")
class Structure(Realizer, Generic[Ct]):
    """Base class for container construct realizers."""

    def __init__(
        self: S, 
        name: Symbol, 
        cycle: Ct,
        assets: Any = None,
        updater: Updater[S] = None,
    ) -> None:
        """
        Initialize a new container realizer.
        """

        super().__init__(name=name, updater=updater)
        self._dict: Dict = {}

        self.cycle = cycle
        self.assets = assets if assets is not None else Assets()

    def __contains__(self, key: ConstructRef) -> bool:

        try:
            self.__getitem__(key)
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[Symbol]:

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

    def __delitem__(self, key: Symbol) -> None:

        # Should probably be recursive like getitem. - Can
        self.drop_links(construct=key)
        del self._dict[key.ctype][key]

    def accepts(self, source: Symbol) -> bool:
        """
        Return true if self pulls information from source.
        
        Self is deemed to pull information from source iff self.cycle expects 
        information from source.
        """

        return self.cycle.expects(construct=source)

    def propagate(self, args: Dict = None) -> None:

        args = args or dict()
        for ctype in self.cycle.sequence:
            for c in self.values(ctype=ctype):
                c.propagate(args=args.get(c.construct))

        ctype = self.cycle.output
        data = {sym: c.output for sym, c in self.items(ctype=ctype)}
        output = self.cycle.emit(data)
        self.update_output(output)

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

    def remove(self, *constructs: Symbol) -> None:
        """Remove a set of constructs from self."""

        for construct in constructs:
            del self[construct]

    def clear(self):
        """Remove all constructs in self."""

        self._dict.clear()

    def keys(self, ctype: ConstructType = None) -> Iterator[Symbol]:
        """Return iterator over all construct symbols in self."""

        for ct in self._dict:
            if ctype is None or bool(ct & ctype):
                for construct in self._dict[ct]:
                    yield construct

    def values(self, ctype: ConstructType = None) -> Iterator[Realizer]:
        """Return iterator over all construct realizers in self."""

        for ct in self._dict:
            if ctype is None or bool(ct & ctype):
                for realizer in self._dict[ct].values():
                    yield realizer

    def items(
        self, ctype: ConstructType = None
    ) -> Iterator[Tuple[Symbol, Realizer]]:
        """Return iterator over all symbol, realizer pairs in self."""

        for ct in self._dict:
            if ctype is None or bool(ct & ctype):
                for construct, realizer in self._dict[ct].items():
                    yield construct, realizer

    def watch(
        self, construct: Symbol, callback: PullFunc
    ) -> None:
        """
        Add construct as an input to self. 
        
        Also adds construct as input to any interested construct in self.
        """

        super().watch(construct, callback)
        for realizer in self.values():
            if realizer.accepts(construct):
                realizer.watch(construct, callback)

    def drop(self, construct: Symbol) -> None:
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

    def update_links(self, construct: Symbol) -> None:
        """Add any acceptable links associated with a new realizer."""

        target = self[construct]
        for c, realizer in self.items():
            if realizer.accepts(target.construct):
                realizer.watch(target.construct, target.view)
            if target.accepts(c):
                target.watch(c, realizer.view)
        for c, callback in self._inputs.items():
            if target.accepts(c):
                target.watch(c, callback)

    def drop_links(self, construct: Symbol) -> None:
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
            self._output = self.cycle.emit()
            return self._output
