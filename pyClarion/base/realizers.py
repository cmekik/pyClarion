"""Provides tools for defining the behavior of simulated constructs."""


__all__ = ["Realizer", "Construct", "Structure"]


from pyClarion.base.symbols import ConstructType, Symbol, ConstructRef
from pyClarion.base.propagators import Emitter, Propagator, Cycle, Assets
from itertools import combinations, chain
from types import MappingProxyType
from typing import (
    TypeVar, Union, Tuple, Dict, Callable, Hashable, Generic, Any, Optional, 
    Text, Iterator, Mapping, cast
)


PullFunc = Callable[[], Any]
PullFuncs = Dict[Symbol, PullFunc]
Rt = TypeVar('Rt', bound="Realizer") 
Updater = Callable[[Rt], None] # Could this be improved? - Can
StructureItem = Tuple[Symbol, "Realizer"]


# Autocomplete only works properly when bound is passed as str. Why? - Can
# Et = TypeVar("Et", bound="Emitter[It, Ot]") is more desirable and similar 
# for Pt & Ct below; but, this is not supported as of 2020-07-20. - Can
Et = TypeVar("Et", bound="Emitter")
R = TypeVar("R", bound="Realizer")
class Realizer(Generic[Et]):
    """
    Base class for construct realizers.

    Construct realizers facilitate communication between constructs by 
    providing a standard interface for creating, inspecting, modifying and 
    propagating information across construct networks. 

    Message passing among constructs follows a pull-based architecture. 
    """

    def __init__(
        self: R, name: Symbol, emitter: Et, updater: Updater[R] = None
    ) -> None:
        """
        Initialize a new construct realizer.
        
        :param name: Identifier for construct.  
        :param updater: Procedure for updating persistent construct data.
        """

        if not isinstance(name, Symbol):
            raise TypeError(
                "Agrument 'name' must be of type Symbol"
                "got {} instead.".format(type(name))
            )

        self._construct = name
        self._inputs: PullFuncs = {}
        self._output: Optional[Any] = None

        self.emitter = emitter
        self.updater = updater

    def __repr__(self) -> Text:

        return "<{}: {}>".format(self.__class__.__name__, str(self.construct))

    def propagate(self, kwds: Dict = None) -> None:
        """
        Propagate activations.

        :param kwds: Keyword arguments for propagation procedure.
        """

        raise NotImplementedError()

    def update(self: R) -> None:
        """Update persistent data associated with self."""
        
        if self.updater is not None:
            self.updater(self)

    def accepts(self, source: Symbol) -> bool:
        """Return true iff self pulls information from source."""

        return self.emitter.expects(source)

    def watch(self, construct: Symbol, callback: PullFunc) -> None:
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

        try:
            del self._inputs[construct]
        except KeyError:
            pass

    def clear_inputs(self) -> None:
        """Disconnect self from all linked constructs."""

        self._inputs.clear()

    def view(self) -> Any:
        """Return current output of self."""
        
        return self.output

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
            self._output = self.emitter.emit() # Default/empty output.
            return self._output

    @output.setter
    def output(self, output: Any) -> None:

        self._output = output

    @output.deleter
    def output(self) -> None:
        
        self._output = None


Pt = TypeVar("Pt", bound="Propagator")
C = TypeVar("C", bound="Construct")
class Construct(Realizer[Pt]):
    """
    A basic construct.
    
    Responsible for defining the behaviour of the lowest-level constructs such 
    as individual nodes, bottom level networks, top level rule databases, 
    subsystem output terminals, short term memory buffers and so on.
    """

    def __init__(
        self: C,
        name: Symbol,
        emitter: Pt,
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

        super().__init__(name=name, emitter=emitter, updater=updater)

    def propagate(self, kwds: Dict = None) -> None:
        """Update output of self with result of propagator on current input."""

        inputs = self.inputs
        kwds = kwds or dict()
        self.output = self.emitter(self.construct, inputs, **kwds)


Ct = TypeVar("Ct", bound="Cycle")
S = TypeVar("S", bound="Structure")
class Structure(Realizer[Ct]):
    """A composite construct."""

    def __init__(
        self: S, 
        name: Symbol, 
        emitter: Ct,
        assets: Any = None,
        updater: Updater[S] = None,
    ) -> None:
        """Initialize a new Structure instance."""

        super().__init__(name=name, emitter=emitter, updater=updater)
        
        self._dict: Dict = {}
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

    def propagate(self, kwds: Dict = None) -> None:

        kwds = kwds or dict()
        for ctype in self.emitter.sequence:
            for c in self.values(ctype=ctype):
                c.propagate(kwds=kwds.get(c.construct))

        ctype = self.emitter.output
        data = {sym: c.output for sym, c in self.items(ctype=ctype)}
        self.output = self.emitter.emit(data)

    def update(self):
        """Update persistent data in self and all members."""

        super().update()
        for realizer in self.values():
            realizer.update()

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

    def items(self, ctype: ConstructType = None) -> Iterator[StructureItem]:
        """Return iterator over all symbol, realizer pairs in self."""

        for ct in self._dict:
            if ctype is None or bool(ct & ctype):
                for construct, realizer in self._dict[ct].items():
                    yield construct, realizer

    def watch(self, construct: Symbol, callback: PullFunc) -> None:
        """Add construct as an input to self and any accepting members."""

        super().watch(construct, callback)
        for realizer in self.values():
            if realizer.accepts(construct):
                realizer.watch(construct, callback)

    def drop(self, construct: Symbol) -> None:
        """Remove construct as an input to self and any accepting members."""

        super().drop(construct)
        for realizer in self.values():
            realizer.drop(construct)

    def clear_inputs(self) -> None:
        """Remove all inputs to self from self and any accepting members."""

        for construct in self._inputs:
            for realizer in self.values():
                realizer.drop(construct)
        super().clear_inputs()           

    def update_links(self, construct: Symbol) -> None:
        """Add any acceptable links associated with construct."""

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
        """Remove any existing links from construct to any member."""

        for realizer in self.values():
            realizer.drop(construct)

    def clear_links(self) -> None:
        """Remove all links to, among, and within all members of self."""

        for realizer in self.values():
            realizer.clear_inputs()
            if isinstance(realizer, Structure):
                realizer.clear_links()

    def reweave(self) -> None:
        """Recompute all links to, among, and within constructs in self."""

        self.clear_links()
        for construct, realizer in self.items():
            if isinstance(realizer, Structure):
                realizer.reweave()
            self.update_links(construct)

    def clear_outputs(self) -> None:
        """Clear output of self and all members."""

        del self._output
        for realizer in self.values():
            if isinstance(realizer, Structure):
                realizer.clear_outputs()
            else:
                del realizer.output
