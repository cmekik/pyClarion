"""Tools for defining construct behavior."""


__all__ = [
    "Realizer", "Construct", "Structure", "Emitter", "Propagator", "Cycle", 
    "Assets", "Updater"
]


from pyClarion.base.symbols import ConstructType, Symbol, ConstructRef, MatchSet
from itertools import combinations, chain
from abc import abstractmethod
from types import MappingProxyType, SimpleNamespace
from typing import (
    TypeVar, Union, Tuple, Dict, Callable, Hashable, Generic, Any, Optional, 
    Text, Iterator, Iterable, Mapping, ClassVar, List, cast, no_type_check
)
import logging
from contextvars import ContextVar


Dt = TypeVar('Dt') # type variable for inputs to emitters
Inputs = Mapping[Symbol, Dt]
PullFunc = Callable[[], Dt]
PullFuncs = Mapping[Symbol, Callable[[], Dt]]

Rt = TypeVar('Rt', bound="Realizer") 
Updater = Callable[[Rt], None] # Could this be improved? - Can
StructureItem = Tuple[Symbol, "Realizer"]

It = TypeVar('It', contravariant=True) # type variable for emitter inputs
Ot = TypeVar('Ot', covariant=True) # type variable for emitter outputs

# Context variables for automating/simplifying agent construction. Helps track
# items to be added to structures. 
build_ctx: ContextVar[ConstructRef] = ContextVar("build_ctx")
build_list: ContextVar[List["Realizer"]] = ContextVar("build_list")

# Autocomplete only works properly when bound is str. Why? - Can
# Et = TypeVar("Et", bound="Emitter[It, Ot]") is more desirable and similar 
# for Pt & Ct below; but, this is not supported as of 2020-07-20. - Can
Et = TypeVar("Et", bound="Emitter")
R = TypeVar("R", bound="Realizer")
class Realizer(Generic[Et]):
    """
    Base class for construct realizers.

    Provides a standard interface for creating, inspecting, modifying and 
    propagating information across construct networks. 

    Follows a pull-based message-passing pattern for activation propagation and 
    a blackboard pattern for updates to persistent data. 
    """

    def __init__(
        self: R, name: Symbol, emitter: Et, updater: Updater[R] = None
    ) -> None:
        """
        Initialize a new Realizer instance.
        
        :param name: Identifier for client construct.  
        :param emitter: Procedure for activation propagation. Expected to be of 
            type Emitter.
        :param updater: Procedure for updating persistent construct data.
        """

        if not isinstance(name, Symbol):
            raise TypeError(
                "Agrument 'name' must be of type Symbol "
                "got '{}' instead.".format(type(name).__name__)
            )

        self._construct = name
        self._inputs: Dict[Symbol, Callable[[], Any]] = {}
        self._output: Optional[Any] = emitter.emit()

        self.emitter = emitter
        self.updater = updater

        # If current context contains an add stack, add self to it. 
        # If not, do nothing.
        try:
            parent, lst = build_ctx.get(), build_list.get()
            lst.append(self)
            logging.debug(
                "Built %s %s in %s.", 
                type(self).__name__, self.construct, parent
            )
        except LookupError:
            logging.debug("Built %s %s.", type(self).__name__, self.construct)

    def __repr__(self) -> Text:

        return "<{}: {}>".format(self.__class__.__name__, str(self.construct))

    @abstractmethod
    def propagate(self, kwds: Dict = None) -> None:
        """
        Propagate activations.

        :param kwds: Keyword arguments for emitter.
        """

        raise NotImplementedError()

    def update(self: R) -> None:
        """Update persistent data associated with self."""
        
        if self.updater is not None:
            self.updater(self)

    def accepts(self, source: Symbol) -> bool:
        """Return true iff self pulls information from source."""

        return self.emitter.expects(source)

    def watch(self, construct: Symbol, callback: PullFunc[Any]) -> None:
        """
        Add link from construct to self.
        
        :param construct: Symbol for target construct.
        :param callback: A callable that returns data representing the output 
            of construct. Typically this will be the `view()` method of a 
            Realizer instance.
        """

        try:
            parent = build_ctx.get()
            logging.debug(
                "Connecting %s to %s in %s.", construct, self.construct, parent
            )
        except LookupError:
            logging.debug("Connecting %s to %s.", construct, self.construct)
            
        self._inputs[construct] = callback

    def drop(self, construct: Symbol) -> None:
        """Remove link from construct to self."""

        try:
            parent = build_ctx.get()
            logging.debug(
                "Disconnecting %s from %s in %s.", 
                construct, self.construct, parent
            )
        except LookupError:
            logging.debug(
                "Disconnecting %s from %s.", construct, self.construct
            )

        try:
            del self._inputs[construct]
        except KeyError:
            pass

    def clear_inputs(self) -> None:
        """Clear self.inputs."""

        self._inputs.clear()

    def view(self) -> Any:
        """Return current output of self."""
        
        return self._output

    @property
    def construct(self) -> Symbol:
        """Symbol for client construct."""

        return self._construct

    @property 
    def inputs(self) -> Mapping[Symbol, PullFunc[Any]]:
        """Mapping from input constructs to pull funcs."""

        return MappingProxyType(self._inputs)

    @property
    def output(self) -> Any:
        """
        Current output of self.
        
        Deleteing this attribute will simply revert it to the default value set 
        by self.emitter.
        """

        return self._output

    @output.setter
    def output(self, output: Any) -> None:

        self._output = output

    @output.deleter
    def output(self) -> None:
        
        self._output = self.emitter.emit() # default/empty output


Pt = TypeVar("Pt", bound="Propagator")
C = TypeVar("C", bound="Construct")
class Construct(Realizer[Pt]):
    """
    A basic construct.
    
    Responsible for defining the behaviour of lowest-level constructs such as 
    individual nodes, bottom level networks, top level rule databases, 
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
        
        :param name: Identifier for client construct.  
        :param emitter: Procedure for activation propagation. Expected to be of 
            type Propagator.
        :param updater: Procedure for updating persistent construct data.
        """

        super().__init__(name=name, emitter=emitter, updater=updater)

    def propagate(self, kwds: Dict = None) -> None:

        inputs = self.inputs
        kwds = kwds or dict()
        self.output = self.emitter(self.construct, inputs, **kwds)


Ct = TypeVar("Ct", bound="Cycle")
S = TypeVar("S", bound="Structure")
class Structure(Realizer[Ct]):
    """
    A composite construct.
    
    Defines behaviour of higher-level constructs, such as agents and 
    subsystems, which may contain other constructs. 
    """

    # NOTE: It would be nice to have structures automatically connect to 
    # elements that their members accept. My first past attempt at this failed 
    # miserably. Does not seem doable w/o allowing realizers to have knowledge 
    # of their parents. I don't really want to do that unless it is absolutely 
    # necessary. - Can

    def __init__(
        self: S, 
        name: Symbol, 
        emitter: Ct,
        assets: Any = None,
        updater: Updater[S] = None,
    ) -> None:
        """
        Initialize a new Structure instance.
        
        :param name: Identifier for client construct.  
        :param emitter: Procedure for activation propagation. Expected to be of 
            type Emitter.
        :param assets: Data structure storing persistent data shared among 
            members of self.
        :param updater: Procedure for updating persistent construct data.
        """

        super().__init__(name=name, emitter=emitter, updater=updater)
        
        self._dict: Dict[ConstructType, Dict[Symbol, Realizer]] = {}
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

    # TODO: Recursive application needs testing. - Can
    def __delitem__(self, key: ConstructRef) -> None:

        if isinstance(key, tuple):
            if len(key) == 0:
                raise KeyError("Key sequence must be of length 1 at least.")
            elif len(key) == 1:
                del self[key[0]]
            else:
                # Catch & output more informative error here? - Can
                head = self[key[0]]
                del head[key[1:]] 
        else:
            
            self.drop_links(construct=key)

            try:
                parent = build_ctx.get()
                logging.debug(
                    "Removing %s from %s in %s.", key, self.construct, parent
                )
            except LookupError:
                logging.debug("Removing %s from %s.", key, self.construct)

            del self._dict[key.ctype][key]

    def __enter__(self):

        logging.debug("Entering context %s.", self.construct)
        # This sets the context variable up to track objects to be added to 
        # self.
        parent = build_ctx.get(())
        self._build_ctx_token = build_ctx.set(parent + (self.construct,))
        self._build_list_token = build_list.set([])


    def __exit__(self, exc_type, exc_value, traceback):

        # Add any newly defined realizers to self and clean up the context.
        _, add_list = build_ctx.get(), build_list.get()
        for realizer in add_list:
            self.add(realizer)
        build_ctx.reset(self._build_ctx_token)
        build_list.reset(self._build_list_token)
        logging.debug("Exiting context %s.", self.construct)

    def propagate(self, kwds: Dict = None) -> None:

        kwds = kwds or dict()
        for ctype in self.emitter.sequence:
            for c in self.values(ctype=ctype):
                c.propagate(kwds=kwds.get(c.construct))

        ctype = self.emitter.output
        data = {sym: c.output for sym, c in self.items(ctype=ctype)}
        self.output = self.emitter.emit(data)

    def update(self):
        """
        Update persistent data in self and all members.
        
        First calls `self.updater`, then calls `realizer.update()` on members. 
        In otherwords, updates are applied in a top-down manner relative to
        construct containment.
        """

        super().update()
        for realizer in self.values():
            realizer.update()

    def add(self, *realizers: Realizer) -> None:
        """
        Add realizers to self and any associated links.
        
        Calling add directly when building an agent may be error-prone, not to 
        mention cumbersome. A better approach may be to try using self as a 
        context manager. 
        
        When self is used as a context manager, any construct initialized 
        within the body of a with statement having self as its context manager 
        will automatically be added to self upon exit from the context. Nested 
        use of with statements in this way (e.g. to add objects to subsystems 
        within an agent) is well-behaved.
        """

        for realizer in realizers:

            try:
                parent = build_ctx.get()
                logging.debug(
                    "Adding %s to %s in %s.", 
                    realizer.construct, self.construct, parent
                )
            except LookupError:
                logging.debug(
                    "Adding %s to %s.", realizer.construct, self.construct
                )

            ctype = realizer.construct.ctype
            d = self._dict.setdefault(ctype, {})
            d[realizer.construct] = realizer
            self.update_links(construct=realizer.construct)

    def remove(self, *constructs: Symbol) -> None:
        """Remove constructs from self and any associated links."""

        for construct in constructs:
            del self[construct]

    def clear(self):
        """Remove all constructs in self."""

        self._dict.clear()

    def keys(self, ctype: ConstructType = None) -> Iterator[Symbol]:
        """
        Return iterator over all construct symbols in self.
        
        :param ctype: If provided, only constructs of a type that have a 
            non-empty intersection with ctype will be included.
        """

        for ct in self._dict:
            if ctype is None or bool(ct & ctype):
                for construct in self._dict[ct]:
                    yield construct

    def values(self, ctype: ConstructType = None) -> Iterator[Realizer]:
        """
        Return iterator over all construct realizers in self.
        
        :param ctype: If provided, only constructs of a type that have a 
            non-empty intersection with ctype will be included.
        """

        for ct in self._dict:
            if ctype is None or bool(ct & ctype):
                for realizer in self._dict[ct].values():
                    yield realizer

    def items(self, ctype: ConstructType = None) -> Iterator[StructureItem]:
        """
        Return iterator over all symbol, realizer pairs in self.
        
        :param ctype: If provided, only constructs of a type that have a 
            non-empty intersection with ctype will be included.
        """

        for ct in self._dict:
            if ctype is None or bool(ct & ctype):
                for construct, realizer in self._dict[ct].items():
                    yield construct, realizer

    def watch(self, construct: Symbol, callback: PullFunc) -> None:
        """
        Add links from construct to self and any accepting members.
        
        :param construct: Symbol for target construct.
        :param callback: A callable that returns data representing the output 
            of construct. Typically this will be the `view()` method of a 
            Realizer instance.
        """

        super().watch(construct, callback)
  
        # Context included for logging purposes only.
        with self:
            for realizer in self.values():
                if realizer.accepts(construct):
                    realizer.watch(construct, callback)

    def drop(self, construct: Symbol) -> None:
        """Remove links from construct to self and any accepting members."""

        super().drop(construct)
        for realizer in self.values():
            realizer.drop(construct)

    def clear_inputs(self) -> None:
        """Clear self.inputs and remove all associated links."""

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
        """Remove any existing links from construct to any member of self."""

        for realizer in self.values():
            realizer.drop(construct)

    def clear_links(self) -> None:
        """Remove all links to, among, and within all members of self."""

        for realizer in self.values():
            realizer.clear_inputs()
            if isinstance(realizer, Structure):
                realizer.clear_links()

    def reweave(self) -> None:
        """Recompute all links to, among, and within members of self."""

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


Xt = TypeVar("Xt")
class Emitter(Generic[Xt, Ot]):
    """
    Base class for propagating strengths, decisions, etc.

    Emitters define how constructs connect, process inputs, and set outputs.
    """

    @abstractmethod
    def expects(self, construct: Symbol):
        """Return True iff self expects input from construct."""

        raise NotImplementedError

    @abstractmethod
    def emit(self, data: Xt = None) -> Ot:
        """
        Emit output.

        If no data is passed in, emits a default or null value of the expected
        output type. Otherwise, ensures output is of the expected type and 
        before returning the result. 
        """

        raise NotImplementedError()


T = TypeVar('T', bound="Propagator")
class Propagator(Emitter[Xt, Ot], Generic[It, Xt, Ot]):
    """Emitter for basic constructs."""

    def __copy__(self: T) -> T:
        """
        Make a copy of self.

        Enables use of propagator instances as templates. Should ensure that 
        mutation of copies do not have unwanted side-effects.
        """
        raise NotImplementedError() 

    def __call__(
        self, construct: Symbol, inputs: PullFuncs[It], **kwds: Any
    ) -> Ot:
        """
        Execute construct's forward propagation cycle.

        Pulls expected data from inputs constructs, delegates processing to 
        self.call(), and passes result to self.emit().
        """

        inputs_ = {
            source: pull_func() for source, pull_func in inputs.items()
            if self.expects(source)
        }
        intermediate: Xt = self.call(construct, inputs_, **kwds)
        
        return self.emit(intermediate)

    @abstractmethod
    def call(self, construct: Symbol, inputs: Inputs[It], **kwds: Any) -> Xt:
        """
        Compute construct's output.

        :param construct: Name of the client construct. 
        :param inputs: Pairs the names of input constructs with their outputs. 
        :param kwds: Optional parameters. Propagator instances are recommended 
            to throw errors upon receipt of unexpected keywords.
        """

        raise NotImplementedError()


class Cycle(Emitter[Xt, Ot]):
    """Emitter for composite constructs."""

    # Specifies data required to construct the output packet
    output: ClassVar[ConstructType] = ConstructType.null_construct
    sequence: Iterable[ConstructType]
    

# @no_type_check disables type_checking for Assets (but not subclasses). 
# Included b/c dynamic usage of Assets causes mypy to complain.
# @no_type_check is not supported on mypy as of 2020-06-10. 'type: ignore' will 
# do for now. - Can
@no_type_check
class Assets(SimpleNamespace): # type: ignore
    """
    Dynamic namespace for construct assets.
    
    Provides handles for various datastructures such as chunk databases, rule 
    databases, bla information, etc. In general, all resources shared among 
    different components of a container construct are considered assets. 
    """
    pass
