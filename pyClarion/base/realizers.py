"""Tools for defining construct behavior."""


__all__ = [
    "Realizer", "Construct", "Structure", "Emitter", "Propagator", "Cycle", 
    "Assets", "Updater", "FeatureInterface"
]


from pyClarion.base.symbols import ConstructType, Symbol, ConstructRef, feature
from pyClarion.utils.funcs import group_by_dims
from itertools import combinations, chain
from abc import abstractmethod
from types import MappingProxyType, SimpleNamespace
from typing import (
    TypeVar, Union, Tuple, Dict, Callable, Hashable, Generic, Any, Optional, 
    Text, Iterator, Iterable, Mapping, ClassVar, List, FrozenSet, cast, 
    no_type_check
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

    _inputs: Dict[Symbol, Callable[[], Any]]
    _output: Optional[Any]

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

        self._validate_name(name)
        self._log_init(name)

        self._construct = name
        self._inputs = {}
        self._output = emitter.emit()

        self.emitter = emitter
        self.updater = updater

        self._update_add_queue()

    def __repr__(self) -> Text:

        return "<{}: {}>".format(self.__class__.__name__, str(self.construct))

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

    @abstractmethod
    def propagate(self) -> None:
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

    def offer(self, construct: Symbol, callback: PullFunc[Any]) -> None:
        """
        Add link from construct to self if self accepts construct.
        
        :param construct: Symbol for target construct.
        :param callback: A callable that returns data representing the output 
            of construct. Typically this will be the `view()` method of a 
            Realizer instance.
        """

        if self.accepts(construct):
            self._log_watch(construct)            
            self._inputs[construct] = callback

    def drop(self, construct: Symbol) -> None:
        """Remove link from construct to self."""

        self._log_drop(construct)
        try:
            del self._inputs[construct]
        except KeyError:
            pass

    def clear_inputs(self) -> None:
        """Clear self.inputs."""

        for construct in self.inputs:
            self.drop(construct)

    def view(self) -> Any:
        """Return current output of self."""
        
        return self._output

    def _update_add_queue(self) -> None:
        """If current context contains an add queue, add self to it."""

        try:
            lst = build_list.get()
        except LookupError:
            pass
        else:
            lst.append(self)

    def _log_init(self, construct) -> None:

        tname = type(self).__name__
        try:
            context = build_ctx.get()
        except LookupError:
            msg = "Initializing %s %s."
            logging.debug(msg, tname, construct)
        else:
            msg = "Initializing %s %s in %s."
            logging.debug(msg, tname, construct, context)

    def _log_watch(self, construct: Symbol) -> None:

        try:
            context = build_ctx.get()
        except LookupError:
            logging.debug("Connecting %s to %s.", construct, self.construct)
        else:
            msg = "Connecting %s to %s in %s."
            logging.debug(msg, construct, self.construct, context)

    def _log_drop(self, construct: Symbol) -> None:

        try:
            context = build_ctx.get()
        except LookupError:
            msg = "Disconnecting %s from %s."
            logging.debug(msg, construct, self.construct)
        else:
            msg = "Disconnecting %s from %s in %s."
            logging.debug(msg, construct, self.construct, context)

    @staticmethod
    def _validate_name(name) -> None:

        if not isinstance(name, Symbol):
            msg = "Agrument 'name' must be of type Symbol got '{}' instead."
            raise TypeError(msg.format(type(name).__name__))


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

    def propagate(self) -> None:

        items = self.inputs.items()
        inputs = {src: ask() for src, ask in items if self.emitter.expects(src)}
        self.output = self.emitter(self.construct, inputs)


# TODO: Make sure that structure outputs reflect their contents accurately even 
# on the first cycle.

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
            self._log_del(key)
            self.drop(construct=key)
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

    def propagate(self) -> None:

        for ctype in self.emitter.sequence:
            for c in self.values(ctype=ctype):
                c.propagate()

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
            self._log_add(realizer.construct)
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

        self.remove(*self.keys())

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

    def offer(self, construct: Symbol, callback: PullFunc) -> None:
        """
        Add links from construct to self and any accepting members.
        
        :param construct: Symbol for target construct.
        :param callback: A callable that returns data representing the output 
            of construct. Typically this will be the `view()` method of a 
            Realizer instance.
        """

        super().offer(construct, callback)  
        with self:
            for realizer in self.values():
                realizer.offer(construct, callback)

    def drop(self, construct: Symbol) -> None:
        """Remove links from construct to self and any accepting members."""

        super().drop(construct)
        for realizer in self.values():
            realizer.drop(construct)

    def clear_inputs(self) -> None:
        """Clear self.inputs and remove all associated links."""

        for construct in self.inputs:
            for realizer in self.values():
                realizer.drop(construct)
        super().clear_inputs()           

    def update_links(self, construct: Symbol) -> None:
        """Add any acceptable links associated with member construct."""

        target = self[construct]
        for realizer in self.values():
            if target.construct != realizer.construct:
                realizer.offer(target.construct, target.view)
                target.offer(realizer.construct, realizer.view)
        for c, callback in self.inputs.items():
            target.offer(c, callback)

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

        del self.output
        for realizer in self.values():
            if isinstance(realizer, Structure):
                realizer.clear_outputs()
            else:
                del realizer.output

    def _log_del(self, construct):

        try:
            context = build_ctx.get()
        except LookupError:
            logging.debug("Removing %s from %s.", construct, self.construct)
        else:
            msg = "Removing %s from %s in %s."
            logging.debug(msg, construct, self.construct, context)

    def _log_add(self, construct):

        try:
            context = build_ctx.get()
        except LookupError:
            logging.debug("Adding %s to %s.", construct, self.construct)
        else:
            msg = "Adding %s to %s in %s." 
            logging.debug(msg, construct, self.construct, context)


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

    @staticmethod
    @abstractmethod
    def emit(data: Xt = None) -> Ot:
        """
        Emit output.

        If no data is passed in, emits a default or null value of the expected
        output type. Otherwise, ensures output is of the expected type and 
        (preferably) immutable before returning the result. 
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

    def __call__(self, construct: Symbol, inputs: Inputs[It]) -> Ot:
        """
        Execute construct's forward propagation cycle.

        Pulls expected data from inputs constructs, delegates processing to 
        self.call(), and passes result to self.emit().
        """

        return self.emit(self.call(construct, inputs))

    @abstractmethod
    def call(self, construct: Symbol, inputs: Inputs[It]) -> Xt:
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

class FeatureInterface(object):
    """
    Control interface for a component.
    
    Defines control features and default values. Provides parsing utilities.
    Each defined feature dimension is interpreted as defining a specific set of 
    alternate actions. A default value must be defined for each dimension, 
    representing the 'do nothing' action.
    """

    _features: FrozenSet[feature]
    _defaults: FrozenSet[feature]
    _tags: FrozenSet[Hashable]
    _dims: FrozenSet[Tuple[Hashable, int]]

    def __post_init__(self):

        self._validate_data()
        self._set_interface_properties()
        self._validate_interface_properties()

    @property
    def features(self):
        """The set of features defined by self."""
        
        return self._features

    @property
    def defaults(self):
        """Feature, defined by self, indicating default values, if any."""
        
        return self._defaults

    @property
    def tags(self):
        """The set of dimensional labels defined by self."""
        
        return self._tags

    @property
    def dims(self):
        """The set of feature dimensions (w/ lags) defined by self."""
        
        return self._dims

    def parse_commands(self, data):
        """
        Determine the value associated with each control dimension.
        
        :param data: A set of features.
        """

        _cmds = set(f for f in data if f in self.features)

        cmds, groups = {}, group_by_dims(features=_cmds)
        for k, g in groups.items():
            if len(g) > 1:
                msg = "Received multiple commands for dim '{}'."
                raise ValueError(msg.format(k))
            cmds[k] = g[0].val
        
        for f in self.defaults:
            if f.dim not in cmds:
                cmds[f.dim] = f.val

        return cmds

    def _validate_data(self):

        raise NotImplementedError()

    def _set_interface_properties(self):

        raise NotImplementedError()

    def _validate_interface_properties(self):

        _features_dims = set(f.dim for f in self.features)
        _features_tags = set(f.tag for f in self.features)
        _defaults_dims = set(f.dim for f in self.defaults)

        # TODO: Use a more suitable exception class. - Can

        if self.tags != _features_tags:
            raise ValueError("self.tag conflicts with self.features.")
        if self.dims != _features_dims:
            raise ValueError("self.dims conflicts with self.features.")
        if not self.defaults.issubset(self.features):
            raise ValueError("self.defaults not a subset of self.features.")
        if not self.dims.issubset(_defaults_dims):
            raise ValueError("self.defaults conflicts with self.dims.")
        if len(self.dims) != len(_defaults_dims):
            raise ValueError("multiple defaults assigned to a single dim.")
