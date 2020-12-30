"""Tools for networking constructs and defining construct behavior."""


__all__ = ["Realizer", "Construct", "Structure"]


from .symbols import ConstructType, Symbol, SymbolicAddress, SymbolTrie, feature
from .components import Process, Assets
from .. import numdicts as nd

from itertools import combinations, chain
from abc import abstractmethod
from types import MappingProxyType
from typing import (
    TypeVar, Union, Tuple, Dict, Callable, Hashable, Generic, Any, Optional, 
    Text, Iterator, Iterable, Mapping, ClassVar, List, ContextManager, cast
)
from contextlib import nullcontext
from contextvars import ContextVar
import logging


Pt = TypeVar("Pt", bound="Process")
Ot = TypeVar("Ot", bound=Union[nd.NumDict, SymbolTrie[nd.NumDict]])

PullFunc = Union[Callable[[], nd.NumDict], Callable[[], SymbolTrie[nd.NumDict]]]
PullFuncs = Mapping[Symbol, PullFunc]
StructureItem = Tuple[Symbol, "Realizer"]


# Context variables for agent construction. Helps track items to be added to 
# structures. 
BUILD_CTX: ContextVar[Tuple[Symbol, ...]] = ContextVar("BUILD_CTX", default=())
BUILD_LIST: ContextVar[List["Realizer"]] = ContextVar("BUILD_LIST")


class Realizer(Generic[Ot]):
    """
    Base class for construct realizers.

    Provides a standard interface for creating, inspecting, and propagating 
    information across construct networks.  
    """

    _inputs: Dict[Symbol, PullFunc]
    _output: Ot

    def __init__(self, name: Symbol) -> None:
        """
        Initialize a new Realizer instance.
        
        :param name: Identifier for client construct.  
        """

        self._validate_name(name)
        self._log_init(name)

        self._construct = name
        self._inputs = {}
        self._inputs_proxy = MappingProxyType(self._inputs)
        self._update_add_queue()

    def __repr__(self) -> Text:

        return "<{}: {}>".format(type(self).__name__, str(self.construct))

    @property
    def construct(self) -> Symbol:
        """Symbol for client construct."""

        return self._construct

    @property 
    def inputs(self) -> Mapping[Symbol, PullFunc]:
        """Mapping from input constructs to pull funcs."""

        return self._inputs_proxy

    @property
    @abstractmethod
    def output(self) -> Ot:
        """Construct output."""

        raise NotImplementedError()

    def view(self) -> Ot:
        """Return current output of self."""
        
        return self.output

    @abstractmethod
    def step(self) -> None:
        """Advance the simulation by one time step."""

        raise NotImplementedError()

    @abstractmethod
    def _offer(self, construct: Symbol, callback: PullFunc) -> None:
        """
        Add link from construct to self if self accepts construct.
        
        :param construct: Symbol for target construct.
        :param callback: A callable that returns data representing the output 
            of construct. Typically this will be the `view()` method of a 
            Realizer instance.
        """

        raise NotImplementedError()

    @abstractmethod
    def _finalize_assembly(self):
        """Execute final initialization and checks prior to simulation."""

        raise NotImplementedError()

    def _update_add_queue(self) -> None:
        """If current context contains an add queue, add self to it."""

        try:
            lst = BUILD_LIST.get()
        except LookupError:
            pass
        else:
            lst.append(self)

    def _log_init(self, construct) -> None:

        tname = type(self).__name__
        try:
            context = BUILD_CTX.get()
        except LookupError:
            msg = "Initializing %s %s."
            logging.debug(msg, tname, construct)
        else:
            msg = "Initializing %s %s in %s."
            logging.debug(msg, tname, construct, context)

    @staticmethod
    def _validate_name(name) -> None:

        if not isinstance(name, Symbol):
            msg = "Agrument 'name' must be of type Symbol got '{}' instead."
            raise TypeError(msg.format(type(name).__name__))


class Construct(Realizer[nd.NumDict], Generic[Pt]):
    """
    A basic construct.
    
    Responsible for defining the behaviour of lowest-level constructs such as 
    chunk or feature pools, bottom level networks, subsystem output terminals, 
    short term memory buffers and so on.
    """

    _process: Pt

    def __init__(self, name: Symbol, process: Pt) -> None:
        """
        Initialize a new construct realizer.
        
        :param name: Identifier for client construct.  
        :param process: Procedure for activation propagation. 
        """

        super().__init__(name=name)
        self._output = process.emit()
        self.process = process

    @property
    def process(self) -> Pt:
        """Process for client construct."""

        return self._process

    @process.setter
    def process(self, process: Pt) -> None:

        process.entrust(self.construct)
        self._process = process

    def step(self) -> None:

        self.output = self.process(self._pull())

    @property
    def output(self) -> nd.NumDict:

        return self._output

    @output.setter
    def output(self, output: nd.NumDict) -> None:

        self._output = output

    @output.deleter
    def output(self) -> None:
        
        self._output = self.process.emit() # default/empty output

    def _offer(self, construct: Symbol, callback: PullFunc) -> None:
        """
        Add link from construct to self if self accepts construct.
        
        :param construct: Symbol for target construct.
        :param callback: A callable that returns data representing the output 
            of construct. Typically this will be the `view()` method of a 
            Realizer instance.
        """

        if self.process.expects(construct):
            self._log_watch(construct)            
            self._inputs[construct] = callback

    def _pull(self) -> SymbolTrie[nd.NumDict]:

        return {src: ask() for src, ask in self.inputs.items()}

    def _finalize_assembly(self):
        """Execute final initialization and checks prior to simulation."""

        self.process.check_inputs(self._pull())

    def _log_watch(self, construct: Symbol) -> None:
        # Add context...

        logging.debug("Connecting %s to %s.", construct, self.construct)

        
class Structure(Realizer[SymbolTrie[nd.NumDict]]):
    """
    A composite construct.
    
    Defines behaviour of higher-level constructs, such as agents and subsystems,
    which may contain other constructs. 

    Any Realizer created within the body of a with statement having a Structure 
    object as its context manager will automatically be added to the Structure 
    upon exit from the context. Nested use of with statements in this way (e.g. 
    to add objects to subsystems within an agent) is well-behaved. The order in 
    which realizers are added to self at assembly time determines (roughly, see 
    step()) the order in which they will be stepped.
    """

    _dict: Dict[Symbol, Realizer]
    _assets: Any
    _output_dict: Dict[Symbol, Union[nd.NumDict, SymbolTrie[nd.NumDict]]]

    def __init__(self, name: Symbol, assets: Any = None) -> None:
        """
        Initialize a new Structure instance.
        
        :param name: Identifier for client construct.
        :param assets: Data structure storing persistent data shared among 
            members of self.
        """

        super().__init__(name=name)
        
        self._dict = {}
        self._dict_proxy = MappingProxyType(self._dict)
        self._output_dict = {}
        self._output = MappingProxyType(self._output_dict)
        self.assets = assets if assets is not None else Assets()

    def __contains__(self, key: SymbolicAddress) -> bool:

        try:
            self.__getitem__(key)
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[Symbol]:

        for construct in self._dict:
            yield construct

    def __getitem__(self, key: SymbolicAddress) -> Any:

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
            return self._dict[key]

    def __enter__(self):

        if 0 < len(self._dict):
            raise RuntimeError("Structure already populated.")

        logging.debug("Entering context %s.", self.construct)
        parent = BUILD_CTX.get() # default is ()
        if 1 < len(parent):
            raise RuntimeError("Maximum structure nesting depth (2) exceeded.") 
        self._build_ctx_token = BUILD_CTX.set(parent + (self.construct,))
        self._build_list_token = BUILD_LIST.set([])

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        try:
            if exc_type is None:
                context, add_list = BUILD_CTX.get(), BUILD_LIST.get()
                self._add(*add_list)
                self._reset_output()
                if len(context) <= 1:
                    assert len(context) != 0
                    self._finalize_assembly()
        finally:
            logging.debug("Exiting context %s.", self.construct)
            BUILD_CTX.reset(self._build_ctx_token)
            BUILD_LIST.reset(self._build_list_token)

    @property
    def output(self) -> SymbolTrie[nd.NumDict]:

        return self._output

    @output.setter
    def output(self, output: SymbolTrie[nd.NumDict]) -> None:

        self._output_dict.clear()
        self._output_dict.update(output.items())

    @output.deleter
    def output(self) -> None:
        """Clear output of self and all members."""

        for realizer in self._dict.values():
            del realizer.output
        self._reset_output()

    def step(self) -> None:
        """
        Advance simulation by one time step.

        Steps each member construct. Constructs of the similar types added to 
        self consecutively may theoretically be stepped in any order (e.g., 
        horizontal flows may be stepped in parallel); constructs of dissimilar 
        types will be stepped in the order that they were added to self.
        """

        # The stepping order is correct b/c in Python 3.7 and above, dictionary 
        # iteration returns values in insertion order. 
        for realizer in self._dict.values():
            realizer.step()
        self._reset_output()

    def _reset_output(self) -> None:
        """Set output of self to reflect member outputs."""

        for r in self._dict.values(): 
            if isinstance(r, Structure): 
                r._reset_output()
        self.output = {c: r.output for c, r in self._dict.items()}

    def _add(self, *realizers: Realizer) -> None:
        """Add realizers to self and any associated links."""

        for realizer in realizers:
            self._log_add(realizer.construct)
            self._dict[realizer.construct] = realizer
            self._update_links(construct=realizer.construct)

    def _update_links(self, construct: Symbol) -> None:
        """Add links between member construct and any other member of self."""

        # This may not be correct for deeply nested structures, though it works 
        # for setting up standard Clarion configurations. Current fix is to 
        # disallow nesting depth > 2 in __enter__(). - Can

        target = self[construct]
        for realizer in self._dict.values():
            if target.construct != realizer.construct:
                realizer._offer(target.construct, target.view)
                target._offer(realizer.construct, realizer.view)

    def _offer(self, construct: Symbol, callback: PullFunc) -> None:
        """
        Add links from construct to self and any accepting members.
        
        :param construct: Symbol for target construct.
        :param callback: A callable that returns data representing the output 
            of construct. Typically this will be the `view()` method of a 
            Realizer instance.
        """

        for realizer in self._dict.values():
            realizer._offer(construct, callback)

    def _finalize_assembly(self) -> None:

        for realizer in self._dict.values():
            realizer._finalize_assembly()

    def _log_add(self, construct) -> None:

        try:
            context = BUILD_CTX.get()
        except LookupError:
            logging.debug("Adding %s to %s.", construct, self.construct)
        else:
            msg = "Adding %s to %s in %s." 
            logging.debug(msg, construct, self.construct, context)
