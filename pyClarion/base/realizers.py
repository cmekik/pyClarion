"""Tools for networking constructs and defining construct behavior."""


__all__ = ["Realizer", "Construct", "Structure"]


from .symbols import ConstructType, Symbol, SymbolicAddress, feature
from .components import Process, Assets
from .. import numdicts as nd

from itertools import combinations, chain
from abc import abstractmethod
from types import MappingProxyType
from contextlib import nullcontext
from contextvars import ContextVar
from typing import (
    TypeVar, Union, Tuple, Dict, Callable, Hashable, Generic, Any, Optional, 
    Text, Iterator, Iterable, Mapping, ClassVar, List, ContextManager, cast
)
import logging


Pt = TypeVar("Pt", bound="Process")
Ot = TypeVar("Ot", bound=Union[nd.NumDict, Mapping[Any, nd.NumDict]])

PullFunc = Callable[[], nd.NumDict]
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

    _parent: Tuple[Symbol, ...]
    _inputs: Dict[Tuple[Symbol, ...], PullFunc]

    def __init__(self, name: Symbol) -> None:
        """
        Initialize a new Realizer instance.
        
        :param name: Identifier for client construct.  
        """

        self._validate_name(name)
        self._log_init(name)

        try:
            self._parent = BUILD_CTX.get()
        except LookupError:
            self._parent = ()
        self._name = name
        self._inputs = {}
        self._inputs_proxy = MappingProxyType(self._inputs)
        self._update_add_queue()

    def __repr__(self) -> Text:

        return "<{}: {}>".format(type(self).__name__, str(self.name))

    @property
    def name(self) -> Symbol:
        """Symbol for client construct."""

        return self._name

    @property
    def parent(self) -> Tuple[Symbol, ...]:
        """Symbolic path to parent structure."""

        return self._parent

    @property
    def path(self) -> Tuple[Symbol, ...]:
        """Symbolic path to self."""

        return (*self._parent, self._name)

    @property 
    def inputs(self) -> Mapping[Tuple[Symbol, ...], PullFunc]:
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

        process.entrust(self.path)
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

    def _link(self, path: Tuple[Symbol, ...], callback: PullFunc) -> None:
        """
        Add link from construct at path to self.
        
        :param path: Symbolic path for target construct.
        :param callback: A callable that returns data representing the output 
            of construct. Typically this will be the `view()` method of a 
            Realizer instance.
        """

        logging.debug("Connecting %s to %s.", path, self.path)
        self._inputs[path] = callback

    def _pull(self) -> Mapping[Tuple[Symbol, ...], nd.NumDict]:

        return {src: ask() for src, ask in self.inputs.items()}

        
class Structure(Realizer[Mapping[Any, nd.NumDict]]):
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

    # TODO: Deep nesting needs testing. - Can

    _dict: Dict[Symbol, Realizer]
    _assets: Any

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

        logging.debug("Entering context %s.", self.name)
        if 0 < len(self._dict): # This could probably be relaxed.
            raise RuntimeError("Structure already populated.")
        parent = BUILD_CTX.get() # default is ()
        if 1 < len(parent): # See _upate_links() for rationale.
            raise RuntimeError("Maximum structure nesting depth (2) exceeded.") 
        self._build_ctx_token = BUILD_CTX.set(parent + (self.name,))
        self._build_list_token = BUILD_LIST.set([])

        return self

    def __exit__(self, exc_type, exc_value, traceback):

        try: # Populate structure
            if exc_type is None:
                context, add_list = BUILD_CTX.get(), BUILD_LIST.get()
                self._add(*add_list)
                if len(context) <= 1:
                    assert len(context) != 0
                    self._weave()
        finally:
            logging.debug("Exiting context %s.", self.name)
            BUILD_CTX.reset(self._build_ctx_token)
            BUILD_LIST.reset(self._build_list_token)

    @property
    def output(self) -> Mapping[Any, nd.NumDict]:

        key: SymbolicAddress
        output, split = {}, len(self.path)
        for r in self._leaves():
            tail = r.path[split:]
            if len(tail) == 1:
                key, = tail
            else:
                key = tail
            output[key] = r.output

        return output

    @output.deleter
    def output(self) -> None:
        """Clear output of self and all members."""

        for realizer in self._dict.values():
            del realizer.output

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

    def _add(self, *realizers: Realizer) -> None:
        """Add realizers to self and any associated links."""

        for realizer in realizers:
            logging.debug("Adding %s to %s.", realizer.name, self.path)
            self._dict[realizer.name] = realizer       

    def _leaves(self) -> Iterator[Construct]:
        """Iterate over all Construct instances in self."""

        for realizer in self._dict.values():
            if isinstance(realizer, Construct):
                yield realizer
            else:
                assert isinstance(realizer, Structure)
                for element in realizer._leaves():
                    yield element

    def _weave(self) -> None:
        """Link all constructs in self."""

        split = len(self.path)
        for realizer in self._leaves():
            for path in realizer.process.expected:
                head, tail = path[:split], path[split:] 
                if head != self.path:
                    raise ValueError("Unexpected path.")
                try:
                    view = self[tail].view
                except KeyError as e:
                    raise RuntimeError("Missing construct") from e
                else:
                    realizer._link(path, view)
