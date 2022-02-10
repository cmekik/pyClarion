"""Tools for networking simulated constructs."""


from __future__ import annotations
from abc import abstractmethod
from types import MappingProxyType
from contextvars import ContextVar
from typing import (Union, Tuple, Callable, Any, Sequence, Iterator, ClassVar, 
    List, OrderedDict)
from functools import partial
import logging

from .processes import Process
from . import uris
from .. import numdicts as nd


__all__ = ["Module", "Structure"]


BUILD_CTX: ContextVar = ContextVar("BUILD_CTX", default=uris.SEP)
BUILD_LIST: ContextVar = ContextVar("BUILD_LIST")


class Construct:
    """Base class for simulated constructs."""

    _parent: str
    _name: str

    def __init__(self, name: str) -> None:
        """
        Initialize a new Construct instance.
        
        :param name: Construct identifier. Must be a valid Python identifier.  
        """

        if not name.isidentifier():
            raise ValueError("Name must be a valid Python identifier.")
        self._log_init(name)
        self._parent = BUILD_CTX.get()
        self._name = name
        self._update_add_queue()

    def __repr__(self) -> str:
        return f"<{type(self).__qualname__}:{self.path}>"

    @property
    def name(self) -> str:
        """Construct identifier."""
        return self._name

    @property
    def parent(self) -> str:
        """Symbolic path to parent structure."""
        return self._parent

    @property
    def path(self) -> str:
        """Symbolic path to self."""
        return uris.join(self.parent, self.name)

    @abstractmethod
    def step(self) -> None:
        raise NotImplementedError()

    def _update_add_queue(self) -> None:
        try:
            lst = BUILD_LIST.get()
        except LookupError:
            pass
        else:
            lst.append(self)

    def _log_init(self, name: str) -> None:
        tname = type(self).__name__.lower()
        context = BUILD_CTX.get().rstrip(uris.SEP)
        if not context:
            logging.debug(f"Initializing {tname} '{name}'.")
        else:
            logging.debug(f"Initializing {tname} '{name}' in '{context}'.")


class Module(Construct):
    """An elementary module."""

    _constant: ClassVar[float] = 0.0

    _process: Process
    _inputs: List[Tuple[str, Callable]]
    _i_uris: Tuple[str, ...]
    _fs_uris: Tuple[str, ...]

    def __init__(
        self, 
        name: str, 
        process: Process, 
        i_uris: Sequence[str] = (),
        fs_uris: Sequence[str] = ()
    ) -> None:
        """
        Initialize a new module.
        
        :param name: Construct identifier. Must be a valid Python identifier.  
        :param process: Module process; issues updates and emits activations.
        :param i_uris: Paths to process inputs.
        :param f_uris: Paths to external feature spaces.
        """

        super().__init__(name=name)
        self._inputs = []
        self._i_uris = tuple(i_uris)
        self._fs_uris = tuple(fs_uris)
        self.process = process

    @property
    def i_uris(self) -> Tuple[str, ...]:
        """Paths to process inputs."""
        return self._i_uris

    @property
    def fs_uris(self) -> Tuple[str, ...]:
        """Paths to external feature spaces."""
        return self._fs_uris

    @property
    def process(self) -> Process:
        """Module process; issues updates and emits activations."""
        return self._process

    @process.setter
    def process(self, process: Process) -> None:
        self._process = process
        process.prefix = uris.split_head(self.path.lstrip(uris.SEP))[1]

    @property 
    def inputs(self) -> List[Tuple[str, Callable]]:
        """Mapping from input constructs to pull funcs."""
        return list(self._inputs)

    def step(self) -> None:
        try:
            self.output = self.process.call(*self._pull())
        except Exception as e:
            raise RuntimeError(f"Error in process "
            f"{type(self.process).__name__} of module '{self.path}'") from e

    @property
    def output(self) -> Union[nd.NumDict, Tuple[nd.NumDict, ...]]:
        return self._output

    @output.setter
    def output(self, output: Union[nd.NumDict, Tuple[nd.NumDict, ...]]) -> None:
        if isinstance(output, tuple):
            if any(d.c != self._constant for d in output):
                raise RuntimeError("Unexpected strength constant.") 
            for d in output:
                d.prot = True
        else:   
            if output.c != self._constant:
                raise RuntimeError("Unexpected strength constant.")
            output.prot = True
        self._output = output

    def clear_output(self) -> None:
        """Set output to initial state."""
        self.output = self.process.initial # default output

    def _view(self) -> Union[nd.NumDict, Tuple[nd.NumDict, ...]]:        
        return self.output

    def _link(self, path: str, callback: Callable) -> None:
        logging.debug(f"Connecting '{path}' to '{self.path}'")
        self._inputs.append((path, callback))

    def _pull(self) -> Tuple[nd.NumDict, ...]:
        return tuple(ask() for _, ask in self._inputs)

        
class Structure(Construct):
    """
    A composite construct.
    
    Defines constructs that may contain other constructs. 

    Complex Structure instances may be assembled using with statements.

    >>> with Structure("agent") as agent:
    ...     Construct("stimulus", Process())
    ...     with Structure("acs"):
    ...         Construct("qnet", Process(), i_uris=["../stimulus"])

    During simulation, each constituent is updated in the order that 
    it was defined.
    """

    _dict: OrderedDict[str, Construct]
    _assets: Any

    def __init__(self, name: str) -> None:
        """
        Initialize a new Structure instance.
        
        :param name: Construct identifier. Must be a valid Python identifier.
        """
        
        super().__init__(name=name)
        self._dict = OrderedDict[str, Construct]()
        self._dict_proxy = MappingProxyType(self._dict)

    def __contains__(self, key: str) -> bool:
        try:
            self.__getitem__(key)
        except KeyError:
            return False
        return True

    def __iter__(self) -> Iterator[str]:
        for construct in self._dict:
            yield construct

    def __getitem__(self, key: str) -> Any:
        if not uris.ispath(key):
            raise ValueError(f"Invalid URI '{key}'.")
        head, tail = uris.split_head(key)
        if tail:
            return self[head][tail]
        else:
            return self._dict[head]

    def __enter__(self):
        logging.debug(f"Entering context '{self.path}'")
        if 0 < len(self._dict): # This could probably be relaxed.
            raise RuntimeError("Structure already populated.")
        new_parent = uris.join(BUILD_CTX.get(), f"{self.name}{uris.SEP}")
        self._build_ctx_token = BUILD_CTX.set(new_parent)
        self._build_list_token = BUILD_LIST.set([])
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None: # Populate structure
            add_list = BUILD_LIST.get()
            self._add(*add_list)
            if self.parent == uris.SEP:
                self._weave()
        logging.debug(f"Exiting context '{self.path}'")
        BUILD_CTX.reset(self._build_ctx_token)
        BUILD_LIST.reset(self._build_list_token)

    def step(self) -> None:
        """Advance simulation by one time step."""
        for construct in self._dict.values():
            construct.step()

    def modules(self) -> Iterator[Module]:
        for construct in self._dict.values():
            if isinstance(construct, Module):
                yield construct
            else:
                assert isinstance(construct, Structure)
                for element in construct.modules():
                    yield element

    def _add(self, *constructs: Construct) -> None:
        for construct in constructs:
            logging.debug(f"Adding '{construct.name}' to '{self.path}'")
            self._dict[construct.name] = construct       

    def _weave(self) -> None:
        for module in self.modules():
            self._set_links(module)
            self._set_fspaces(module)
            module.output = module.process.initial
        for module in self.modules():
            self._validate_module(module)

    def _set_links(self, module: Module) -> None:
        for ref in module.i_uris:
            path, frag = self._parse_ref(module.path, ref)
            try:
                obj = self[path]
            except KeyError as e:
                raise KeyError(f"Module '{path}' not found.") from e
            else:
                if not isinstance(obj, Module):
                    raise TypeError(f"Expected Module instance at '{path}', "
                        f"got '{type(obj).__name__}' instead.")
                else:
                    view = partial(type(obj).output.fget, obj) # type: ignore
                    if frag:
                        module._link(ref, lambda v=view, i=int(frag): v()[i])
                    else:
                        module._link(ref, view) 

    def _set_fspaces(self, module: Module) -> None:
        getters = []
        for ref in module.fs_uris:
            path, frag = self._parse_ref(module.path, ref)
            if frag not in Process.fspace_names:
                raise ValueError(f"Unexpected fs_uri fragment '{frag}'.")
            try:
                process = self[path].process
            except KeyError as e:
                raise RuntimeError(f"Construct '{path}' not found.") from e
            else:
                getters.append(partial(getattr, process, frag))
        module.process.fspaces = tuple(getters)

    def _validate_module(self, module):
        try:
            module.process.validate()
        except Exception as e:
            raise RuntimeError(f"Validation failed for module '{module.path}'")
        for path, view in module.inputs:
            if isinstance(view(), tuple):
                raise RuntimeError(f"Missing output index in path from "
                    f"multi-output Module '{path}' to '{module.path}'.")
        
    def _parse_ref(self, parent, ref):
        uri = uris.split(uris.join(parent, ref))
        return uris.relativize(uri.path, self.path), uri.fragment
