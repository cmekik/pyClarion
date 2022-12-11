from typing import Tuple, List, Dict, Callable, Generic, TypeVar, Union, Optional
from contextvars import ContextVar
from contextlib import contextmanager

from ..numdicts import NumDict
from . import symbols as sym

BUILD_LIST: ContextVar = ContextVar("BUILD_LIST")


def _append_to_build_list(process: "Process") -> None:
    try:
        processes = BUILD_LIST.get()
    except LookupError:
        pass
    else:
        processes.append(process)


@contextmanager
def subprocesses():
    token = BUILD_LIST.set([])
    yield
    BUILD_LIST.reset(token)


T = TypeVar("T")
class Process(Generic[T]):
    """Base class for simulated processes."""
    path: str
    inputs: List[str]
    initial: Callable[[], T]
    call: Callable[..., T]

    def __init__(
        self, path: str = "", inputs: Optional[List[str]] = None
    ) -> None:
        """
        Initialize a new Process instance.
        
        :param path: Address of self within simulated agent.
        :param inputs: Addresses afferent inputs.
        """
        self.path = path
        self.inputs = inputs or []
        self.__validate()
        _append_to_build_list(self)

    def __validate(self) -> None:
        if self.path and not sym.match(self.path, "adr"):
            raise ValueError(f"Value '{self.path}' is not a valid path.")
        for i, loc in enumerate(self.inputs):
            if not sym.match(loc, "loc"):
                raise ValueError(f"Value '{loc}' at index {i} of inputs to "
                    f"{self.path} is not a valid output site locator.")


class Agent:
    """
    A pyClarion agent.
    
    Agents may be assembled using with statements.

    >>> with Agent() as agent:
    ...     Process("percep", [])
    ...     Process("acs/bi", ["percep"])
    """
    processes: List[Process]
    state: Dict[str, NumDict]

    def __init__(self):
        self.processes = []
        self.state = {}

    def __enter__(self):
        try:
            BUILD_LIST.get()
        except LookupError:
            self._build_list_token = BUILD_LIST.set(self.processes)
        else:
            raise RuntimeError("Agents cannot be nested.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self._initialize()
        BUILD_LIST.reset(self._build_list_token)

    def step(self):
        """Advance simulation by one time step."""
        for p in self.processes:
            self._update(p.path, p.call(*(self.state[a] for a in p.inputs)))

    def _initialize(self):
        for p in self.processes:
            self._update(p.path, p.initial())

    def _update(
        self, path: str, output: Union[NumDict, Tuple[NumDict, ...]]
    ) -> None:
        if isinstance(output, tuple):
            for i, d in enumerate(output):
                self.state[sym.make_loc(i, path)] = d
        else:
            assert isinstance(output, NumDict)
            self.state[path] = output

    def _validate_network(self):
        addresses = set()
        for i, p in enumerate(self.processes):
            if not sym.match(p.path, "adr"):
                raise RuntimeError(f"Invalid path '{p.path}' for process {i}.")
            elif p.path in addresses:
                raise RuntimeError(f"Duplicate address: {p.path}.")
            else:
                addresses.add(p.path)
        for p in self.processes:
            for path in p.inputs:
                if path not in addresses:
                    raise RuntimeError(f"Undefined input {path} to "
                        f"process {p.__class__.__name__} at {p.path}")
