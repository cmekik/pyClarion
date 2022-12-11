from typing import (TypeVar, Dict, List, Type, Optional, Any, Generic, 
    Generator, ChainMap, Callable, Tuple, OrderedDict, Set)
from dataclasses import dataclass, field, InitVar
from contextlib import contextmanager
import re

from ..base import symbols as sym

from .exc import CCMLException, CCMLValueError, CCMLNameError, CCMLTypeError


T = TypeVar("T", bound="Frame")
@dataclass
class Frame:
    linenos: List[int] = field(default_factory=list)
    iternos: List[int] = field(default_factory=list)
    kwds: ChainMap[str, sym.Pat] = field(default_factory=ChainMap)
    vars: ChainMap[str, str] = field(default_factory=ChainMap)
    lsts: ChainMap[str, List[str]] = field(default_factory=ChainMap)
    fs: ChainMap[str, "FSpec"] = field(default_factory=ChainMap)
    cs: Set[sym.C] = field(default_factory=set)
    rs: Set[sym.R] = field(default_factory=set)

    @property
    def lineno(self) -> int:
        return self.linenos[-1]

    @contextmanager
    def line(self, lineno:int) -> Generator[None, None, None]:
        self.linenos.append(lineno)
        yield
        self.linenos.pop()

    @contextmanager
    def iter(self, iterno: int) -> Generator[None, None, None]:
        self.iternos.append(iterno)
        yield
        self.iternos.pop()

    @contextmanager
    def scope(self) -> Generator[None, None, None]:
        self.kwds = self.kwds.new_child()
        self.vars = self.vars.new_child()
        self.lsts = self.lsts.new_child()
        self.fs = self.fs.new_child()
        yield
        self.fs = self.fs.parents
        self.lsts = self.lsts.parents
        self.vars = self.vars.parents
        self.kwds = self.kwds.parents
        
    def new_exc(self, t: Type[CCMLException], msg: str):
        return t(" ".join([msg, f"on line {self.lineno}"]))

    def sub(self, s: str, ref: sym.Pat) -> str:
        vs = r"|".join(sym.RE["var"].findall(s))
        ret = re.sub(vs, self._repl, s) if vs else s
        if sym.match(ret, ref):
            return ret
        else:
            raise self.new_exc(CCMLTypeError, f"Data '{ret}' does not match "
                f"expected type {ref}")

    def _repl(self, m: re.Match):
        try:
            return self.vars[m.group(0)]
        except KeyError:
            raise self.new_exc(CCMLNameError, f"Var '{m.group(0)}' not bound")


N, R = TypeVar("N", bound="AST"), TypeVar("R")
def line(func: Callable[[N, Frame], R]) -> Callable[[N, Frame], R]:
    def wrapper(self: N, frame: Frame) -> R:
        with frame.line(self.lineno):
            return func(self, frame)
    wrapper.__qualname__ = func.__qualname__
    return wrapper

def scope(func: Callable[[N, Frame], R]) -> Callable[[N, Frame], R]:
    def wrapper(self: N, frame: Frame) -> R:
        with frame.scope():
            return func(self, frame)
    wrapper.__qualname__ = func.__qualname__
    return wrapper


S = TypeVar("S", bound="Spec")
@dataclass
class Spec:
    data: OrderedDict = field(default_factory=OrderedDict)
    def add(self: S, other: "S", frame: "Frame") -> None:
        if set(self.data).intersection(other.data):
            raise frame.new_exc(CCMLValueError, "Duplicated specs")
        self.data.update(other.data)

@dataclass
class FSpec(Spec):
    data: OrderedDict[sym.D, Tuple[List[sym.F], Dict[str, float]]] \
        = field(default_factory=OrderedDict)

@dataclass
class CSpec(Spec):
    data: OrderedDict[sym.C, Tuple[FSpec, Dict[str, float]]] \
        = field(default_factory=OrderedDict)

@dataclass
class RSpec(Spec):
    data: OrderedDict[sym.R, Tuple[CSpec, Dict[str, float]]] \
        = field(default_factory=OrderedDict)

class TSpec(List[str]):
    pass


T = TypeVar("T", bound="AST")
R = TypeVar("R")
@dataclass
class AST(Generic[R]):
    lineno: int
    _args: InitVar[str]
    _kwds: InitVar[str]
    args: List[str] = field(init=False)
    kwds: Dict[str, str] = field(init=False)
    parent: Optional["AST"] = field(init=False, default=None)
    children: List["AST"] = field(init=False, default_factory=list)

    def __post_init__(self, _args, _kwds) -> None:
        self.args = _args.split()
        self.kwds = dict(tuple(kv.split("=")) for kv in _kwds.split())
        self.check_signature()

    def spawn(self, t: Type[T], lineno: int, *data: Any) -> T:
        child = t(lineno, *data)
        self.children.append(child)
        child.parent = self
        return child

    def check_signature(self) -> None:
        ...

    def exec(self, frame: Frame) -> R:
        ...

    @classmethod
    def discover(cls) -> List[Type["AST"]]:
        lst = []
        for _cls in cls.__subclasses__():
            if not _cls.__name__.startswith("_"):
                lst.append(_cls)
            lst.extend(_cls.discover())
        return lst
