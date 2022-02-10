from __future__ import annotations

from typing import Tuple, List, TypeVar, Sequence

from ..base import Process, feature
from .wm import Flags
from .utils import expand_dim
from .. import numdicts as nd


__all__ = ["Gates", "DimFilter"]


T = TypeVar("T")


class Gates(Process):
    """Selectively gates inputs."""

    def __init__(self, fs: Sequence[str]) -> None:
        self._flags = Flags(fs=fs, vs=(0, 1))

    def call(
        self, c: nd.NumDict[feature], *inputs: nd.NumDict
    ) -> Tuple[nd.NumDict, ...]:
        """Gate inputs, then update gate settings according to c."""

        gs = [self.store.isolate(key=k) for k in self.flags]
        self.update(c) 
        return (self.store, *(x.mul(g) for g, x in zip(gs, inputs)))

    def update(self, c):
        self._flags.update(c)

    @property
    def prefix(self) -> str:
        return self._flags.prefix

    @prefix.setter
    def prefix(self, val: str) -> None:
        self._flags.prefix = val

    @property
    def fs(self) -> Tuple[str, ...]:
        return self._flags.fs

    @fs.setter
    def fs(self, val: Sequence[str]) -> None:
        self._flags.fs = tuple(val)

    @property
    def store(self) -> nd.NumDict[feature]:
        return self._flags.store

    @property
    def initial(self):
        return tuple(nd.NumDict() for _ in range(len(self.fs) + 1))    

    @property
    def flags(self) -> Tuple[feature, ...]:
        return self._flags.flags

    @property
    def cmds(self) -> Tuple[feature, ...]:
        return self._flags.cmds

    @property
    def nops(self) -> Tuple[feature, ...]:
        return self._flags.nops


class DimFilter(Process):
    """Selectively filters dimensions."""

    initial = (nd.NumDict(), nd.NumDict())

    def __init__(self) -> None:
        self._flags = Flags(fs=(), vs=(0, 1))

    def call(
        self, c: nd.NumDict[feature], d: nd.NumDict[feature]
    ) -> Tuple[nd.NumDict[feature], nd.NumDict[feature]]:
        
        store = self.store
        self.update(c)
        return store, d.mul_from(store, kf=self._feature2flag)

    def _feature2flag(self, f):
        return feature(expand_dim(f.d.replace("#", "."), self.prefix))

    def update(self, c):
        self._flags.fs = tuple(f.d.replace("#", ".") 
            for fspace in self.fspaces for f in fspace()) 
        self._flags.update(c)

    def validate(self):
        self.update(nd.NumDict())
        self._flags.validate()

    @property
    def prefix(self) -> str:
        return self._flags.prefix

    @prefix.setter
    def prefix(self, val: str) -> None:
        self._flags.prefix = val

    @property
    def fs(self) -> Tuple[str, ...]:
        return self._flags.fs

    @fs.setter
    def fs(self, val: Sequence[str]) -> None:
        self._flags.fs = tuple(val)

    @property
    def store(self) -> nd.NumDict[feature]:
        return self._flags.store

    @property
    def flags(self) -> Tuple[feature, ...]:
        return self._flags.flags

    @property
    def cmds(self) -> Tuple[feature, ...]:
        return self._flags.cmds

    @property
    def nops(self) -> Tuple[feature, ...]:
        return self._flags.nops
