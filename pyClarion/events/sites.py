from typing import Iterator
from math import isnan
from collections import deque

from .system import Process
from ..numdicts import Index, NumDict, numdict


class State:
    """A simulated process state."""

    index: Index
    const: float
    data: deque[NumDict]
    grad: deque[NumDict]

    def __init__(self, i: Index, d: dict, c: float, l: int = 1) -> None:
        l = 1 if l < 1 else l
        self.index = i
        self.const = c
        self.data = deque([numdict(i, d, c) for _ in range(l)], maxlen=l)
        self.grad = deque([numdict(i, {}, 0.0) for _ in range(l)], maxlen=l)

    def __iter__(self) -> Iterator[NumDict]:
        yield from self.data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i: int) -> NumDict:
        return self.data[i]
    
    def new(self, d: dict, c: float | None = None) -> NumDict:
        return numdict(self.index, d, self.const if c is None else c)


class Site:
    def __init__(self, lax: bool = False) -> None:
        self.lax = lax

    def __set_name__(self, owner: Process, name: str) -> None:
        self._name = "_" + name

    def __get__(self, obj: Process, objtype: type[Process] | None = None) \
        -> State:
        return getattr(obj, self._name)

    def __set__(self, obj: Process, value: State) -> None:
        self.validate(obj, value)
        setattr(obj, self._name, value)

    def validate(self, obj: Process, value: State) -> None:
        old = getattr(obj, self._name, None)
        if old is None:
            pass
        elif not isinstance(value, State):
            raise TypeError("Process site assigned object of wrong type")
        elif any(d.d for d in old.data) or any(d.d for d in old.grads):
            raise ValueError(f"Site '{self._name}' of process {obj.name} "
                "contains data")
        elif old.index != value.index and not self.lax \
            or old.index < value.index:
            raise ValueError("Incompatible index in site assignment")
        elif not (isnan(old.const) and isnan(value.const) \
            or old.const == value.const):
            raise ValueError("Incompatible default value in site assignment")
        elif len(old.data) != len(value.data) and not self.lax:
            raise ValueError("Incompatible lag values")
