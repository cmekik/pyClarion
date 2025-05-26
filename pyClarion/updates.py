from typing import Literal
from dataclasses import dataclass, field
from math import isnan

from .system import Update, State
from .knowledge import Sort, Term, Atom, Chunk, Rule
from .numdicts import Key, NumDict


@dataclass(slots=True)
class StateUpdate(Update[State]):
    state: "State"
    data: NumDict | dict[Key, float]
    method: Literal["push", "add", "write"] = "push"
    grad: bool = field(init=False) 

    def __post_init__(self) -> None:
        data = self.data
        state = self.state
        if isinstance(data, NumDict) and data.i != state.index:
            raise ValueError("Index of data numdict does not match site")
        if isinstance(data, NumDict) and \
            not (isnan(data.c) and isnan(state.const) or data.c == state.const):
            raise ValueError(f"Default constant {data.c} of data does not "
                f"match site {state.const}")

    def apply(self) -> None:
        data = self.data
        if not isinstance(data, NumDict):
                data = self.state.new(data)
        channel = self.state.grad if self.grad else self.state.data
        match self.method:
            case "push":
                channel.appendleft(data)
            case "add":
                channel[0] = channel[0].sum(data)
            case "write":
                with channel[0].mutable():
                    channel[0].update(data.d)
            case _:
                assert False

    def target(self) -> "State":
        return self.state


class ForwardUpdate(StateUpdate):
    __slots__ = ()
    def __post_init__(self) -> None:
        super().__post_init__()
        self.grad = False


class BackwardUpdate(StateUpdate):
    __slots__ = ()
    def __post_init__(self) -> None:
        super().__post_init__()
        self.grad = True


@dataclass(slots=True)
class SortUpdate[T: Term](Update[Sort[T]]):
    sort: Sort[T]
    add: tuple[T, ...] = ()
    remove: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.add or self.remove)

    def apply(self) -> None:
        for term in self.add:
            try:
                self.sort[term._name_] = term
            except AttributeError:
                self.sort[f"_{next(self.sort._counter_)}"] = term
        for name in self.remove:
            self.sort[name]

    def target(self) -> Sort[T]:
        return self.sort
    

class AtomUpdate(SortUpdate[Atom]):
    __slots__ = ()
    

class ChunkUpdate(SortUpdate[Chunk]):
    __slots__ = ()


class RuleUpdate(SortUpdate[Rule]):
    __slots__ = ()