from typing import Literal
from dataclasses import dataclass, field

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
        raise NotImplementedError()

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

    def key(self) -> type:
        return type(self)

    def target(self) -> "State":
        return self.state


@dataclass(slots=True)
class ForwardUpdate(StateUpdate):
    def __post_init__(self) -> None:
        self.grad = False


@dataclass(slots=True)
class BackwardUpdate(StateUpdate):
    def __post_init__(self) -> None:
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