from typing import Literal
from dataclasses import dataclass
from collections import deque
from math import isnan

from .system import Update
from .sites import State
from ..numdicts import Key, NumDict
from ..numdicts.keyspaces import KSParent, KSChild


@dataclass(slots=True)
class StateUpdate(Update[State]):
    state: "State"
    data: NumDict | dict[Key, float]
    method: Literal["push", "add", "write"] = "push"

    def __post_init__(self) -> None:
        data = self.data
        state = self.state
        if isinstance(data, NumDict) and data.i != state.index:
            raise ValueError("Index of data numdict does not match site")
        if isinstance(data, NumDict) and data.c != state.const:
            raise ValueError(f"Default constant {data.c} of data does not "
                f"match site {state.const}")
        if isinstance(data, NumDict):
            self.data = data.d

    def apply(self) -> None:
        data = self.data
        assert isinstance(data, dict)
        channel = self._get_channel()
        if self.method == "write":
            with channel[0].mutable():
                channel[0].update(data)
            return
        data = self.state.new(data)
        match self.method:
            case "push":
                channel.appendleft(data)
            case "add":
                channel[0] = channel[0].sum(data)
            case _:
                assert False

    def _get_channel(self) -> deque[NumDict]:
        raise NotImplementedError()

    @property
    def target(self) -> "State":
        return self.state


class ForwardUpdate(StateUpdate):
    __slots__ = ()
    def __post_init__(self) -> None:
        super().__post_init__()
    def _get_channel(self) -> deque[NumDict]:
        return self.state.data


class BackwardUpdate(StateUpdate):
    __slots__ = ()
    def __post_init__(self) -> None:
        super().__post_init__()
    def _get_channel(self) -> deque[NumDict]:
        return self.state.grad


@dataclass(slots=True)
class KeyspaceUpdate[P: KSParent, C: KSChild](Update[P]):
    node: P
    add: tuple[C, ...] = ()
    remove: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.add or self.remove)

    def apply(self) -> None:
        for child in self.add:
            try:
                self.node[child._name_] = child
            except AttributeError:
                # attempt automatic name generation
                self.node[None] = child
        for name in self.remove:
            del self.node[name]

    @property
    def target(self) -> KSParent:
        return self.node
