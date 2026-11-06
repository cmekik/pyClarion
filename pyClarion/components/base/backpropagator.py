from typing import Final
from collections import deque
from datetime import timedelta
from dataclasses import dataclass

from ...events.system import Process, Action, Fluent, Update
from ...knowledge import Root
from ...numdicts import NumDict
from ...numdicts.ops.tape import GradientTape


class Backpropagator(Process[Root]):
    tapes: Final[deque[tuple[GradientTape[NumDict], NumDict, list[NumDict]]]]

    def __init__(self, name: str, root: Root | None = None) -> None:
        super().__init__(name, root)
        self.tapes = deque([], maxlen=self.system.depth)


@dataclass(slots=True)
class PushTape[P: Backpropagator](Update[P]):
    process: P
    tape: GradientTape
    output: NumDict
    args: list[NumDict]

    def apply(self) -> None:
        self.process.tapes.appendleft((self.tape, self.output, self.args))

    @property
    def target(self) -> P:
        return self.process


class Backpropagation[P: Backpropagator](Action[P]):
    __match_args__ = ("source", "index")
    __slots__ = ("index",)

    index: Final[int]

    def __init__(self, source: P, index: int = -1) -> None:
        super().__init__(source)
        self.index = index

