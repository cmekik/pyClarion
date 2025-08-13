from typing import Self, Sequence, Callable, Any
from datetime import timedelta
from collections import deque
from enum import IntEnum

from ..events import Process, Site, State, Event, ForwardUpdate, KeyspaceUpdate
from ..knowledge import Root, Family, Sort, Term, Atoms, Atom, Chunks, Chunk, Rules, Rule
from ..numdicts import Index, keyform, Key, NumDict
from ..numdicts.ops.tape import GradientTape


type D = Family | Sort | Term
type V = Family | Sort
type DV = tuple[D, V]


class Priority(IntEnum):
    """
    An event priority enum.
    
    Used to indicate event priority in case two events are scheduled at the 
    same time within a simulation. 
    """
    MAX = 128
    PARAM = 120
    CHOICE = 112
    LEARNING = 96
    PROPAGATION = 64
    DEFERRED = 32
    MIN = 0


class AtomUpdate(KeyspaceUpdate[Atoms, Atom]):
    __slots__ = ()


class ChunkUpdate(KeyspaceUpdate[Chunks, Chunk]):
    __slots__ = ()


class RuleUpdate(KeyspaceUpdate[Rules, Rule]):
    __slots__ = ()


class Component(Process[Root]):
    
    def __init__(self, name: str) -> None:
        super().__init__(name, Root())

    def __rshift__[T](self: Self, other: T) -> T:
        input = getattr(other, "input", None)
        main = getattr(self, "main", None)
        if isinstance(input, State) and isinstance(main, State) :
            setattr(other, "input", main)
            return other
        return NotImplemented
     
    def __rrshift__(self: Self, other: Any) -> Self:
        input = getattr(self, "input", None)
        if isinstance(input, State):
            if isinstance(other, State):
                setattr(self, "input", other)
                return self
            elif isinstance(other, Component):
                main = getattr(other, "main", None)
                if isinstance(main, State):
                    setattr(self, "input", main)
                    return self
        return NotImplemented

    def _init_indexes(self, *keyspaces: D | V | DV) -> Sequence[Index]:
        indices = []
        for item in keyspaces:
            match item:
                case (d, v):
                    self.system.check_root(d, v)
                    idx_d = self.system.get_index(keyform(d))
                    idx_v = self.system.get_index(keyform(v))
                    indices.append(idx_d * idx_v)
                case d:
                    self.system.check_root(d)
                    idx_d = self.system.get_index(keyform(d))
                    indices.append(idx_d)
        return indices

    def _init_sort[S: Sort](self, 
        family: Family, 
        sort_cls: type[S],
        c: float = float("nan"), 
        l: int = 1,
        **params: float
    ) -> tuple[S, State]:
        self.system.check_root(family)
        sort = sort_cls(); family[self.name] = sort
        site = State(
            i=self.system.get_index(keyform(sort)), 
            d={~sort[k]: v for k, v in params.items()}, 
            c=c,
            l=l)
        return sort, site
    

class Parametric[P: Atoms](Component):
    p: P
    params: Site = Site()

    def set_params(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PARAM,
        **kwargs: float
    ) -> Event:
        data = {~self.p[param]: value for param, value in kwargs.items()}
        return Event(self.set_params, 
            [ForwardUpdate(self.params, data, "write")], 
            dt, priority)
    

class Stateful[S: Atoms](Component):
    s: S
    state: Site = Site()

    @property
    def current_state(self) -> Key:
        return self.state[0].argmax()
    

class Backpropagator(Component):
    tapes: deque[tuple[GradientTape[NumDict], NumDict, list[NumDict]]]
    forward: Callable[..., Event]
    backward: Callable[..., Event]

    def push_tape(self, 
        tape: GradientTape, 
        main: NumDict, 
        args: list[NumDict]
    ) -> None:
        self.tapes.appendleft((tape, main, args))
