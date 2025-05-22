from typing import Self, Sequence, Callable, Any
from datetime import timedelta
from collections import deque
from dataclasses import dataclass

from ..system import Process, Site, Priority, Event, Update
from ..knowledge import Family, Sort, Term, Atoms
from ..numdicts import Index, keyform, Key, NumDict
from ..numdicts.ops.tape import GradientTape


type D = Family | Sort | Term
type V = Family | Sort
type DV = tuple[D, V]


class Component(Process):
    
    def __rshift__[T](self: Self, other: T) -> T:
        input = getattr(other, "input", None)
        main = getattr(self, "main", None)
        if isinstance(input, Site) and isinstance(main, Site) :
            setattr(other, "input", main)
            return other
        return NotImplemented
     
    def __rrshift__(self: Self, other: Any) -> Self:
        input = getattr(self, "input", None)
        if isinstance(input, Site):
            if isinstance(other, Site):
                setattr(self, "input", other)
                return self
            elif isinstance(other, Component):
                main = getattr(other, "main", None)
                if isinstance(main, Site):
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
    ) -> tuple[S, Site]:
        self.system.check_root(family)
        sort = sort_cls(); family[self.name] = sort
        site = Site(
            i=self.system.get_index(keyform(sort)), 
            d={~sort[k]: v for k, v in params.items()}, 
            c=c,
            l=l)
        return sort, site
    

class Parametric[P: Atoms](Component):
    p: P
    params: Site

    def set_params(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PARAM,
        **kwargs: float
    ) -> Event:
        data = {~self.p[param]: value for param, value in kwargs.items()}
        return Event(self.set_params, 
            [self.params.update(data, Site.write_inplace)], 
            dt, priority)
    

class Stateful[S: Atoms](Component):
    s: S
    state: Site

    @property
    def current_state(self) -> Key:
        return self.state[0].argmax()
    

class Backpropagator(Component):
    @dataclass(slots=True)
    class UpdateTapes(Update):
        proc: "Backpropagator"
        tape: GradientTape
        main: NumDict
        args: list[NumDict]

        def apply(self) -> None:
            self.proc.tapes.appendleft((self.tape, self.main, self.args))

    tapes: deque[tuple[GradientTape[NumDict], NumDict, list[NumDict]]]
    forward: Callable[..., Event]
    backward: Callable[..., Event]

    def push_tape(self, 
        tape: GradientTape, 
        main: NumDict, 
        args: list[NumDict]
    ) -> Update:
        return self.UpdateTapes(self, tape, main, args)