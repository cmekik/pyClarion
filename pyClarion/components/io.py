from typing import ClassVar, overload
from datetime import timedelta
from math import exp

from .base import V, DV, Component, Parametric, Stateful
from ..system import Event, Priority, State, Site
from ..knowledge import (Family, Term, Atoms, Atom, Chunk, Var)
from ..numdicts import Key, KeyForm, numdict, keyform


class Input(Component):
    """
    An input receiver process.
    
    Receives activations from external sources.
    """

    main: Site = Site()
    reset: bool

    def __init__(self, 
        name: str, 
        s: V | DV,
        *,
        c: float = 0.0,
        reset: bool = True,
        l: int = 1
    ) -> None:
        super().__init__(name)
        index, = self._init_indexes(s) 
        self.main = State(index, {}, c, l)
        self.reset = reset

    @overload
    def send(self, d: dict[Term, float], 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> Event:
        ...

    @overload
    def send(self, d: dict[Key, float], 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> Event:
        ...

    @overload
    def send(self, d: dict[Key | Term, float], 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> Event:
        ...

    @overload
    def send(self, d: Chunk, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> Event:
        ...

    def send(self, d: dict | Chunk, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> Event:
        """Update input data."""
        data = self._parse_input(d)
        method = State.push if self.reset else State.write_inplace
        return Event(self.send, [self.main.update(data, method)], dt, priority)

    def _parse_input(self, d: dict | Chunk) -> dict[Key, float]:
        data = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(k, Term):
                    self.system.check_root(k)
                    k =  ~k
                if k not in self.main.index:
                    raise ValueError(f"Unexpected key {k}")
                data[k] = v
        if isinstance(d, Chunk):
            for (t1, t2), weight in d._dyads_.items():
                if isinstance(t1, Var) or isinstance(t2, Var):
                    raise TypeError("Var not allowed in input chunk.")
                key = ~t1 * ~t2
                if key not in self.main.index:
                    raise ValueError(f"Unexpected dimension-value pair {key}")
                data[key] = weight
        return data


class Choice(Stateful, Parametric):
    """
    A choice process.
    
    Makes discrete stochastic decisions based on activation strengths.
    """

    class Params(Atoms):
        sd: Atom
        f: Atom

    class State(Atoms):
        free: Atom
        busy: Atom

    p: Params
    s: State
    by: KeyForm
    main: Site = Site()
    input: Site = Site(lax=True)
    sample: Site = Site()
    params: Site = Site()
    state: Site = Site()
    locked: bool

    def __init__(self, 
        name: str, 
        p: Family,
        s_: Family, 
        s: V | DV, 
        *, 
        sd: float = 1.0,
        f: float = 1.0,
        l: int = 1
    ) -> None:
        super().__init__(name)
        self.system.check_root(p)
        index, = self._init_indexes(s)
        self.p, self.params = self._init_sort(p, type(self).Params, sd=sd, f=f)
        self.s, self.state = self._init_sort(s_, type(self).State, c=0., free=1.)
        self.main = State(index, {}, 0.0, l=l)
        self.input = State(index, {}, 0.0, l=l)
        self.sample = State(index, {}, 0.0, l=l)
        self.by = self._init_by(s)
        self.locked = False

    @staticmethod
    def _init_by(s: V | DV) -> KeyForm:
        match s:
            case (d, v):
                return keyform(d) * keyform(v, -1)
            case s:
                return keyform(s, -1)

    def poll(self) -> dict[Key, Key]:
        """Return a symbolic representation of current decision."""
        return self.main[0].argmax(by=self.by)

    def resolve(self, event: Event) -> None:
        if event.source == self.trigger and not self.locked:
            self.locked = True
            self.system.schedule(self.select())
        if event.source == self.select:
            self.locked = False

    def trigger(self, 
        dt: timedelta = timedelta(), 
        priority=Priority.DEFERRED
    ) -> Event:
        """Generate a dummy event to trigger selection of a new choice."""
        return Event(self.trigger, 
            [self.state.update({~self.s.busy: 1.0})], 
            dt, priority)

    def select(self, 
        dt: timedelta = timedelta(), 
        priority=Priority.CHOICE
    ) -> Event:
        """
        Generate a selection event. 

        Makes a stochastic choice based on the immediate state of the system and 
        generates an event to update relevant sites.  
    
        Direct use of this method is discouraged in favor of Choice.trigger().
        """
        input = self.main.new({}).sum(self.input[0])
        sd = numdict(self.main.index, {}, c=self.params[0][~self.p.sd])
        sample = input.normalvariate(sd)
        choices = sample.argmax(by=self.by)
        f = self.params[0][~self.p.f]
        if f > 0 and not dt:
            dt = timedelta(seconds=f * exp(-sample.valmax()))
        return Event(self.select,
            [self.main.update({v: 1.0 for v in choices.values()}),
             self.sample.update(sample),
             self.state.update({~self.s.free: 1.0})],
            time=dt, priority=priority)

