from typing import overload, Callable
from datetime import timedelta
from math import exp

from .base import Component, Parametric, Stateful, Priority
from ..events import Event, State, Site, ForwardUpdate
from ..knowledge import (Family, Term, Atoms, Atom, Chunk, Bus, Nodes)
from ..numdicts import NumDict, Key, KeyForm, numdict, keyform


class Input[D: Nodes](Component):
    """
    An input receiver process.
    
    Receives activations from external sources.
    """

    d: D
    main: Site = Site()
    reset: bool

    def __init__(self, 
        name: str, 
        d: D,
        *,
        c: float = 0.0,
        reset: bool = True,
        l: int = 1
    ) -> None:
        super().__init__(name)
        index, = self._init_indexes(d)
        self.d = d 
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
        method = "push" if self.reset else "write"
        return Event(self.send, 
            [ForwardUpdate(self.main, data, method)], 
            dt, priority)

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
        elif isinstance(d, Chunk):
            for (t1, t2), weight in d._dyads_.items():
                if not isinstance(t1, Term) or not isinstance(t2, Term):
                    raise TypeError("Input chunk may not contain variables.")
                key = ~t1 * ~t2
                if key not in self.main.index:
                    raise ValueError(f"Unexpected dimension-value pair {key}")
                data[key] = weight
        else:
            raise TypeError(f"Unexpected input of type '{type(d).__name__}'")
        return data


class Choice[D: Nodes](Stateful, Parametric):
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
    d: D
    by: KeyForm
    main: Site = Site()
    input: Site = Site(lax=True)
    sample: Site = Site()
    params: Site = Site()
    state: Site = Site()

    def __init__(self, 
        name: str, 
        p: Family,
        s: Family, 
        d: D, 
        *, 
        sd: float = 1.0,
        f: float = 1.0,
        l: int = 1
    ) -> None:
        super().__init__(name)
        self.system.check_root(p)
        index, = self._init_indexes(d)
        self.p, self.params = self._init_sort(p, type(self).Params, sd=sd, f=f)
        self.s, self.state = self._init_sort(s, type(self).State, c=0., free=1.)
        self.d = d
        self.main = State(index, {}, 0.0, l=l)
        self.input = State(index, {}, 0.0, l=l)
        self.sample = State(index, {}, 0.0, l=l)
        self.by = self._init_by(d)

    @staticmethod
    def _init_by(s: D) -> KeyForm:
        match s:
            case tuple():
                (d, v) = s
                return keyform(d) * keyform(v, -1)
            case s:
                return keyform(s, -1)
    
    @property
    def locked(self) -> bool:
        return self.current_state == ~self.s.busy \
            or any(event.source == self.trigger for event in self.system.queue)

    def poll(self) -> dict[Key, Key]:
        """Return a symbolic representation of current decision."""
        return self.main[0].argmax(by=self.by)

    def resolve(self, event: Event) -> None:
        if event.source == self.trigger:
            self.system.schedule(self.select())
            
    def trigger(self, 
        dt: timedelta = timedelta(), 
        priority=Priority.DEFERRED
    ) -> Event:
        """
        Generate a dummy event to trigger selection of a new choice.
        
        Raises a RuntimeError if called when self.locked evaluates to True.
        """
        if self.locked:
            raise RuntimeError(f"Process {self.name} already triggered.")
        return Event(self.trigger, 
            [ForwardUpdate(self.state, {~self.s.busy: 1.0})], 
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
            [ForwardUpdate(self.main, {v: 1.0 for v in choices.values()}),
             ForwardUpdate(self.sample, sample),
             ForwardUpdate(self.state, {~self.s.free: 1.0})],
            time=dt, priority=priority)
    

class Discriminal[D: Nodes](Parametric):
    """
    A discriminal process.
    
    Makes discrete stochastic decisions based on activation strengths.
    """

    class Params(Atoms):
        sd: Atom
        th: Atom

    p: Params
    d: D
    main: Site = Site()
    input: Site = Site(lax=True)
    sample: Site = Site()
    params: Site = Site()

    def __init__(self, 
        name: str, 
        p: Family,
        d: D, 
        *, 
        sd: float = 1e-1,
        th: float = .32905, # critical value for two-tailed alpha = 1e-3
        l: int = 1
    ) -> None:
        super().__init__(name)
        self.system.check_root(p)
        index, = self._init_indexes(d)
        self.p, self.params = self._init_sort(p, type(self).Params, sd=sd, th=th)
        self.d = d
        self.main = State(index, {}, 0.0, l=l)
        self.input = State(index, {}, 0.0, l=l)
        self.sample = State(index, {}, 0.0, l=l)

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
        th = self.params[0][~self.p.th]
        pos = sample.isbetween(lb=th)
        neg = sample.isbetween(ub=-th)
        choices = pos.sub(neg)
        return Event(self.select,
            [ForwardUpdate(self.main, choices),
             ForwardUpdate(self.sample, sample)],
            time=dt, priority=priority)


class Controller(Component):
    
    input: Site = Site(lax=True)
    callbacks: dict[Key, Callable[[], Event]]
    _const: NumDict

    def __init__(self, 
        name: str, 
        b: Bus, 
        v: Atoms, 
        **kwargs: Callable[[], Event]
    ) -> None:
        for atom in v:
            if atom != "nil" and atom not in kwargs:
                raise ValueError(f"Unbound action key '{atom}'")
        super().__init__(name)
        self.system.check_root(b, v)
        idx_b, idx_v = self._init_indexes(b, v)
        self.input = State(idx_b * idx_v, {}, 0.0)
        self._const = numdict(idx_b * idx_v, {}, 0.0)
        self.callbacks = {~b * ~v[kwd]: cb for kwd, cb in kwargs.items()}

    def resolve(self, event: Event) -> None:
        forward = event.index(ForwardUpdate)
        if self.input in forward:
            action = self.callbacks.get(self.poll(), None)
            if action is not None: 
                self.system.schedule(action())

    def poll(self) -> Key:
        return self._const.sum(self.input[0]).argmax()
