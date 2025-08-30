from datetime import timedelta
import math

from .base import Parametric, AtomUpdate, ChunkUpdate, RuleUpdate, Priority
from ..events import Event, State, Site, ForwardUpdate
from ..knowledge import Family, Atoms, Chunks, Rules, Chunk, Rule, Atom
from ..numdicts import Key, keyform


class BaseLevel[D: Atoms | Chunks | Rules](Parametric):
    """
    A base-level activation process.

    Maintains and propagates base level activations.
    """

    _ud_type = {Atom: AtomUpdate, Chunk: ChunkUpdate, Rule: RuleUpdate}

    class Params(Atoms):
        th: Atom
        sc: Atom
        de: Atom
    
    e: Atoms
    p: Params
    d: D
    unit: timedelta
    ignore: set[Key]
    main: Site = Site()
    input: Site = Site()
    times: Site = Site()
    decay: Site = Site()
    scale: Site = Site()
    weights: Site = Site()
    params: Site = Site()
    
    def __init__(self, 
        name: str, 
        p: Family, 
        e: Family, 
        d: D, 
        *, 
        unit: timedelta = timedelta(milliseconds=1),
        th: float = 0.0, 
        sc: float = 1.0, 
        de: float = 0.5
    ) -> None:
        super().__init__(name)
        self.system.check_root(p, e, d)
        if e == p:
            raise ValueError("Args p and e must be distinct")
        self.p = type(self).Params(); p[name] = self.p
        self.e = Atoms(); e[name] = self.e
        self.d = d
        idx_p = self.system.get_index(keyform(self.p))
        idx_e = self.system.get_index(keyform(self.e))
        idx_d = self.system.get_index(keyform(d))
        self.unit = unit
        self.ignore = set()
        self.main = State(idx_d, {}, 1.0)
        self.input = State(idx_d, {}, 0.0)
        self.times = State(idx_e, {}, float("nan"))
        self.decay = State(idx_e, {}, float("nan"))
        self.scale = State(idx_e, {}, float("nan"))
        self.weights = State(idx_e * idx_d, {}, 0.0)
        self.params = State(idx_p, 
            {~self.p.th: th, ~self.p.sc: sc, ~self.p.de: de}, 
            float("nan"))

    def resolve(self, event: Event) -> None:
        state_updates = event.index(ForwardUpdate).get(self.input, [])
        sort_updates = event.index(self._ud_type[self.d]).get(self.d, [])
        invoked = set()
        if state_updates:
            th = self.params[0][~self.p.th]
            invoked.update((k for ud in state_updates for k in ud.data 
                if k not in self.ignore and th < ud.data[k]))
        if sort_updates:
            invoked.update((~k for ud in sort_updates for k in ud.add 
                if k not in self.ignore))
        if invoked:
            self.invoke(invoked)

    def invoke(self,
        invoked: set[Key], 
        dt: timedelta = timedelta(), 
        priority: int = Priority.LEARNING
    ) -> Event:
        ke = ~self.e; name = f"e{next(self.e._counter_)}"
        key = ke.link(Key(name), ke.size)
        time = self.system.clock.time / self.unit
        sc = self.params[0][~self.p.sc] 
        de = self.params[0][~self.p.de]
        atom = Atom()
        atom._name_ = name
        return Event(self.invoke, 
            [AtomUpdate(self.e, add=(atom,)),
            ForwardUpdate(self.times, {key: time}, "write"),
            ForwardUpdate(self.scale, {key: sc}, "write"),
            ForwardUpdate(self.decay, {key: de}, "write"),
            ForwardUpdate(self.weights, {ke * k: 1.0 for k in invoked}, "write")],
            dt, priority)

    def advance(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.LEARNING
    ) -> Event:
        time = self.system.clock.time / self.unit
        terms = (self.times[0]
            .neg()
            .shift(time)
            .div(self.scale[0])
            .log()
            .mul(self.decay[0].neg())
            .exp())
        blas = (self.weights[0]
            .mul(terms)
            .sum(by=self.main.index.kf, c=0.0))
        with blas.mutable():
            for k in self.ignore:
                blas[k] = 1.0
        return Event(self.advance, [ForwardUpdate(self.main, blas)], dt, priority)


class MatchStats[D: Atoms | Chunks | Rules](Parametric):
    """A process that maintains match statistics."""
    
    class Params(Atoms):
        c1: Atom
        c2: Atom
        discount: Atom
        th_cond: Atom
        th_crit: Atom

    p: Params
    d: D
    main: Site = Site()
    posm: Site = Site()
    negm: Site = Site()
    cond: Site = Site()
    crit: Site = Site()
    params: Site = Site()

    def __init__(self,
        name: str, 
        p: Family, 
        d: D, 
        *, 
        c1=1.0, 
        c2=2.0, 
        discount=.9, 
        th_cond=0.0, 
        th_crit=0.0
    ) -> None:
        super().__init__(name)
        self.system.check_root(d)
        self.p, self.params = self._init_sort(p, type(self).Params, 
            c1=c1, c2=c2, discount=discount, th_cond=th_cond, th_crit=th_crit)
        self.d = d
        index = self.system.get_index(keyform(d))
        self.main = State(index, {}, math.log(c1/c2, 2))
        self.posm = State(index, {}, 0.0)
        self.negm = State(index, {}, 0.0)
        self.cond = State(index, {}, 0.0)
        self.crit = State(index, {}, 0.0)

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> Event:
        main = (self.posm[0].shift(self.params[0][~self.p.c1])
            .div(self.posm[0]
                .sum(self.negm[0])
                .shift(self.params[0][~self.p.c2]))
            .log()
            .scale(1/math.log(2)))
        return Event(self.update, [ForwardUpdate(self.main, main)], dt, priority)

    def increment(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> Event:
        cond = self.cond[0].isbetween(lb=self.params[0][~self.p.th_cond])
        pos = self.crit[0].isbetween(lb=self.params[0][~self.p.th_crit]) 
        neg = pos.neg().shift(1.0)
        posm = self.posm[0].sum(pos.mul(cond))
        negm = self.negm[0].sum(neg.mul(cond))
        return Event(self.increment,
            [ForwardUpdate(self.posm, posm),
             ForwardUpdate(self.negm, negm)],
            dt, priority)

    def discount(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> Event:
        posm = self.posm[0].scale(self.params[0][~self.p.discount])
        negm = self.negm[0].scale(self.params[0][~self.p.discount])
        return Event(self.discount,
            [ForwardUpdate(self.posm, posm),
             ForwardUpdate(self.negm, negm)],
            dt, priority)
