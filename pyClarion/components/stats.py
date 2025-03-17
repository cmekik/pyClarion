from datetime import timedelta
import math

from .base import ParamMixin
from ..system import Process, UpdateSort, Priority, Event, Site
from ..knowledge import Family, Sort, Atoms, Atom
from ..numdicts import Key, keyform


class BaseLevel(Process):
    """
    A base-level activation process.

    Maintains and propagates base level activations.
    """

    class Params(Atoms):
        th: Atom
        sc: Atom
        de: Atom
    
    e: Atoms
    p: Params
    unit: timedelta
    ignore: set[Key]
    main: Site
    input: Site
    times: Site
    decay: Site
    scale: Site
    weights: Site
    params: Site
    
    def __init__(self, 
        name: str, 
        p: Family, 
        e: Family, 
        s: Sort, 
        *, 
        unit: timedelta = timedelta(milliseconds=1),
        th: float = 0.0, 
        sc: float = 1.0, 
        de: float = 0.5
    ) -> None:
        super().__init__(name)
        self.system.check_root(p, e, s)
        if e == p:
            raise ValueError("Args p and e must be distinct")
        self.p = type(self).Params(); p[name] = self.p
        self.e = Atoms(); e[name] = self.e
        idx_p = self.system.get_index(keyform(self.p))
        idx_e = self.system.get_index(keyform(self.e))
        idx_s = self.system.get_index(keyform(s))
        self.unit = unit
        self.ignore = set()
        self.main = Site(idx_s, {}, 1.0)
        self.input = Site(idx_s, {}, 0.0)
        self.times = Site(idx_e, {}, float("nan"))
        self.decay = Site(idx_e, {}, float("nan"))
        self.scale = Site(idx_e, {}, float("nan"))
        self.weights = Site(idx_e * idx_s, {}, 0.0)
        self.params = Site(idx_p, 
            {~self.p.th: th, ~self.p.sc: sc, ~self.p.de: de}, 
            float("nan"))

    def resolve(self, event: Event) -> None:
        if self.input.affected_by(*event.updates):
            self.invoke(event)

    def invoke(self,
        event: Event, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.LEARNING
    ) -> None:
        ke = ~self.e; name = f"e{next(self.e._counter_)}"
        key = ke.link(Key(name), ke.size)
        invoked = set(); th = self.params[0][~self.p.th]
        for ud in (ud for ud in event.updates if self.input.affected_by(ud)):
            if isinstance(ud, Site.Update):
                for k in ud.data:
                    if k not in self.ignore and th < ud.data[k]:
                        invoked.add(key.link(k, 0))
            if isinstance(ud, UpdateSort):
                for term in ud.add:
                    k = key.link(~term, 0)
                    if k not in self.ignore:
                        invoked.add(k)
        time = self.system.clock.time / self.unit
        sc = self.params[0][~self.p.sc] 
        de = self.params[0][~self.p.de]
        atom = Atom()
        atom._name_ = name
        self.system.schedule(self.invoke, 
            UpdateSort(self.e, add=(atom,)),
            self.times.update({key: time}, Site.write_inplace),
            self.scale.update({key: sc}, Site.write_inplace),
            self.decay.update({key: de}, Site.write_inplace),
            self.weights.update({k: 1.0 for k in invoked}, Site.write_inplace),
            dt=dt, priority=priority)

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.LEARNING
    ) -> None:
        time = self.system.clock.time / self.unit
        terms = (self.times[0]
            .neg()
            .shift(x=time)
            .div(self.scale[0])
            .log()
            .mul(self.decay[0].neg())
            .exp())
        blas = (self.weights[0]
            .mul(terms)
            .sum(by=self.main.index.kf)
            .with_default(c=0.0))
        with blas.mutable():
            for k in self.ignore:
                blas[k] = 1.0
        self.system.schedule(self.update, 
            self.main.update(blas.d), 
            dt=dt, priority=priority)


class MatchStats(ParamMixin, Process):
    """A process that maintains match statistics."""
    
    class Params(Atoms):
        c1: Atom
        c2: Atom
        discount: Atom
        th_cond: Atom
        th_crit: Atom

    p: Params
    main: Site
    posm: Site
    negm: Site
    cond: Site
    crit: Site
    params: Site

    def __init__(self,
        name: str, 
        p: Family, 
        s: Sort, 
        *, 
        c1=1.0, 
        c2=2.0, 
        discount=.9, 
        th_cond=0.0, 
        th_crit=0.0
    ) -> None:
        super().__init__(name)
        self.system.check_root(s)
        self.p, self.params = self._init_params(p, type(self).Params, 
            c1=c1, c2=c2, discount=discount, th_cond=th_cond, th_crit=th_crit)
        index = self.system.get_index(keyform(s))
        self.main = Site(index, {}, math.log(c1/c2, 2))
        self.posm = Site(index, {}, 0.0)
        self.negm = Site(index, {}, 0.0)
        self.cond = Site(index, {}, 0.0)
        self.crit = Site(index, {}, 0.0)

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> None:
        main = (self.posm[0].shift(x=self.params[0][~self.p.c1])
            .div(self.posm[0]
                .sum(self.negm[0])
                .shift(x=self.params[0][~self.p.c2]))
            .log()
            .scale(x=1/math.log(2)))
        self.system.schedule(
            self.update,
            self.main.update(main),
            dt=dt, priority=priority)

    def increment(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> None:
        cond = (self.crit.new({})
            .with_default(c=self.params[0][~self.p.th_cond])
            .lt(self.cond[0]))
        pos = (self.crit.new({})
            .with_default(c=self.params[0][~self.p.th_crit])
            .lt(self.crit[0]))
        neg = pos.neg().shift(x=1.0)
        posm = self.posm[0].sum(pos.mul(cond))
        negm = self.negm[0].sum(neg.mul(cond))
        self.system.schedule(
            self.increment,
            self.posm.update(posm),
            self.negm.update(negm))

    def discount(self, 
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> None:
        posm = self.posm[0].scale(x=self.params[0][~self.p.discount])
        negm = self.negm[0].scale(x=self.params[0][~self.p.discount])
        self.system.schedule(
            self.discount,
            self.posm.update(posm),
            self.negm.update(negm),
            dt=dt, priority=priority)