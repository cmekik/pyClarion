from datetime import timedelta

from ..system import Process, UpdateSort, Priority, Event, Site
from ..knowledge import Family, Sort, Atoms, Atom, keyform
from ..numdicts import Key, path


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
            {path(self.p.th): th, path(self.p.sc): sc, path(self.p.de): de}, 
            float("nan"))

    def resolve(self, event: Event) -> None:
        if self.input.affected_by(*event.updates):
            self.invoke(event)

    def invoke(self,
        event: Event, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.LEARNING
    ) -> None:
        ke = path(self.e); name = f"e{next(self.e._counter_)}"
        key = ke.link(Key(name), ke.size)
        invoked = set(); th = self.params[0][path(self.p.th)]
        for ud in (ud for ud in event.updates if self.input.affected_by(ud)):
            if isinstance(ud, Site.Update):
                for k in ud.data:
                    if k not in self.ignore and th < ud.data[k]:
                        invoked.add(key.link(k, 0))
            if isinstance(ud, UpdateSort):
                for _, term in ud.add:
                    k = key.link(path(term), 0)
                    if k not in self.ignore:
                        invoked.add(k)
        time = self.system.clock.time / self.unit
        sc = self.params[0][path(self.p.sc)] 
        de = self.params[0][path(self.p.de)]
        self.system.schedule(self.invoke, 
            UpdateSort(self.e, add=((name, Atom()),)),
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
            .sum(by=self.main.index.keyform)
            .with_default(c=0.0))
        with blas.mutable():
            for k in self.ignore:
                blas[k] = 1.0
        self.system.schedule(self.update, 
            self.main.update(blas.d), 
            dt=dt, priority=priority)
