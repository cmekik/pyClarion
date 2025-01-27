from datetime import timedelta

from ..system import Process, UpdateSite, UpdateSort, Priority, Event
from ..knowledge import Family, Sort, Atoms, Atom, keyform
from ..numdicts import Key, NumDict, path, numdict


class BaseLevel(Process):
    class Params(Atoms):
        th: Atom
        sc: Atom
        de: Atom
    
    e: Atoms
    p: Params
    unit: timedelta
    main: NumDict
    input: NumDict
    times: NumDict
    decay: NumDict
    scale: NumDict
    weights: NumDict
    params: NumDict
    
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
        self.main = numdict(idx_s, {}, 0.0)
        self.input = numdict(idx_s, {}, 0.0)
        self.times = numdict(idx_e, {}, float("nan"))
        self.decay = numdict(idx_e, {}, float("nan"))
        self.scale = numdict(idx_e, {}, float("nan"))
        self.weights = numdict(idx_e * idx_s, {}, 0.0)
        self.params = numdict(idx_p, 
            {path(self.p.th): th, path(self.p.sc): sc, path(self.p.de): de}, 
            float("nan"))

    def resolve(self, event: Event) -> None:
        if event.affects(self.input):
            self.invoke(event)

    def invoke(self,
        event: Event, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.LEARNING
    ) -> None:
        ke = path(self.e); name = f"e{next(self.e._counter_)}"
        key = ke.link(Key(name), ke.size)
        invoked = set(); th = self.params[path(self.p.th)]
        for ud in (ud for ud in event.updates if ud.affects(self.input)):
            if isinstance(ud, UpdateSite):
                for k, v in ud.data.items():
                    if th < v:
                        invoked.add(key.link(k, 0))
            if isinstance(ud, UpdateSort):
                for _, term in ud.add:
                    invoked.add(key.link(path(term), 0))
        time = self.system.clock.time / self.unit
        sc = self.params[path(self.p.sc)]; de = self.params[path(self.p.de)]
        self.system.schedule(self.invoke, 
            UpdateSort(self.e, add=((name, Atom()),)),
            UpdateSite(self.times, {key: time}, reset=False),
            UpdateSite(self.scale, {key: sc}, reset=False),
            UpdateSite(self.decay, {key: de}, reset=False),
            UpdateSite(self.weights, {k: 1.0 for k in invoked}, reset=False),
            dt=dt, priority=priority)

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.LEARNING
    ) -> None:
        time = self.system.clock.time / self.unit
        terms = (self.times
            .neg()
            .shift(x=time)
            .div(self.scale)
            .log()
            .mul(self.decay.neg())
            .exp())
        blas = (self.weights
            .mul(terms)
            .sum(by=self.main.i.keyform)
            .with_default(c=0.0))
        self.system.schedule(self.update, 
            UpdateSite(self.main, blas.d), 
            dt=dt, priority=priority)
