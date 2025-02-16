from typing import Any, ClassVar
from datetime import timedelta

from ..system import Process, Event, Priority, Site
from ..knowledge import (Family, Sort, Chunks, Term, Atoms, Atom, Chunk, Var, 
    keyform)
from ..numdicts import Key, KeyForm, numdict, path


class Simulation(Process):
    pass


class Agent(Process):
    def __init__(self, name: str, **families: Family) -> None:
        super().__init__(name)
        for name, family in families.items():
            self.system.root[name] = family


class Input(Process):
    main: Site
    reset: bool

    def __init__(self, 
        name: str, 
        t: Family | Sort,
        *,
        c: float = 0.0,
        reset: bool = True
    ) -> None:
        super().__init__(name)
        self.system.check_root(t)
        idx_t = self.system.get_index(keyform(t))
        self.main = Site(idx_t, {}, c)
        self.reset = reset

    def send(self, d: dict[Term | Key, float], 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        data = {}
        for k, v in d.items():
            if isinstance(k, Term):
                self.system.check_root(k)
                k = path(k)
            if k not in self.main.index:
                raise ValueError(f"Unexpected key {k}")
            data[k] = v
        if self.reset:
            main = self.main.new(data)
        else:
            main = self.main[0].copy()
            with main.mutable():
                main.update(data)
        self.system.schedule(self.send, 
            self.main.update(main), 
            dt=dt, priority=priority)
        

class InputBL(Process):
    main: Site
    reset: bool

    def __init__(self, 
        name: str, 
        d: Family | Sort | Atom, 
        v: Family | Sort,
        *,
        c: float = 0.0,
        reset: bool = True
    ) -> None:
        super().__init__(name)
        self.system.check_root(d, v)
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        self.main = Site(idx_d * idx_v, {}, c)
        self.reset = reset

    def send(self, c: Chunk, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        data = {}
        for (t1, t2), weight in c._dyads_.items():
            if isinstance(t1, Var) or isinstance(t2, Var):
                raise TypeError("Var not allowed in input chunk.")
            key = path(t1).link(path(t2), 0)    
            if key not in self.main.index:
                raise ValueError(f"Unexpected dimension-value pair {key}")
            data[key] = weight
        if self.reset:
            main = self.main.new(data)
        else:
            main = self.main[0].copy()
            with main.mutable():
                main.update(data)
        self.system.schedule(self.send, 
            self.main.update(main),
            dt=dt, priority=priority)


class InputTL(Input):
    pass


class ChoiceBase(Process):
    class Params(Atoms):
        sd: Atom

    lax: ClassVar[tuple[str, ...]] = ("input",)

    p: Params
    by: KeyForm
    main: Site
    input: Site
    bias: Site
    sample: Site
    params: Site

    def __init__(self, name: str, p: Family, *, sd: float = 1.0) -> None:
        super().__init__(name)
        self.system.check_root(p)
        self.p = type(self).Params(); p[name] = self.p
        self.params = Site(
            i=self.system.get_index(keyform(self.p)), 
            d={path(self.p.sd): sd}, 
            c=float("nan"))

    def poll(self) -> dict[Key, Key]:
        return self.main[0].argmax(by=self.by)

    def select(self, 
        dt: timedelta = timedelta(), 
        priority=Priority.CHOICE
    ) -> None:
        input = self.bias[0].sum(self.input[0])
        sd = numdict(self.main.index, {}, c=self.params[0][path(self.p.sd)])
        sample = input.normalvariate(sd)
        choices = sample.argmax(by=self.by)
        self.system.schedule(
            self.select,
            self.main.update({v: 1.0 for v in choices.values()}),
            self.sample.update(sample),
            dt=dt, priority=priority)


class ChoiceBL(ChoiceBase):
    def __init__(
        self, 
        name: str, 
        p: Family,
        d: Family | Sort | Atom,
        v: Family | Sort,
        *,
        sd: float = 1.0
    ) -> None:
        super().__init__(name, p, sd=sd)
        self.system.check_root(d, v)
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        index = idx_d * idx_v
        self.main = Site(index, {}, 0.0)
        self.input = Site(index, {}, 0.0)
        self.bias = Site(index, {}, 0.0)
        self.sample = Site(index, {}, float("nan"))
        self.by = keyform(d) * keyform(v, trunc=1)


class ChoiceTL(ChoiceBase):
    def __init__(
        self, 
        name: str, 
        p: Family,
        t: Family | Sort,
        *,
        sd: float = 1.0
    ) -> None:
        super().__init__(name, p, sd=sd)
        self.system.check_root(t)
        index = self.system.get_index(keyform(t))
        self.main = Site(index, {}, 0.0)
        self.input = Site(index, {}, 0.0)
        self.bias = Site(index, {}, 0.0)
        self.sample = Site(index, {}, float("nan"))
        self.by = keyform(t, trunc=1)


class PoolBase(Process):
    class Params(Atoms):
        pass

    p: Params
    main: Site
    params: Site
    inputs: dict[Key, Site]

    def __init__(self, name: str, p: Family) -> None:
        super().__init__(name)
        self.system.check_root(p)
        self.p = type(self).Params(); p[name] = self.p
        idx_p = self.system.get_index(keyform(self.p))
        self.params = Site(idx_p, {}, c=float("nan"))
        self.inputs = {}

    def __setitem__(self, name: str, site: Any) -> None:
        if not isinstance(site, Site):
            raise TypeError()
        if not self.main.index.keyform <= site.index.keyform:
            raise ValueError()
        self.p[name] = Atom()
        key = path(self.p[name])
        self.inputs[key] = site
        with self.params[0].mutable():
            self.params[0][key] = 1.0

    def __getitem__(self, name: str) -> Site:
        return self.inputs[path(self.p[name])]
        
    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if (self.params.affected_by(*updates) \
            or any(site.affected_by(*updates) 
                for site in self.inputs.values())):
            self.update()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        inputs = [s[0].scale(x=self.params[0][k]) 
            for k, s in self.inputs.items()]
        main = self.main.new({})
        pro = main.max(*inputs)
        con = main.min(*inputs)
        main = pro.sum(con)
        self.system.schedule(
            self.update, 
            self.main.update(main),
            dt=dt, priority=priority)


class PoolBL(PoolBase):
    def __init__(self, 
        name: str, 
        p: Family, 
        d: Family | Sort | Atom, 
        v: Family | Sort
    ) -> None:
        super().__init__(name, p)
        self.system.check_root(d, v)
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        self.main = Site(idx_d * idx_v, {}, 0.0)


class PoolTL(PoolBase):
    def __init__(self, name: str, p: Family, t: Family | Sort) -> None:
        super().__init__(name, p)
        self.system.check_root(t)
        idx_t = self.system.get_index(keyform(t))
        self.main = Site(idx_t, {}, 0.0)


class AssociationsBase(Process):
    main: Site
    input: Site
    weights: Site
    sum_by: KeyForm

    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.update()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        main = self.weights[0].mul(self.input[0]).sum(by=self.sum_by)
        self.system.schedule(self.update, 
            self.main.update(main), 
            dt=dt, priority=priority)


class ChunkAssocs(AssociationsBase):
    def __init__(self, 
        name: str, 
        c_in: Chunks, 
        c_out: Chunks | None = None
    ) -> None:
        c_out = c_in if c_out is None else c_out
        super().__init__(name)
        self.system.check_root(c_in, c_out)
        idx_in = self.system.get_index(keyform(c_in))
        idx_out = self.system.get_index(keyform(c_out))
        self.main = Site(idx_out, {}, 0.0)
        self.input = Site(idx_in, {}, 0.0)
        self.weights = Site(idx_in * idx_out, {}, 0.0)
        self.by = keyform(c_in).agg * keyform(c_out)


class BottomUp(Process):    
    main: Site
    input: Site
    weights: Site
    mul_by: KeyForm
    sum_by: KeyForm
    max_by: KeyForm

    def __init__(self, 
        name: str, 
        c: Chunks, 
        d: Family | Sort | Atom, 
        v: Family | Sort
    ) -> None:
        super().__init__(name)
        self.system.check_root(c, d, v)
        idx_c = self.system.get_index(keyform(c))
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        self.main = Site(idx_c, {}, c=0.0)
        self.input = Site(idx_d * idx_v, {}, c=0.0)
        self.weights = Site(idx_c * idx_d * idx_v, {}, c=float("nan"))
        self.mul_by = keyform(c).agg * keyform(d) * keyform(v)
        self.sum_by = keyform(c) * keyform(d).agg * keyform(v, trunc=1).agg
        self.max_by = keyform(c) * keyform(d) * keyform(v, trunc=1)

    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.update()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        main = (self.weights[0]
            .mul(self.input[0], by=self.mul_by)
            .max(by=self.max_by)
            .sum(by=self.sum_by)
            .with_default(c=0.0))
        self.system.schedule(self.update, 
            self.main.update(main), 
            dt=dt, priority=priority)


class TopDown(Process):    
    main: Site
    input: Site
    weights: Site
    mul_by: KeyForm
    maxmin_by: KeyForm

    def __init__(self, 
        name: str, 
        c: Chunks, 
        d: Family | Sort | Atom, 
        v: Family | Sort
    ) -> None:
        super().__init__(name)
        self.system.check_root(c, d, v)
        idx_c = self.system.get_index(keyform(c))
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        self.main = Site(idx_d * idx_v, {}, c=0.0)
        self.input = Site(idx_c, {}, c=0.0)
        self.weights = Site(idx_c * idx_d * idx_v, {}, c=float("nan")) 
        self.mul_by = keyform(c) * keyform(d).agg * keyform(v).agg
        self.maxmin_by = keyform(c).agg * keyform(d) * keyform(v)         

    def resolve(self, event: Event) -> None:
        updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
        if self.input.affected_by(*updates):
            self.update()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        cf = self.weights[0].mul(self.input[0], by=self.mul_by)
        pro = cf.max(by=self.maxmin_by).bound_min(x=0.0).with_default(c=0.0) 
        con = cf.min(by=self.maxmin_by).bound_max(x=0.0).with_default(c=0.0)
        main = pro.sum(con)
        self.system.schedule(self.update, 
            self.main.update(main),
            dt=dt, priority=priority)
