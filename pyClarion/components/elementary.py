from typing import Any, ClassVar
from datetime import timedelta

from ..system import Process, UpdateSite, Event, Priority
from ..knowledge import (Family, Sort, Chunks, Term, Atoms, Atom, Chunk, Var, 
    keyform)
from ..numdicts import Key, KeyForm, NumDict, numdict, path


class Simulation(Process):
    pass


class Agent(Process):
    def __init__(self, name: str, **families: Family) -> None:
        super().__init__(name)
        for name, family in families.items():
            self.system.root[name] = family


class Input(Process):
    main: NumDict
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
        self.main = numdict(idx_t, {}, c)
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
            if k not in self.main.i:
                raise ValueError(f"Unexpected key {k}")
            data[k] = v
        self.system.schedule(self.send, 
            UpdateSite(self.main, data, reset=self.reset), 
            dt=dt, priority=priority)
        

class InputBL(Process):
    main: NumDict
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
        self.main = numdict(idx_d * idx_v, {}, c)
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
            if key not in self.main.i:
                raise ValueError(f"Unexpected dimension-value pair {key}")
            data[key] = weight
        self.system.schedule(self.send, 
            UpdateSite(self.main, data, self.reset), 
            dt=dt, priority=priority)


class InputTL(Input):
    pass


class ChoiceBase(Process):
    class Params(Atoms):
        sd: Atom

    lax: ClassVar[tuple[str, ...]] = ("input",)

    p: Params
    by: KeyForm
    main: NumDict
    input: NumDict
    bias: NumDict
    sample: NumDict
    params: NumDict

    def __init__(self, name: str, p: Family, *, sd: float = 1.0) -> None:
        super().__init__(name)
        self.system.check_root(p)
        self.p = type(self).Params(); p[name] = self.p
        self.params = numdict(
            i=self.system.get_index(keyform(self.p)), 
            d={path(self.p.sd): sd}, 
            c=float("nan"))

    def poll(self) -> dict[Key, Key]:
        return self.main.argmax(by=self.by)

    def select(self, 
        dt: timedelta = timedelta(), 
        priority=Priority.CHOICE
    ) -> None:
        input = self.bias.sum(self.input)
        sd = numdict(self.main.i, {}, c=self.params[path(self.p.sd)])
        sample = input.normalvariate(sd)
        choices = sample.argmax(by=self.by)
        self.system.schedule(
            self.select,
            UpdateSite(self.main, {v: 1.0 for v in choices.values()}),
            UpdateSite(self.sample, sample.d),
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
        self.main = numdict(index, {}, 0.0)
        self.input = numdict(index, {}, 0.0)
        self.bias = numdict(index, {}, 0.0)
        self.sample = numdict(index, {}, float("nan"))
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
        self.main = numdict(index, {}, 0.0)
        self.input = numdict(index, {}, 0.0)
        self.bias = numdict(index, {}, 0.0)
        self.sample = numdict(index, {}, float("nan"))
        self.by = keyform(t, trunc=1)


class PoolBase(Process):
    class Params(Atoms):
        pass

    p: Params
    main: NumDict
    params: NumDict
    inputs: dict[Key, NumDict]

    def __init__(self, name: str, p: Family) -> None:
        super().__init__(name)
        self.system.check_root(p)
        self.p = type(self).Params(); p[name] = self.p
        idx_p = self.system.get_index(keyform(self.p))
        self.params = numdict(idx_p, {}, c=float("nan"))
        self.inputs = {}

    def __setitem__(self, name: str, site: Any) -> None:
        if not isinstance(site, NumDict):
            raise TypeError()
        if not self.main.i.keyform <= site.i.keyform:
            raise ValueError()
        self.p[name] = Atom()
        key = path(self.p[name])
        self.inputs[key] = site
        with self.params.mutable():
            self.params[key] = 1.0

    def __getitem__(self, name: str) -> NumDict:
        return self.inputs[path(self.p[name])]
        
    def resolve(self, event: Event) -> None:
        if (event.affects(self.params, ud_type=UpdateSite) 
            or any(event.affects(site, ud_type=UpdateSite) 
                for site in self.inputs.values())):
            self.update()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        inputs = [d.scale(x=self.params[k]) for k, d in self.inputs.items()]
        zeros = numdict(self.main.i, {}, 0.0) 
        pro = zeros.max(*inputs)
        con = zeros.min(*inputs)
        self.system.schedule(
            self.update, 
            UpdateSite(self.main, pro.sum(con).d),
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
        self.main = numdict(idx_d * idx_v, {}, 0.0)


class PoolTL(PoolBase):
    def __init__(self, name: str, p: Family, t: Family | Sort) -> None:
        super().__init__(name, p)
        self.system.check_root(t)
        idx_t = self.system.get_index(keyform(t))
        self.main = numdict(idx_t, {}, 0.0)


class AssociationsBase(Process):
    main: NumDict
    input: NumDict
    weights: NumDict
    sum_by: KeyForm

    def resolve(self, event: Event) -> None:
        if event.affects(self.weights):
            self.update(priority=Priority.LEARNING)
        elif event.affects(self.input):
            self.update(priority=Priority.PROPAGATION)

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        output = self.weights.mul(self.input).sum(by=self.sum_by)
        self.system.schedule(self.update, 
            UpdateSite(self.main, output.d), 
            dt=dt, priority=priority)


class ChunkAssocs(AssociationsBase):
    main: NumDict
    input: NumDict
    weights: NumDict
    sum_by: KeyForm

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
        self.main = numdict(idx_out, {}, 0.0)
        self.input = numdict(idx_in, {}, 0.0)
        self.weights = numdict(idx_in * idx_out, {}, 0.0)
        self.by = keyform(c_in).agg * keyform(c_out)


class BottomUp(Process):    
    main: NumDict
    input: NumDict
    weights: NumDict
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
        self.main = numdict(idx_c, {}, c=0.0)
        self.input = numdict(idx_d * idx_v, {}, c=0.0)
        self.weights = numdict(idx_c * idx_d * idx_v, {}, c=float("nan"))
        self.mul_by = keyform(c).agg * keyform(d) * keyform(v)
        self.sum_by = keyform(c) * keyform(d).agg * keyform(v, trunc=1).agg
        self.max_by = keyform(c) * keyform(d) * keyform(v, trunc=1)

    def resolve(self, event: Event) -> None:
        if event.affects(self.input, ud_type=UpdateSite):
            self.update()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        result = (self.weights
            .mul(self.input, by=self.mul_by)
            .max(by=self.max_by)
            .sum(by=self.sum_by)
            .with_default(c=0.0))
        self.system.schedule(self.update, UpdateSite(self.main, result.d), 
            dt=dt, priority=priority)


class TopDown(Process):    
    main: NumDict
    input: NumDict
    weights: NumDict
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
        self.main = numdict(idx_d * idx_v, {}, c=0.0)
        self.input = numdict(idx_c, {}, c=0.0)
        self.weights = numdict(idx_c * idx_d * idx_v, {}, c=float("nan")) 
        self.mul_by = keyform(c) * keyform(d).agg * keyform(v).agg
        self.maxmin_by = keyform(c).agg * keyform(d) * keyform(v)         

    def resolve(self, event: Event) -> None:
        if event.affects(self.input, ud_type=UpdateSite):
            self.update()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        cf = self.weights.mul(self.input, by=self.mul_by)
        pro = cf.max(by=self.maxmin_by).bound_min(x=0.0).with_default(c=0.0) 
        con = cf.min(by=self.maxmin_by).bound_max(x=0.0).with_default(c=0.0)
        result = pro.sum(con)
        self.system.schedule( 
            self.update, 
            UpdateSite(self.main, result.d),
            dt=dt, priority=priority)
