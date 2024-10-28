from typing import Any
from datetime import timedelta
from math import exp

from ..system import Process, UpdateSite, Event
from ..knowledge import Family, Chunks, Atoms, Atom, Monads, Dyads, Triads, Chunk, Var, ByKwds
from ..numdicts import Key, KeyForm, Index, NumDict, numdict, path, parent


class Simulation(Process):
    pass


class Agent(Process):
    pass


class Input(Process):
    main: NumDict

    def __init__(self, name: str, index: Dyads) -> None:
        super().__init__(name)
        root = self.system.root
        if index.root is not root:
            raise ValueError()
        self.main = numdict(index, {}, 0.0)

    def send(self, d: Chunk, dt: timedelta = timedelta()) -> None:
        data = {}
        for (t1, t2), weight in d._dyads_.items():
            if isinstance(t1, Var) or isinstance(t2, Var):
                raise TypeError("Var not allowed in input chunk.")
            key = path(t1).link(path(t2), 0)    
            if key not in self.main.i:
                raise ValueError(f"Unexpected dimension-value pair {key}")
            data[key] = weight
        self.system.schedule(self.send, UpdateSite(self.main, data), dt=dt)


class Choice(Process):
    class Params(Atoms):
        sd: Atom
        lf: Atom

    p: Params
    by: KeyForm
    main: NumDict
    input: NumDict
    bias: NumDict
    sample: NumDict
    params: NumDict

    def __init__(
        self, 
        name: str, 
        index: Monads | Dyads,
        *,
        sd: float = 1.0,
        lf: float = 0.0
    ) -> None:
        super().__init__(name)
        root = self.system.root
        if index.root is not root:
            raise ValueError()
        if "p" not in root: 
            root.p = Family()
        self.p = type(self).Params(); root.p[name] = self.p
        self.params = numdict(Monads(self.p), 
            {path(self.p.sd): sd, path(self.p.lf): lf}, float("nan"))
        self.main = numdict(index, {}, 0.0)
        self.input = numdict(index, {}, 0.0)
        self.bias = numdict(index, {}, 0.0)
        self.sample = numdict(index, {}, float("nan"))
        self.by = index.trunc((0, 1) if isinstance(index, Monads) else (1, 1))

    def poll(self) -> dict[Key, Key]:
        return self.main.argmax(by=self.by)

    def select(self) -> None:
        input = self.bias.sum(self.input)
        sd = numdict(self.main.i, {}, c=self.params[path(self.p.sd)])
        sample = input.normalvariate(sd)
        choices = sample.argmax(by=self.by)
        rt = (self.params[path(self.p.lf)] 
            * exp(-sample.max(by=self.by).valmin()))
        self.system.schedule(
            self.select,
            UpdateSite(self.main, {v: 1.0 for v in choices.values()}),
            UpdateSite(self.sample, sample.d),
            dt=timedelta(milliseconds=rt))


class Pool(Process):
    class Params(Atoms):
        pass

    p: Params
    main: NumDict
    params: NumDict
    inputs: dict[Key, NumDict]

    def __init__(self, name: str, index: Monads | Dyads) -> None:
        super().__init__(name)
        root = self.system.root
        if index.root is not root:
            raise ValueError()
        if "p" not in root: 
            root.p = Family() 
        self.p = type(self).Params(); root.p[name] = self.p
        self.params = numdict(Monads(self.p), {}, c=float("nan"))
        self.main = numdict(index, {}, c=0.0)
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
        if (event.affects(self.params) 
            or any(event.affects(site) for site in self.inputs.values())):
            self.update()

    def update(self, dt: timedelta = timedelta()) -> None:
        inputs = [d.scale(x=self.params[k]) for k, d in self.inputs.items()]
        zeros = numdict(self.main.i, {}, 0.0) 
        pro = zeros.max(*inputs)
        con = zeros.min(*inputs)
        self.system.schedule(
            self.update, 
            UpdateSite(self.main, pro.sum(con).d),
            dt=dt)


class BottomUp(Process):    
    main: NumDict
    input: NumDict
    weights: NumDict
    max_by: KeyForm

    def __init__(self, name: str, tl: Chunks, bl1: Family, bl2: Family) -> None:
        super().__init__(name)
        root = self.system.root
        kt = path(tl); kb1 = path(bl1); kb2 = path(bl2)
        idx_m = Index(root, kt, (1,))
        idx_i = Index(root, kb1.link(kb2, 0), (2, 2))
        idx_w = Index(root, kt.link(kb1, 0).link(kb2, 0), (2, 2, 1))
        self.main = numdict(idx_m, {}, c=0.0)
        self.input = numdict(idx_i, {}, c=0.0)
        self.weights = numdict(idx_w, {}, c=float("nan"))
        self.max_by = KeyForm(kt.link(kb1, 0).link(kb2, 0), (2, 1, 1))

    def resolve(self, event: Event) -> None:
        if event.affects(self.input) or event.affects(self.weights):
            self.update()

    def update(self, dt: timedelta = timedelta()) -> None:
        result = (self.weights
            .mul(self.input, bs=(1,))
            .max(by=self.max_by)
            .sum(by=self.main.i.keyform)
            .with_default(c=0.0))
        self.system.schedule(self.update, UpdateSite(self.main, result.d), dt=dt)


class TopDown(Process):    
    main: NumDict
    input: NumDict
    weights: NumDict
    by: ByKwds

    def __init__(self, name: str, tl: Chunks, bl1: Family, bl2: Family) -> None:
        super().__init__(name)
        root = self.system.root
        kt = path(tl); kb1 = path(bl1); kb2 = path(bl2)
        idx_m = Index(root, kb1.link(kb2, 0), (2, 2))
        idx_i = Index(root, kt, (1,))
        idx_w = Index(root, kt.link(kb1, 0).link(kb2, 0), (2, 2, 1))
        self.main = numdict(idx_m, {}, c=0.0)
        self.input = numdict(idx_i, {}, c=0.0)
        self.weights = numdict(idx_w, {}, c=float("nan"))
        self.by = ByKwds(
            by=self.main.i.keyform, 
            b=0 if parent(tl) != bl1 else 1)

    def resolve(self, event: Event) -> None:
        if event.affects(self.input) or event.affects(self.weights):
            self.update()

    def update(self, dt: timedelta = timedelta()) -> None:
        cf = self.weights.mul(self.input, bs=(0,))
        pro = cf.max(**self.by).bound_min(x=0.0).with_default(c=0.0) 
        con = cf.min(**self.by).bound_max(x=0.0).with_default(c=0.0)
        result = pro.sum(con)
        self.system.schedule( 
            self.update, 
            UpdateSite(self.main, result.d),
            dt=dt)
