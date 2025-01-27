from typing import Any
from datetime import timedelta

from ..system import Process, UpdateSite, Event, Priority
from ..knowledge import Family, Sort, Chunks, Atoms, Atom, Chunk, Var, ByKwds, keyform
from ..numdicts import Key, KeyForm, NumDict, numdict, path


class Simulation(Process):
    pass


class Agent(Process):
    pass


class Input(Process):
    main: NumDict

    def __init__(self, 
        name: str, 
        b: Family | Sort | Atom, 
        f: Family | Sort
    ) -> None:
        super().__init__(name)
        self.system.check_root(b, f)
        idx_b = self.system.get_index(keyform(b))
        idx_f = self.system.get_index(keyform(f))
        self.main = numdict(idx_b * idx_f, {}, 0.0)

    def send(self, d: Chunk, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        data = {}
        for (t1, t2), weight in d._dyads_.items():
            if isinstance(t1, Var) or isinstance(t2, Var):
                raise TypeError("Var not allowed in input chunk.")
            key = path(t1).link(path(t2), 0)    
            if key not in self.main.i:
                raise ValueError(f"Unexpected dimension-value pair {key}")
            data[key] = weight
        self.system.schedule(self.send, UpdateSite(self.main, data), 
            dt=dt, priority=priority)


class ChoiceBase(Process):
    class Params(Atoms):
        sd: Atom

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
        b: Family | Sort | Atom,
        f: Family | Sort,
        *,
        sd: float = 1.0
    ) -> None:
        super().__init__(name, p, sd=sd)
        self.system.check_root(b, f)
        idx_b = self.system.get_index(keyform(b))
        idx_f = self.system.get_index(keyform(f))
        index = idx_b * idx_f
        self.main = numdict(index, {}, 0.0)
        self.input = numdict(index, {}, 0.0)
        self.bias = numdict(index, {}, 0.0)
        self.sample = numdict(index, {}, float("nan"))
        self.by = keyform(b) * keyform(f, trunc=1)


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


# class Pool(Process):
#     class Params(Atoms):
#         pass

#     p: Params
#     main: NumDict
#     params: NumDict
#     inputs: dict[Key, NumDict]

#     def __init__(self, 
#         name: str,         
#         pfam: Family,
#         fam1: Family,
#         fam2: Family | None = None
#     ) -> None:
#         super().__init__(name)
#         root = self.system.root
#         if not root == get_root(pfam) == get_root(fam1) \
#             or fam2 is not None and root != get_root(fam2):
#             raise ValueError("Mismatched root keyspaces")
#         if fam2 is None:
#             index = Index(root, path(fam1), (2,))
#         else:
#             k1 = path(fam1); k2 = path(fam2)
#             index = Index(root, k1.link(k2, 0), (2, 2))
#         self.p = type(self).Params(); pfam[name] = self.p
#         idx_p = Index(root, path(self.p), (1,))
#         self.params = numdict(idx_p, {}, c=float("nan"))
#         self.main = numdict(index, {}, c=0.0)
#         self.inputs = {}

#     def __setitem__(self, name: str, site: Any) -> None:
#         if not isinstance(site, NumDict):
#             raise TypeError()
#         if not self.main.i.keyform <= site.i.keyform:
#             raise ValueError()
#         self.p[name] = Atom()
#         key = path(self.p[name])
#         self.inputs[key] = site
#         with self.params.mutable():
#             self.params[key] = 1.0

#     def __getitem__(self, name: str) -> NumDict:
#         return self.inputs[path(self.p[name])]
        
#     def resolve(self, event: Event) -> None:
#         if (event.affects(self.params, ud_type=UpdateSite) 
#             or any(event.affects(site, ud_type=UpdateSite) 
#                 for site in self.inputs.values())):
#             self.update()

#     def update(self, 
#         dt: timedelta = timedelta(), 
#         priority: int = Priority.PROPAGATION
#     ) -> None:
#         inputs = [d.scale(x=self.params[k]) for k, d in self.inputs.items()]
#         zeros = numdict(self.main.i, {}, 0.0) 
#         pro = zeros.max(*inputs)
#         con = zeros.min(*inputs)
#         self.system.schedule(
#             self.update, 
#             UpdateSite(self.main, pro.sum(con).d),
#             dt=dt, priority=priority)


class BottomUp(Process):    
    main: NumDict
    input: NumDict
    weights: NumDict
    max_by: ByKwds

    def __init__(self, 
        name: str, 
        c: Chunks, 
        b: Family | Sort | Atom, 
        f: Family | Sort
    ) -> None:
        super().__init__(name)
        self.system.check_root(c, b, f)
        idx_c = self.system.get_index(keyform(c))
        idx_b = self.system.get_index(keyform(b))
        idx_f = self.system.get_index(keyform(f))
        self.main = numdict(idx_c, {}, c=0.0)
        self.input = numdict(idx_b * idx_f, {}, c=0.0)
        self.weights = numdict(idx_c * idx_b * idx_f, {}, c=float("nan"))
        self.max_by = ByKwds(by=keyform(c) * keyform(b) * keyform(f, trunc=1))

    def resolve(self, event: Event) -> None:
        if event.affects(self.input, ud_type=UpdateSite):
            self.update()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        result = (self.weights
            .mul(self.input, bs=(2,))
            .max(**self.max_by)
            .sum(by=self.main.i.keyform)
            .with_default(c=0.0))
        self.system.schedule(self.update, UpdateSite(self.main, result.d), 
            dt=dt, priority=priority)


class TopDown(Process):    
    main: NumDict
    input: NumDict
    weights: NumDict
    by: ByKwds

    def __init__(self, 
        name: str, 
        c: Chunks, 
        b: Family | Sort | Atom, 
        f: Family | Sort
    ) -> None:
        super().__init__(name)
        self.system.check_root(c, b, f)
        idx_c = self.system.get_index(keyform(c))
        idx_b = self.system.get_index(keyform(b))
        idx_f = self.system.get_index(keyform(f))
        self.main = numdict(idx_b * idx_f, {}, c=0.0)
        self.input = numdict(idx_c, {}, c=0.0)
        self.weights = numdict(idx_c * idx_b * idx_f, {}, c=float("nan")) 
        self.by = ByKwds(
            by=keyform(b) * keyform(f), 
            b=0 if not keyform(b) <= keyform(c) 
                else 1 if not keyform(f) <= keyform(b) 
                else 2)

    def resolve(self, event: Event) -> None:
        if event.affects(self.input, ud_type=UpdateSite):
            self.update()

    def update(self, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        cf = self.weights.mul(self.input, bs=(0,))
        pro = cf.max(**self.by).bound_min(x=0.0).with_default(c=0.0) 
        con = cf.min(**self.by).bound_max(x=0.0).with_default(c=0.0)
        result = pro.sum(con)
        self.system.schedule( 
            self.update, 
            UpdateSite(self.main, result.d),
            dt=dt, priority=priority)
