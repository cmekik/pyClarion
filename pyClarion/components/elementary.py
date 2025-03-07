from typing import Any, ClassVar, overload, Callable
from datetime import timedelta

from .base import V, DV, DualRepMixin, ParamMixin
from ..system import Process, Event, Priority, Site
from ..knowledge import (Family, Sort, Chunks, Term, Atoms, Atom, Chunk, Var, 
    keyform)
from ..numdicts import Key, KeyForm, numdict, path, NumDict


class Environment(Process):
    """
    A simulation environment.
    
    Initializes shared keyspaces for multi-agent simulations.
    """

    def __init__(self, name: str, **families: Family) -> None:
        super().__init__(name)
        for name, family in families.items():
            self.system.root[name] = family


class Agent(Process):
    """
    A simulated agent.
    
    Initializes keyspaces specific to an agent.
    """

    def __init__(self, name: str, **families: Family) -> None:
        super().__init__(name)
        for name, family in families.items():
            self.system.root[name] = family


class Input(DualRepMixin, Process):
    """
    An input receiver process.
    
    Receives activations from external sources.
    """

    main: Site
    reset: bool

    def __init__(self, 
        name: str, 
        s: V | DV,
        *,
        c: float = 0.0,
        reset: bool = True,
        lags: int = 0
    ) -> None:
        super().__init__(name)
        index, = self._init_indexes(s) 
        self.main = Site(index, {}, c, lags)
        self.reset = reset

    @overload
    def send(self, d: dict[Term, float], 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        ...

    @overload
    def send(self, d: dict[Key, float], 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        ...

    @overload
    def send(self, d: dict[Key | Term, float], 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        ...

    @overload
    def send(self, d: Chunk, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        ...

    def send(self, d: dict | Chunk, 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        """Update input data."""
        data = self._parse_input(d)
        method = Site.push if self.reset else Site.write_inplace
        self.system.schedule(self.send, 
            self.main.update(data, method), 
            dt=dt, priority=priority)

    def _parse_input(self, d: dict | Chunk) -> dict[Key, float]:
        data = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(k, Term):
                    self.system.check_root(k)
                    k = path(k)
                if k not in self.main.index:
                    raise ValueError(f"Unexpected key {k}")
                data[k] = v
        if isinstance(d, Chunk):
            for (t1, t2), weight in d._dyads_.items():
                if isinstance(t1, Var) or isinstance(t2, Var):
                    raise TypeError("Var not allowed in input chunk.")
                key = path(t1).link(path(t2), 0)    
                if key not in self.main.index:
                    raise ValueError(f"Unexpected dimension-value pair {key}")
                data[key] = weight
        return data


class Choice(ParamMixin, DualRepMixin, Process):
    """
    A choice process.
    
    Makes discrete stochastic decisions based on activation strengths.
    """

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

    def __init__(self, 
        name: str, 
        p: Family, 
        s: V | DV, 
        *, 
        sd: float = 1.0
    ) -> None:
        super().__init__(name)
        self.system.check_root(p)
        index, = self._init_indexes(s)
        self.p, self.params = self._init_params(p, type(self).Params, sd=sd)
        self.main = Site(index, {}, 0.0)
        self.input = Site(index, {}, 0.0)
        self.bias = Site(index, {}, 0.0)
        self.sample = Site(index, {}, float("nan"))
        self.by = self._init_by(s)

    @staticmethod
    def _init_by(s: V | DV) -> KeyForm:
        match s:
            case (d, v):
                return keyform(d) * keyform(v, trunc=1)
            case s:
                return keyform(s, trunc=1)

    def poll(self) -> dict[Key, Key]:
        """Return a symbolic representation of current decision."""
        return self.main[0].argmax(by=self.by)

    def select(self, 
        dt: timedelta = timedelta(), 
        priority=Priority.CHOICE
    ) -> None:
        """Make a new stochastic decision and update sites accordingly."""
        input = self.bias[0].sum(self.input[0])
        sd = numdict(self.main.index, {}, c=self.params[0][path(self.p.sd)])
        sample = input.normalvariate(sd)
        choices = sample.argmax(by=self.by)
        self.system.schedule(
            self.select,
            self.main.update({v: 1.0 for v in choices.values()}),
            self.sample.update(sample),
            dt=dt, priority=priority)


class Pool(ParamMixin, DualRepMixin, Process):
    """
    An activation pooling process.

    Combines activation strengths from multiple sources.
    """

    class Params(Atoms):
        pass

    p: Params
    main: Site
    params: Site
    inputs: dict[Key, Site]
    func: Callable

    @staticmethod
    def CAM(main: NumDict, *numdicts: NumDict) -> NumDict:
        pro = main.max(*numdicts)
        con = main.min(*numdicts)
        return pro.sum(con)

    @staticmethod
    def Heckerman(main: NumDict, *numdicts: NumDict) -> NumDict:
        return main.shift(x=1).scale(x=0.5).logit().sum(*(d.shift(x=1).scale(x=0.5).logit() for d in numdicts)).expit()


    def __init__(self, name: str, p: Family, s: V | DV, *, func: Callable = CAM) -> None:
        super().__init__(name)
        index, = self._init_indexes(s)
        psort, psite = self._init_params(p, type(self).Params)
        self.p = psort
        self.params = psite
        self.main = Site(index, {}, 0.0)
        self.inputs = {}
        self.func = func

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
        main = self.func(self.main.new({}), *inputs)
        self.system.schedule(
            self.update, 
            self.main.update(main),
            dt=dt, priority=priority)


# # class AssociationsBase(Process):
# #     main: Site
# #     input: Site
# #     weights: Site
# #     sum_by: KeyForm

# #     def resolve(self, event: Event) -> None:
# #         updates = [ud for ud in event.updates if isinstance(ud, Site.Update)]
# #         if self.input.affected_by(*updates):
# #             self.update()

# #     def update(self, 
# #         dt: timedelta = timedelta(), 
# #         priority: int = Priority.PROPAGATION
# #     ) -> None:
# #         main = self.weights[0].mul(self.input[0]).sum(by=self.sum_by)
# #         self.system.schedule(self.update, 
# #             self.main.update(main), 
# #             dt=dt, priority=priority)


# class ChunkAssocs(AssociationsBase):
#     def __init__(self, 
#         name: str, 
#         c_in: Chunks, 
#         c_out: Chunks | None = None
#     ) -> None:
#         c_out = c_in if c_out is None else c_out
#         super().__init__(name)
#         self.system.check_root(c_in, c_out)
#         idx_in = self.system.get_index(keyform(c_in))
#         idx_out = self.system.get_index(keyform(c_out))
#         self.main = Site(idx_out, {}, 0.0)
#         self.input = Site(idx_in, {}, 0.0)
#         self.weights = Site(idx_in * idx_out, {}, 0.0)
#         self.by = keyform(c_in).agg * keyform(c_out)


class BottomUp(Process):
    """
    A bottom-up activation process.

    Propagates activations from the bottom level to the top level.
    """

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
    """
    A top-down activation process.

    Propagates activations from the top level to the bottom level.
    """

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
