from typing import Self, Callable
from datetime import timedelta

from .base import Backprop, LayerBase, Cost, sq_err
from ..elementary import ChoiceBase, ChoiceBL
from ...system import Site, Priority, Event, PROCESS
from ...knowledge import Family, Sort, Atoms, Atom, keyform, Term
from ...numdicts import NumDict, Index, path, Key


class ErrorSignal(Backprop):
    main: Site

    def __rrshift__(self: Self, other: "LayerBase") -> Self:
        if isinstance(other, LayerBase):
            self.input = other.main
            other.error = self.main
            return self
        return NotImplemented
    
    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> None:
        raise NotImplementedError()


class Supervised(ErrorSignal):
    cost: Cost
    input: Site
    target: Site
    mask: Site

    def __init__(self, 
        name: str, 
        d: Family | Sort | Atom, 
        v: Family | Sort, 
        cost: Cost = sq_err
    ) -> None:
        super().__init__(name)
        type(self).check_grad(cost)
        self.system.check_root(d, v)
        idx_d = self.system.get_index(keyform(d))
        idx_v = self.system.get_index(keyform(v))
        self._init(idx_d * idx_v)
        self.cost = cost

    def _init(self, idx):
        self.main = Site(idx, {}, c=0.0)
        self.input = Site(idx, {}, c=0.0)
        self.target = Site(idx, {}, c=0.0)
        self.mask = Site(idx, {}, c=0.0)
    
    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> None:
        exp_mask = self.mask[0].exp()
        main = self.grad(self.cost)(self.input[0], self.target[0], exp_mask)
        self.system.schedule(self.update, 
            self.main.update(main), 
            dt=dt, priority=priority)
    

class TDError(ErrorSignal):
    class Params(Atoms):
        gamma: Atom

    choice: ChoiceBase
    func: Callable[["TDError"], NumDict]
    main: Site
    input: Site
    reward: Site
    qvals: Site
    action: Site

    def sarsa(self) -> NumDict:
        return (self.main.new({})
            .sum(self.qvals[0].mul(self.action[0]).sum())
            .with_default(c=0.0))

    def qmax(self) -> NumDict:
        return (self.main.new({})
            .sum(self.qvals[0]
                .max(by=self.choice.by)))
        
    def __init__(self, 
        name: str, 
        p: Family, 
        r: Sort, 
        *,
        func: Callable[["TDError"], NumDict] = qmax,
        gamma: float = .5,
        lags: int = 1
    ) -> None:
        if lags < 1:
            raise ValueError(f"Expected 1 <= lags but got lags == {lags}.")
        super().__init__(name)
        self.func = func
        self.system.check_root(p, r)
        self._connect_to_choice()
        self.p = type(self).Params(); p[name] = self.p
        self.params = Site(
            i=self.system.get_index(keyform(self.p)), 
            d={path(self.p.gamma): gamma}, 
            c=float("nan"))
        idx_r = self.system.get_index(keyform(r))
        self._init(idx_r, self.choice.main.index, lags)

    def _connect_to_choice(self):
        try:
            sup = PROCESS.get()
        except LookupError:
            raise RuntimeError("TDError must be initalized in ChoiceBL context")
        if not isinstance(sup, ChoiceBL):
            raise RuntimeError("TDError must be initalized in ChoiceBL context")
        self.choice = sup

    def _init(self, idx_r: Index, idx_a: Index, lags: int):
        self.main = Site(idx_a, {}, c=0.0)
        self.input = Site(idx_a, {}, c=0.0, l=lags)
        self.qvals = Site(idx_a, {}, c=0.0, l=lags)
        self.reward = Site(idx_r, {}, c=0.0, l=lags)
        self.action = Site(idx_a, {}, c=0.0, l=lags)

    def resolve(self, event: Event) -> None:
        if event.source == self.choice.select:
            self.update()

    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.PROPAGATION
    ) -> None:
        gamma = self.params[0][path(self.p.gamma)]
        n = len(self.reward)
        main = (self.func(self)
            .scale(x=gamma ** n)
            .sum(*(rwd.sum().scale(x=gamma ** (n - t - 1)) 
                for t, rwd in enumerate(self.reward.data) if t < n - 1))
            .sub(self.qvals[-1])
            .mul(self.action[-1])
            .with_default(c=self.main.const))
        self.system.schedule(self.update,
            self.main.update(main),
            self.reward.update({}),
            self.qvals.update(self.input[0].d),
            self.action.update(self.choice.main[0].d),
            dt=dt, priority=priority)

    def send(self, d: dict[Term | Key, float], 
        dt: timedelta = timedelta(), 
        priority: int = Priority.PROPAGATION
    ) -> None:
        data = {}
        for k, v in d.items():
            if isinstance(k, Term):
                self.system.check_root(k)
                k = path(k)
            if k not in self.reward.index:
                raise ValueError(f"Unexpected key {k}")
            data[k] = v
        self.system.schedule(self.send, 
            self.reward.update(data, Site.add_inplace), 
            dt=dt, priority=priority)
