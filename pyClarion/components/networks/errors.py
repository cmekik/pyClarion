from typing import Callable
from datetime import timedelta

from .base import ErrorSignal
from ..base import DualRepMixin, ParamMixin, D, V, DV
from ..elementary import Choice
from ...system import Site, Priority, Event, PROCESS
from ...knowledge import Family, Atoms, Atom, Term
from ...numdicts import NumDict, path, Key


class Cost:
    """A differentiable cost function for supervised learning."""

    def __call__(self, est: NumDict, tgt: NumDict, mask: NumDict) -> NumDict:
        """Compute the cost for each estimate."""
        raise NotImplementedError()
    
    def grad(self, est: NumDict, tgt: NumDict, mask: NumDict) -> NumDict:
        """Compute cost derivative with respect to each estimate."""
        raise NotImplementedError()


class LeastSquares(Cost):
    """A differentiable least squares cost function for supervised learning."""

    def __call__(self, est: NumDict, tgt: NumDict, mask: NumDict) -> NumDict:
        return est.sub(tgt).pow(x=2).mul(mask)
    
    def grad(self, est: NumDict, tgt: NumDict, mask: NumDict) -> NumDict:
        return est.sub(tgt).mul(mask)


class Supervised(DualRepMixin, ErrorSignal):
    """
    An error signaling process for supervised learning.
    
    Computes and backpropagates errors based on a supervised cost function.
    """

    cost: Cost
    input: Site
    target: Site
    mask: Site

    def __init__(self, 
        name: str, 
        s: V | DV, 
        cost: Cost = LeastSquares()
    ) -> None:
        super().__init__(name)
        index, = self._init_indexes(s)
        self.cost = cost
        self.main = Site(index, {}, c=0.0)
        self.input = Site(index, {}, c=0.0)
        self.target = Site(index, {}, c=0.0)
        self.mask = Site(index, {}, c=0.0)
    
    def update(self,
        dt: timedelta = timedelta(), 
        priority: Priority = Priority.LEARNING
    ) -> None:
        exp_mask = self.mask[0].exp()
        main = self.cost.grad(self.input[0], self.target[0], exp_mask)
        self.system.schedule(self.update, 
            self.main.update(main), 
            dt=dt, priority=priority)
    

class TDError(ParamMixin, DualRepMixin, ErrorSignal):
    """
    An error signaling process for temporal difference learning.
    
    Computes and backpropagates errors based on a temporal difference function.
    """

    class Params(Atoms):
        gamma: Atom

    p: Params
    choice: Choice
    func: Callable[["TDError"], NumDict]
    main: Site
    input: Site
    reward: Site
    qvals: Site
    action: Site
    params: Site

    def next_Q(self) -> NumDict:
        qvals = self.input[0]
        return (self.main.new({})
            .sum(qvals.mul(self.action[0]).sum())
            .with_default(c=0.0))

    def expected_Q(self) -> NumDict:
        sd = self.choice.params[0][path(self.choice.p.sd)]
        qvals = self.input[0]
        pvec = qvals.scale(x=sd).exp()
        pvec = pvec.div(pvec.sum())
        expected_q = pvec.mul(qvals).sum()
        return self.main.new({}).sum(expected_q).with_default(c=0.0)

    def max_Q(self) -> NumDict:
        qvals = self.input[0]
        return (self.main.new({})
            .sum(qvals
                .max(by=self.choice.by)))
        
    def __init__(self, 
        name: str, 
        p: Family, 
        r: DV | D, 
        *,
        func: Callable[["TDError"], NumDict] = max_Q,
        gamma: float = .5,
        l: int = 1
    ) -> None:
        if l < 1:
            raise ValueError(f"Expected 1 <= l but got l == {l}.")
        super().__init__(name)
        self.func = func
        self._connect_to_choice()
        self.p, self.params = self._init_params(
            p, type(self).Params, gamma=gamma)
        idx_r, = self._init_indexes(r)
        idx_a = self.choice.main.index
        self.main = Site(idx_a, {}, c=0.0)
        self.input = Site(idx_a, {}, c=0.0, l=l + 1)
        self.qvals = Site(idx_a, {}, c=0.0, l=l)
        self.reward = Site(idx_r, {}, c=0.0, l=l)
        self.action = Site(idx_a, {}, c=0.0, l=l)

    def _connect_to_choice(self):
        try:
            sup = PROCESS.get()
        except LookupError:
            raise RuntimeError("TDError must be initalized in Choice context")
        if not isinstance(sup, Choice):
            raise RuntimeError("TDError must be initalized in Choice context")
        self.choice = sup

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
            .sum(*(rwd.sum().scale(x=gamma ** (n - 1 - t)) 
                for t, rwd in enumerate(self.reward.data)))
            .with_default(c=self.main.const)
            .sub(self.qvals[-1])
            .mul(self.action[-1])
            .neg())
        self.system.schedule(self.update,
            self.main.update(main),
            self.reward.update({}),
            self.qvals.update(self.input[0]),
            self.action.update(self.choice.main[0]),
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
